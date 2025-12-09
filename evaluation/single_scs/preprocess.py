import gc
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import mutual_info_regression

from sensession import Database
from sensession.campaign import CampaignProcessor


def calculate_sensitivity_stats(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculates R^2, slope (a), Pearson correlation, Spearman correlation,
    mutual information, and absolute deviation for the linear relationship:
        detected_scaling = a * input_scaling + b

    Args:
        df: Input dataframe. Expected to have the following columns:
            receiver_name : Name of the receiver
            modified_idx  : Subcarrier index on which scaling was introduced
            scale_factor  : Scaling that was introduced on that subcarrier
            scale         : Detected scaling on modality

    Returns:
        A Polars DataFrame with receiver_name, modified_idx, antenna_idx,
        R^2, slope, Pearson correlation, Spearman correlation,
        mutual information, and residual.
    """
    results = []

    # compute absolute deviation column
    df = df.with_columns(dev=(pl.col("scale") - pl.col("scale_factor")).abs())

    # Loop through each receiver_name, modified_idx, antenna_idxs combination
    for (receiver_name, modified_idx, antenna_idx), group in df.group_by(
        ["receiver_name", "modified_idx", "antenna_idxs"], maintain_order=True
    ):
        # Prepare numpy arrays
        x = group["scale_factor"].to_numpy().reshape(-1, 1)  # 2D for sklearn
        y = group["scale"].to_numpy().ravel()  # 1D

        # Pearson correlation
        correlation = np.corrcoef(x.ravel(), y)[0, 1]

        # Spearman correlation
        spearman_corr, _ = spearmanr(x.ravel(), y)

        # Mutual information (nats) via k‑NN estimator
        mi = mutual_info_regression(x, y, discrete_features=False)[0]

        # Fit a linear model for R^2 and slope
        model = LinearRegression().fit(x, y)
        r2 = r2_score(y, model.predict(x))
        slope = model.coef_[0]

        # Mean absolute deviation
        deviation = group["dev"].to_numpy().mean()

        # Collect results
        results.append(
            {
                "receiver_name": receiver_name,
                "modified_idx": modified_idx,
                "antenna_idx": antenna_idx,
                "r_squared": r2,
                "slope": slope,
                "correlation": correlation,
                "spearman_corr": spearman_corr,
                "mutual_info": mi,
                "residual": deviation,
            }
        )

    # Return as Polars DataFrame
    return pl.DataFrame(results)


def preprocess(db_path: Path, modality: str):
    """
    Run preprocessing to directly read off single-subcarrier changes from the data.
    """
    num_packets = 1000

    meta = pl.DataFrame()
    csi = pl.DataFrame()
    with Database(db_path, lazy=False) as db:
        csi = db.get_csi()
        meta = db.get_meta()

    num_scs = meta.item(0, "bandwidth") // 5 * 16
    offset = num_scs // 2

    # NOTE: We dont modify the first phase because of phase preprocessing
    if "phase" in modality:
        offset -= 1

    # Little hack just for easier visualization later. Gives a common ID to each of the group sessions.
    meta = meta.with_columns(
        session_nr=(pl.col("modified_idx") + offset).cast(pl.UInt32) * 10
        + pl.col("rep_nr")
    )

    proc = CampaignProcessor(
        csi,
        meta,
        lazy=False,
    )

    redundant_cols = [
        "timestamp",
        proc.capture_index,
        proc.antenna_index,
        "stream_idxs",
        "antenna_rssi",
        "rssi",
    ]

    # Initial cleaning; Removing faulty subcarriers etc.
    proc = (
        proc.unwrap()
        .drop(*redundant_cols)
        .remove_guard_subcarriers()
        .remove_dc_subcarrier()
    )
    gc.collect()

    # Last packet is somehow buggy. Dont know why, but it happens.
    proc.csi = proc.csi.filter(pl.col("sequence_number") < 999)

    # Simple detrending and magnitude scaling
    exclude_expr = (pl.col("subcarrier_idxs") > (pl.col("modified_idx") + 1)) | (
        pl.col("subcarrier_idxs") < (pl.col("modified_idx") - 1)
    )

    # fmt: off
    gc.collect()
    proc = proc.meta_attach("modified_idx")

    if "abs" in modality:
        proc = proc.scale_magnitude(exclude_expr=exclude_expr)
    else:
        proc = proc.detrend_phase()


    # Session based equalization
    proc.csi.drop("modified_idx")
    gc.collect()

    if "abs" in modality:
        proc = proc.equalize_magnitude()
    else:
        proc = proc.equalize_phase()
    proc = proc.meta_attach("modified_idx")
    # fmt: on

    proc = proc.drop_contains("collection_name", "warmup").drop("collection_name")

    # -----------------------------------------------------------------------
    # Preprocessing is done. Time to get to some statistics and analysis of
    # sensitivity.
    csi, meta = proc.get()

    # Define what we consider as the "detected scaling", either phase or amplitude.
    csi = csi.rename({modality: "scale"})

    # We sweep over the different scaling factors, so they are enumerated by the sequence number
    factor = 1  # if modality == "csi_abs" else np.pi (should be fixed in scale_range on disk! just a debug reminder)
    diff = meta.item(0, "scale_range") * factor / num_packets

    # fmt: off
    csi = (csi
        .with_columns(scale_factor=pl.col("sequence_number") * diff)
        .filter(pl.col("subcarrier_idxs") == pl.col("modified_idx"))
        .drop("subcarrier_idxs")
    )

    csi.write_parquet(db_path / "processed.parquet")
    return csi


def run(db_path: Path, modality: str, prefix: str, channel):
    """
    Run one statistics calculation round (for one channel and modality)
    """
    # Preprocessing is expensive and takes long (up to a few hours)
    # Therefore, we cache the processed results in each of the experiment
    # directories for faster recompute of sensitivity stats.
    procfile = db_path / "processed.parquet"
    if procfile.is_file():
        csi = pl.read_parquet(procfile)
    else:
        csi = preprocess(db_path, modality)

    # preprocessed data is already filtered to modified_idx now
    r_sq = calculate_sensitivity_stats(csi)

    # shared results directory per prefix
    results_dir = data_root / f"{prefix}_results"
    results_dir.mkdir(exist_ok=True)

    out_path = results_dir / f"sensitivity_ch{channel}.parquet"
    r_sq.write_parquet(out_path)


if __name__ == "__main__":
    channels = ["01", "06", "11", "36", "40", "44", "157"]
    data_root = Path.cwd() / "data"

    # (modality, collection_name_prefix)
    experiments = [
        ("csi_abs", "single_scs"),
        ("csi_phase", "single_phases"),
    ]

    for modality, prefix in experiments:
        for ch in channels:
            collection_name = f"{prefix}_ch{ch}"
            db_path = data_root / collection_name
            run(db_path, modality, prefix, ch)
            gc.collect()
