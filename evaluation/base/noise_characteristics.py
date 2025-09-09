from pathlib import Path

import numpy as np
import polars as pl
from evaluation.base.plots import preprocess


def compute_metrics(subdf: pl.DataFrame, modality: str) -> pl.DataFrame:
    # fmt: off
    stddev = pl.col(modality).std().alias("stddev")
    iqr = (pl.col(modality).quantile(0.75) - pl.col(modality).quantile(0.25)).alias("iqr")
    skew = pl.col(modality).skew().alias("skew")
    kurtosis = pl.col(modality).kurtosis().alias("kurtosis")
    outliers = (
        (
            (pl.col(modality) < (pl.col(modality).quantile(0.25) - 1.5 * (pl.col(modality).quantile(0.75) - pl.col(modality).quantile(0.25)))) |
            (pl.col(modality) > (pl.col(modality).quantile(0.75) + 1.5 * (pl.col(modality).quantile(0.75) - pl.col(modality).quantile(0.25))))
        ).sum()
    ).alias("outliers")
    # fmt: on

    metrics = (
        subdf.group_by("subcarrier_idxs")
        .agg([stddev, iqr, skew, kurtosis, outliers])
        .drop("subcarrier_idxs")
        .fill_nan(0)
        .mean()
    )

    outlier_ratio = metrics["outliers"] / subdf.height

    return metrics.drop("outliers").with_columns(
        [
            pl.lit(subdf["receiver_name"].first()).alias("receiver_name"),
            pl.lit(outlier_ratio).alias("outlier_ratio"),
        ]
    )


def remove_csi_outliers(
    csi: pl.DataFrame, threshold: float = 30, verbose: bool = True
) -> pl.DataFrame:
    """
    Remove outlier CSI captures using Mahalanobis distance.

    For each receiver, the function pivots the CSI DataFrame so that each row is a capture
    (with its multi-dimensional CSI vector), computes the Mahalanobis distance of each capture
    from the group mean, and removes captures with a distance above the given threshold.

    Parameters:
        csi (pl.DataFrame): DataFrame containing CSI data with columns:
            - "capture_num": capture identifier,
            - "receiver_name": receiver identifier,
            - "subcarrier_idxs": index of the CSI vector component,
            - "csi_abs": CSI measurement,
            - "meta_id": metadata identifier.
        threshold (float): Maximum allowed Mahalanobis distance; captures with a higher distance are removed.
        verbose (bool): If True, prints the number of outliers removed per receiver.

    Returns:
        pl.DataFrame: The original CSI DataFrame with outlier captures removed.
    """
    # Pivot so that each row is a capture (with its receiver) and columns are CSI vector components.
    df_pivot = csi.pivot(
        index=["capture_num", "receiver_name"], on="subcarrier_idxs", values="csi_abs"
    ).sort(["receiver_name", "capture_num"])

    # Identify the CSI feature columns.
    feature_cols = [
        col for col in df_pivot.columns if col not in ["capture_num", "receiver_name"]
    ]

    # Collect allowed (non-outlier) capture identifiers.
    allowed_capture_nums = []
    allowed_receiver_names = []

    # Process each receiver group separately.
    receivers = df_pivot["receiver_name"].unique().to_list()
    for rec in receivers:
        df_rec = df_pivot.filter(pl.col("receiver_name") == rec)
        X = df_rec.select(feature_cols).to_numpy()
        capture_nums = df_rec["capture_num"].to_list()

        # Compute mean and covariance.
        mean_vector = np.mean(X, axis=0)
        cov_matrix = np.cov(X, rowvar=False)
        try:
            inv_cov = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6
            inv_cov = np.linalg.inv(cov_matrix)

        # Compute Mahalanobis distances for all captures.
        delta = X - mean_vector
        distances = np.sqrt(np.einsum("ij,jk,ik->i", delta, inv_cov, delta))

        # Identify inliers.
        inlier_mask = distances <= threshold
        n_total = len(capture_nums)
        n_inliers = int(np.sum(inlier_mask))
        n_outliers = n_total - n_inliers
        if verbose:
            print(
                f"Receiver '{rec}': {n_outliers} outlier(s) removed out of {n_total} capture(s)."
            )

        # Collect allowed capture identifiers.
        for idx, is_inlier in enumerate(inlier_mask):
            if is_inlier:
                allowed_capture_nums.append(capture_nums[idx])
                allowed_receiver_names.append(rec)

    # Create a DataFrame of allowed capture identifiers.
    allowed_df = pl.DataFrame(
        {"capture_num": allowed_capture_nums, "receiver_name": allowed_receiver_names}
    )

    # Return the original CSI data with only allowed captures.
    return csi.join(allowed_df, on=["capture_num", "receiver_name"], how="semi")


def compute_csi_noise_metrics(df: pl.DataFrame):
    abs_metrics = (
        df.group_by("receiver_name", maintain_order=True)
        .map_groups(lambda g: compute_metrics(g, "csi_abs_eq"))
        .sort("iqr")
    )
    phs_metrics = (
        df.group_by("receiver_name", maintain_order=True)
        .map_groups(lambda g: compute_metrics(g, "csi_phase_eq"))
        .sort("iqr")
    )

    print("Magnitude noise metrics:")
    print(abs_metrics)
    print("Phase noise metrics:")
    print(phs_metrics)


def compute_aggregate_noise_metrics(df: pl.DataFrame):
    """
    Compute the variance of the average and the variance of the sum of squares
    of CSI amplitudes (csi_abs_eq) across subcarriers for each receiver.

    For each capture (identified by "capture_num" and "receiver_name"), the method
    computes:
      - the average amplitude across subcarriers, and
      - the sum of squares of amplitudes across subcarriers.

    It then groups these per-capture aggregates by receiver and calculates:
      - the variance of the averages ("var_avg"), and
      - the variance of the sums of squares ("var_sum_sq").

    Parameters:
        df (pl.DataFrame): A DataFrame with columns:
            - "capture_num": capture identifier,
            - "receiver_name": receiver identifier,
            - "subcarrier_idxs": subcarrier index,
            - "csi_abs_eq": amplitude measurement (already preprocessed).

    Returns:
        pl.DataFrame: A DataFrame with one row per receiver containing:
            - "receiver_name",
            - "var_avg": variance of the per-capture average amplitudes,
            - "var_sum_sq": variance of the per-capture sum of squares.
    """
    # First, aggregate per capture: compute average and sum of squares of amplitudes.
    capture_agg = df.group_by("receiver_name", "capture_num", maintain_order=True).agg(
        pl.col("csi_abs_eq").mean().alias("avg_amp"),
        (pl.col("csi_abs_eq") ** 2).sum().alias("sum_sq_amp"),
    )

    # Then, for each receiver, compute the variance of these per-capture aggregates.
    receiver_metrics = capture_agg.group_by("receiver_name", maintain_order=True).agg(
        pl.col("avg_amp").var().alias("var_avg"),
        pl.col("sum_sq_amp").var().alias("var_sum_sq"),
    )

    print(receiver_metrics)


if __name__ == "__main__":
    valid_iwl_indices = list(range(-28, -1, 2)) + list(range(-1, 28, 2)) + [28]
    data_dir = Path.cwd() / "data" / "base_campaign_ch11"

    # --- Plot rsquared as measure of linearity
    df = pl.read_parquet(data_dir / "csi.parquet")
    meta = pl.read_parquet(data_dir / "meta.parquet")

    df, meta = preprocess(df, meta)
    first_session = df.unique("collection_start", maintain_order=True).item(
        1, "collection_start"
    )
    df = df.filter(pl.col("collection_start") == first_session)
    compute_csi_noise_metrics(df)
    compute_aggregate_noise_metrics(df)

    print("Cleaning dataframe of outliers using Mahalanobis Distance....")
    print("Recalculating noise metrics")
    cleaned_df = remove_csi_outliers(df)
    compute_csi_noise_metrics(cleaned_df)
