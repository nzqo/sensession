from pathlib import Path

import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from evaluation.common import BLACK, DARK_TEAL, DARK_ORANGE, RECEIVER_ORDER

from sensession.campaign import CampaignProcessor

# Discard first 200 packets or so to avoid any "warmup" of the AGC
LOWEND: int = 200


def load_data(basedir: Path) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Load the curves, meta, and CSI data from the given base directory.

    Args:
        basedir (Path): The base directory where the data files are stored.

    Returns:
        tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]: Loaded curves, meta, and CSI data.
    """
    curves = pl.read_parquet(basedir / "curves.parquet")
    meta = pl.read_parquet(basedir / "meta.parquet")
    csi = pl.read_parquet(basedir / "csi.parquet")
    return curves, meta, csi


def process_csi_data(
    csi: pl.DataFrame, meta: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Process the CSI data using CampaignProcessor.

    Args:
        csi (pl.DataFrame): The CSI dataset.
        meta (pl.DataFrame): The metadata dataset.

    Returns:
        tuple[pl.DataFrame, pl.DataFrame]: Processed CSI data and rescaled CSI data.
    """

    proc = (
        CampaignProcessor(csi, meta, lazy=False)
        # .drop_contains("collection_name", "warmup")
        # .filter("receiver_name", "asus1")
        .correct_rssi_by_agc()
        .unwrap()
        .filter("antenna_idxs", 0)
        .drop_contains("collection_name", "warmup")
        .meta_attach("curve_nr", "session_nr")
        .rescale_csi_by_rssi(column_alias="csi_rssi")
    )

    # Extract only the necessary fields for both datasets
    csi = proc.csi.drop(
        "stream_capture_num",
        "rx_antenna_capture_num",
        "collection_name",
        "stream_idxs",
        "antenna_idxs",
    )

    # csi = csi.filter(~pl.col("receiver_name").str.ends_with("2"))
    csi = csi.filter(pl.col("sequence_number") >= LOWEND).with_columns(
        pl.col("sequence_number") - LOWEND
    )

    return csi


def extract_example_timeseries(
    df: pl.DataFrame, curve_idx: int, session_idx: int, subcarrier_idx: int
) -> pl.DataFrame:
    """
    Extracts a single example timeseries from the dataset.

    Args:
        df (pl.DataFrame): The processed CSI dataset.
        curve_idx (int): The index of the curve to filter.
        session_idx (int): The index of the session to extract.
        subcarrier_idx (int): The subcarrier index to filter.

    Returns:
        pl.DataFrame: Filtered example timeseries dataframe.
    """
    # Ensure only matching curve data is used
    df = df.filter(pl.col("curve_nr") == curve_idx)

    unique_ids = df.unique("session_nr", maintain_order=True)
    session_id = unique_ids.item(session_idx, "session_nr")

    # Filter session and subcarrier BEFORE normalization
    return df.filter(
        (pl.col("session_nr") == session_id)
        & (pl.col("subcarrier_idxs") == subcarrier_idx)
    )


def extract_reference_curve(curves: pl.DataFrame, curve_idx: int) -> pl.DataFrame:
    """
    Extracts the reference curve for comparison.

    Args:
        curves (pl.DataFrame): The dataset containing reference curves.
        curve_idx (int): The index of the reference curve to extract.

    Returns:
        pl.DataFrame: The extracted reference curve with generated sequence_number.
    """
    return (
        curves.filter(pl.col("num_curve") == curve_idx)
        .explode("curve")
        .with_row_index(
            "sequence_number"
        )  # Correctly generating sequence_number dynamically
        .filter(pl.col("sequence_number") >= LOWEND)
        .with_columns(pl.col("sequence_number") - LOWEND)
    )


def normalize_curve(example_curve: pl.DataFrame) -> pl.DataFrame:
    """
    Normalizes the reference curve by dividing by its mean.

    Args:
        example_curve (pl.DataFrame): The reference curve dataset.

    Returns:
        pl.DataFrame: Normalized reference curve.
    """
    return example_curve.with_columns(
        (pl.col("curve") / pl.col("curve").mean()).alias("curve_mean")
    )


def normalize_data(df: pl.DataFrame, key: str) -> pl.DataFrame:
    """
    Normalizes CSI data across receivers.

    Args:
        df (pl.DataFrame): The dataset containing CSI values.
        key (str): The column to normalize.

    Returns:
        pl.DataFrame: Normalized CSI data.
    """
    return df.with_columns(
        [
            (pl.col(key) / pl.col(key).mean())
            .over("receiver_name")
            .alias(f"{key}_normalized_mean"),
        ]
    )


def plot_receiver_timeseries(
    example_df: pl.DataFrame, example_curve: pl.DataFrame
) -> None:
    """
    Plots the receiver timeseries including ground truth, normalized CSI values, and rescaled CSI.

    Args:
        example_df (pl.DataFrame): The extracted example session data (with original + rescaled CSI).
        example_curve (pl.DataFrame): The reference ground truth curve.
    """
    fig, axes = plt.subplots(2, 4, figsize=(18, 6), sharex=True, sharey=True)
    axes = axes.flatten()

    legend_handles = []
    legend_labels = []

    receiver_data_map = {
        r: example_df.filter(pl.col("receiver_name") == r) for r in RECEIVER_ORDER
    }

    for i, receiver in enumerate(RECEIVER_ORDER):
        ax = axes[i]
        receiver_data = receiver_data_map[receiver]

        # Plot CSI data
        sns.lineplot(
            data=receiver_data,
            x="sequence_number",
            y="csi_abs_normalized_mean",
            ax=ax,
            color=DARK_TEAL,
            label="reported",
            alpha=0.7,
            lw=0.8,
        )

        # Plot rescaled CSI data
        sns.lineplot(
            data=receiver_data,
            x="sequence_number",
            y="csi_rssi_normalized_mean",
            ax=ax,
            color=DARK_ORANGE,
            label="RSSI rescaled",
            lw=0.8,
        )

        # Plot ground truth curve (Normalized)
        sns.lineplot(
            data=example_curve,
            x="sequence_number",
            y="curve_mean",
            ax=ax,
            label="ground truth",
            color=BLACK,
            linestyle="dashed",
            lw=1.5,
        )

        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            legend_handles.extend(handles)
            legend_labels.extend(labels)

        ax.tick_params(axis="both", labelsize=16)
        ax.legend().remove()
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(f"{receiver}", fontsize=20)

    plt.subplots_adjust(hspace=0.2)  # Reduce vertical spacing

    fig.supxlabel("Sequence Number", fontsize=20, y=0.02)  # Move it closer
    fig.supylabel(r"$|\hat{H_3}|$", fontsize=20, x=0.02)

    # Single legend above the plot
    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="upper center",
        fontsize=16,
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.07),
    )

    plt.tight_layout()
    plt.savefig(
        basedir / "agc.pdf", format="pdf", bbox_inches="tight", pad_inches=0.2, dpi=300
    )


if __name__ == "__main__":
    # Define base directory using Path
    basedir = Path.cwd() / "data" / "random_curves_18dbgain"

    # Load data
    curves, meta, csi = load_data(basedir)

    # Process CSI data (both original and rescaled)
    original_df = process_csi_data(csi, meta)

    # Select example indices
    example_curve_idx = 1
    example_session_idx = 1
    example_subc = 3

    # Extract timeseries for original and rescaled CSI
    example_df = extract_example_timeseries(
        original_df, example_curve_idx, example_session_idx, example_subc
    )

    # Normalize CSI data
    example_df = normalize_data(example_df, "csi_abs")
    example_df = normalize_data(example_df, "csi_rssi")

    # Extract and normalize reference curve
    example_curve = extract_reference_curve(curves, example_curve_idx)
    example_curve = normalize_curve(example_curve)

    # Plot results
    plot_receiver_timeseries(example_df, example_curve)
