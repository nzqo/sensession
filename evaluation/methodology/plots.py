from pathlib import Path

import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from evaluation.common import tgo_cmap

from sensession.campaign import CampaignProcessor


def preprocess(csi: pl.DataFrame, meta: pl.DataFrame):
    """
    Preprocessing steps
    """

    proc = CampaignProcessor(
        csi,
        meta,
        lazy=False,
    )

    # fmt: off
    proc = (proc
        .correct_rssi_by_agc()
        .unwrap()
        .remove_guard_subcarriers()
        .remove_dc_subcarrier()
        .scale_magnitude(column_alias="csi_abs_scaled")
        .detrend_phase(column_alias="csi_phase_detrended")
        .equalize_magnitude(normed_column="csi_abs_scaled", column_alias="csi_abs_eq")
        .equalize_phase(normed_column="csi_phase_detrended", column_alias="csi_phase_eq")
        .drop_contains("collection_name", "run_run")
    )
    # fmt: on

    csi, meta = proc.get()

    # Remove unused columns for simpler working here
    meta = meta.drop(
        "collection_stop",
        "channel",
        "bandwidth",
        "antenna_idxs",
        "stream_idxs",
        "campaign_name",
        "schedule_name",
        "collection_name",
    )

    csi = csi.drop(
        "stream_capture_num",
        "rx_antenna_capture_num",
        "timestamp",
        "timestamp",
        "rssi",
        "antenna_rssi",
        "collection_name",
        "antenna_idxs",
        "stream_idxs",
    )

    # Keep only first six packets for visualization
    csi = csi.group_by(["meta_id", "subcarrier_idxs"], maintain_order=True).head(5)

    return csi, meta


def plot_modality(df: pl.DataFrame, modality: str, ylabel: str, output_file: Path):
    """
    Args:
        df (pl.DataFrame): Dataframe with columns including:
            subcarrier_idxs  : Subcarrier on which scaling was introduced
            modality         : The measurement to be plotted (specified by argument)
        modality (str): The column to be plotted on the y-axis
        ylabel (str): Label for the y-axis
        output_file (Path): File path to save the plot
    """

    # Map sequence_number to enumerated packet numbers
    unique_sequences = (
        df.select("sequence_number").unique().to_series().sort().to_list()
    )
    sequence_to_packet = {seq: i for i, seq in enumerate(unique_sequences)}

    df = df.with_columns(
        pl.col("sequence_number").replace_strict(sequence_to_packet).alias("packet")
    )

    # Assign colors based on packets
    num_packets = len(unique_sequences)

    color_palette = {
        pkt: tgo_cmap(i / (num_packets - 1)) for i, pkt in enumerate(range(num_packets))
    }

    # Plot with lines and markers
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(
        data=df,
        x="subcarrier_idxs",
        y=modality,
        hue="packet",
        palette=color_palette,
        legend=True,
        linewidth=2.5,
        dashes=False,
    )

    # Customize legend to show "Packet"
    ax.legend(title="Packet", fontsize=20, title_fontsize=24)

    # Customize labels and titles
    ax.set_xlabel("Subcarrier", fontsize=28)
    ax.set_ylabel(ylabel, fontsize=28)
    ax.tick_params(axis="x", labelrotation=60, labelsize=22)
    ax.tick_params(axis="y", labelsize=22)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    if modality != "csi_phase":
        ax.set_ylim(-1.1, 1.1)
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.3)
    ax.grid(False, which="minor")  # no minor grid

    plt.tight_layout()

    plt.savefig(
        output_file,
        format="pdf",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


def plot_profiles(df: pl.DataFrame):
    img_dir = data_dir / "images"
    img_dir.mkdir(exist_ok=True)

    for receiver_name, group_df in df.group_by("receiver_name", maintain_order=True):
        receiver_name = receiver_name[0]

        plot_modality(
            group_df,
            "csi_phase",
            ylabel=r"$\arg(\hat{H})$",
            output_file=img_dir / f"raw-{receiver_name}.pdf",
        )

        plot_modality(
            group_df,
            "csi_phase_detrended",
            ylabel=r"$\arg(\hat{H}^N)$",
            output_file=img_dir / f"normalized-{receiver_name}.pdf",
        )

        plot_modality(
            group_df,
            "csi_phase_eq",
            ylabel=r"$\arg(\hat{H}^{eq})$",
            output_file=img_dir / f"equalized-{receiver_name}.pdf",
        )


if __name__ == "__main__":
    valid_iwl_indices = list(range(-28, -1, 2)) + list(range(-1, 28, 2)) + [28]
    data_dir = Path.cwd() / "data" / "methodology"

    # --- Plot rsquared as measure of linearity
    df = pl.read_parquet(data_dir / "csi.parquet")
    meta = pl.read_parquet(data_dir / "meta.parquet")

    df, meta = preprocess(df, meta)

    plot_profiles(df)
