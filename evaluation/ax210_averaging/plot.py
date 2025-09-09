from pathlib import Path

import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from evaluation.common import DARK_ORANGE

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
    exclude_expr = (pl.col("subcarrier_idxs") < 24 - 33) | (
        pl.col("subcarrier_idxs") > 40 - 31
    )

    # fmt: off
    proc = (proc
        .correct_rssi_by_agc()
        .unwrap()
        .filter("antenna_idxs", 0)
        .detrend_phase()
        .scale_magnitude(exclude_expr = exclude_expr)
        .equalize_magnitude()
        .equalize_phase()
        .drop_contains("collection_name", "warmup")
    )
    csi = proc.csi
    csi = csi.filter(pl.col("sequence_number") < 2).filter(pl.col("receiver_name").is_in(["x310", "ax210"]))
    csi = csi.drop("stream_idxs", "antenna_idxs", "rssi", "antenna_rssi", "timestamp", "stream_capture_num", "rx_antenna_capture_num", "capture_num", "collection_name")
    meta = proc.meta
    # fmt: on

    return csi, meta


def plot_csi_scatter(df, schedule_name):
    # Filter data
    df = df.filter(pl.col("schedule_name") == schedule_name)

    # Plot
    plt.figure(figsize=(12, 4))
    custom_palette = [DARK_ORANGE, "#636363"]

    ax = sns.scatterplot(
        data=df,
        x="subcarrier_idxs",
        y="csi_abs",
        hue="receiver_name",
        palette=custom_palette,
        alpha=0.6,
        s=200,
    )

    plt.xlabel("Subcarrier", fontsize=22)
    plt.ylabel(r"$|H^{eq}|$", fontsize=22)

    ax.tick_params(axis="x", labelrotation=60)

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.legend(
        fontsize=20,
        loc="upper left",
        bbox_to_anchor=(0, 1),
        frameon=True,
    )

    plt.tight_layout()
    plt.savefig(
        data_dir / f"{schedule_name.replace('_', '-')}.pdf",
        format="pdf",
        bbox_inches="tight",
        dpi=300,
    )


if __name__ == "__main__":
    data_dir = Path.cwd() / "data" / "ax210_averaging"

    # --- Plot rsquared as measure of linearity
    df = pl.read_parquet(data_dir / "csi.parquet")
    meta = pl.read_parquet(data_dir / "meta.parquet")

    df, meta = preprocess(df, meta)
    df = df.join(
        meta.select("meta_id", "schedule_name"), on="meta_id", maintain_order="left"
    )
    plot_csi_scatter(df, "magnitude_block")
