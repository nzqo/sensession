import sys
import math
from enum import Enum
from pathlib import Path

import numpy as np
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from loguru import logger
from matplotlib.lines import Line2D
from evaluation.common import (
    RECEIVER_ORDER,  # Ensure receivers appear in this given order
)
from evaluation.common import (
    SHOW_PLT,
    palette,
)
from matplotlib.colors import LinearSegmentedColormap


class Mode(Enum):
    AMP = 0
    PHASE = 1


MODE = Mode.AMP


def channel_receiver_facet_plot(df: pl.DataFrame, y: str, ylabel: str, file: Path):
    """
    Produce a facet grid plot grouping primarily by receiver. Each facet (one per receiver)
    shows channels as categorical labels, with the aggregated metric (median over subcarriers)
    per channel and distinct colors for each antenna.

    Facets are arranged in exactly two rows with a shared x-axis. An external legend is placed
    at the top of the subplots (with minimal gap) and the subplot margins are adjusted so that
    no data points are cut off.

    Args:
        df: DataFrame containing columns "channel", "receiver_name", "antenna_idx", and the metric column (y).
        y: The name of the metric to plot (e.g. "log_rsquared_diff").
        ylabel: The label for the y-axis.
        file: Output file path for saving the plot.
    """
    # Aggregate: compute the median per (receiver_name, channel, antenna_idx)
    agg_df = (
        df.group_by(["receiver_name", "channel", "antenna_idx"])
        .agg(median_value=pl.col(y).median())
        .sort("channel")
    )
    plot_df = agg_df.to_pandas()

    # Convert channel to string so that it is used as a categorical label.
    plot_df["channel"] = plot_df["channel"].astype(str)

    # Prepare a color palette for antenna_idx.
    unique_antennas = sorted(plot_df["antenna_idx"].unique())
    if palette:
        if isinstance(palette, LinearSegmentedColormap):
            color_palette = [
                palette(x) for x in np.linspace(0, 1, len(unique_antennas))
            ]
        elif isinstance(palette, (list, tuple)):
            if len(palette) < len(unique_antennas):
                color_palette = sns.color_palette("deep", len(unique_antennas))
            else:
                color_palette = palette[: len(unique_antennas)]
        else:
            try:
                color_palette = sns.color_palette(palette, len(unique_antennas))
            except Exception:
                color_palette = sns.color_palette("deep", len(unique_antennas))
    else:
        color_palette = sns.color_palette("deep", len(unique_antennas))
    color_map = {ant: col for ant, col in zip(unique_antennas, color_palette)}

    # Determine the number of facets (receivers) and compute ncol so that facets are arranged in 2 rows.
    n_receivers = len(RECEIVER_ORDER)
    ncol = math.ceil(n_receivers / 2)

    # Create the facet grid: one facet per receiver.
    g = sns.catplot(
        data=plot_df,
        x="channel",
        y="median_value",
        hue="antenna_idx",
        col="receiver_name",
        col_order=RECEIVER_ORDER,
        kind="strip",  # plots individual points; "swarm" is also an option
        dodge=False,  # points for a given (channel, receiver) group will overlap
        palette=color_palette,
        col_wrap=ncol,
        sharex=True,
        sharey=True,
        height=3,
        size=8,
        aspect=0.8,  # reduced aspect compresses horizontal space
    )
    g.set_axis_labels("Channel", ylabel, size=20)
    g.set_axis_labels("Channel", ylabel, size=20)
    g.set_titles("{col_name}", size=18)

    # 1️⃣ Tweak each subplot’s labels & ticks:
    for ax in g.axes.flatten():
        # tick labels
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=12)

    # Remove any automatically generated legend from the facets.
    if g._legend:
        g._legend.remove()

    # Increase the x-axis margin in every facet so points are not cut off.
    for ax in g.axes.flatten():
        ax.margins(x=0.05)

    # Adjust overall subplots to reserve extra space at the top.
    # Here, we set top=0.80 so subplots extend up to 80% of the figure height.
    g.figure.subplots_adjust(
        left=0.1, right=0.9, top=0.80, bottom=0.15, wspace=0.2, hspace=0.3
    )

    # Create an external legend for antenna_idx.
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"Antenna {ant}",
            markerfacecolor=color_map[ant],
            markersize=8,
        )
        for ant in unique_antennas
    ]
    # Place the legend at the top of the subplots with minimal gap.
    leg = g.figure.legend(
        handles=legend_handles,
        title="Antenna",
        loc="lower center",
        ncol=len(unique_antennas),
        bbox_to_anchor=(0.5, 0.80),
        frameon=False,
        fontsize=18,
        title_fontsize=20,
        labelcolor="#3b3b3b",
    )
    leg.get_title().set_color("#3b3b3b")

    # Apply tight_layout while reserving the top 20% of the figure for the legend.
    plt.tight_layout(rect=(0, 0, 1, 0.80))

    g.savefig(file)
    if SHOW_PLT:
        plt.show()


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")

    # Define the channels and corresponding file names.
    channels = [1, 6, 11, 36, 40, 44, 157]
    name = "single_phases" if MODE == Mode.PHASE else "single_scs"
    pre = "phase" if MODE == Mode.PHASE else "abs"

    data_dir = Path.cwd() / "data" / name
    img_dir = data_dir / "img"
    img_dir.mkdir(exist_ok=True, parents=True)

    df_list = []
    for ch in channels:
        file_path = data_dir / f"sensitivity_ch{ch}.parquet"
        logger.trace(f"Reading {file_path}")
        df_ch = pl.read_parquet(file_path)
        # Add channel information as a new column.
        df_ch = df_ch.with_columns(pl.lit(ch).alias("channel"))
        df_list.append(df_ch)
    df = pl.concat(df_list)

    # Filtering as in the original script.
    valid_iwl_indices = list(range(-28, -1, 2)) + list(range(-1, 28, 2)) + [28]
    df = df.filter(
        (pl.col("receiver_name") != "iwl5300")
        | (pl.col("modified_idx").is_in(valid_iwl_indices))
    )
    if MODE == Mode.PHASE:
        df = df.filter(~pl.col("modified_idx").is_in([-28, 28]))

    logger.trace("Data loaded and filtering applied.")

    # Compute metric transformations.
    df = df.with_columns(
        (-(1 - pl.col("r_squared")).log10()).alias("log_rsquared_diff")
    )
    df = df.with_columns(
        pearson_sensitivity=(-(1 - pl.col("correlation")).log10()),
        spearman_sensitivity=(-(1 - pl.col("spearman_corr")).log10()),
        slope_sensitivity=pl.col("slope").pow(2),
    )
    logger.trace("Metric transformations computed.")

    # Produce facet plots for each metric.
    channel_receiver_facet_plot(
        df,
        y="log_rsquared_diff",
        ylabel=r"$- \log{(1-R^2)}$",
        file=img_dir / f"{pre}-linearity.pdf",
    )
    logger.trace("Finished facet plot for linearity.")

    channel_receiver_facet_plot(
        df,
        y="pearson_sensitivity",
        ylabel="$Pearson Sensitivity$",
        file=img_dir / f"{pre}-pearson-sensitivity.pdf",
    )
    logger.trace("Finished facet plot for Pearson sensitivity.")

    channel_receiver_facet_plot(
        df,
        y="spearman_sensitivity",
        ylabel=r"$\rho$",
        file=img_dir / f"{pre}-spearman-sensitivity.pdf",
    )
    logger.trace("Finished facet plot for Spearman sensitivity.")

    channel_receiver_facet_plot(
        df,
        y="mutual_info",
        ylabel=r"MI",
        file=img_dir / f"{pre}-mi-sensitivity.pdf",
    )
    logger.trace("Finished facet plot for MI sensitivity.")

    channel_receiver_facet_plot(
        df,
        y="slope_sensitivity",
        ylabel="$Sensitivity$ (Slope Sensitivity)",
        file=img_dir / f"{pre}-slope-sensitivity.pdf",
    )
    logger.trace("Finished facet plot for slope sensitivity.")

    label = r"$\bar{D}^{phs}_k(H)$" if MODE == Mode.PHASE else r"$\bar{D}^{amp}_k(H)$"
    channel_receiver_facet_plot(
        df, y="residual", ylabel=label, file=img_dir / f"{pre}-mean-prd.pdf"
    )
    logger.trace("Finished facet plot for deviations.")
