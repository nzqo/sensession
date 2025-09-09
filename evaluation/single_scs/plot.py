"""
Single subcarrier precoding sweep experiment evaluation plots.
"""

import sys
from enum import Enum
from pathlib import Path

import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from loguru import logger
from matplotlib.cm import ScalarMappable
from evaluation.common import (
    SHOW_PLT,
    RECEIVER_ORDER,
    palette,
    fmlp_cmap,
    subcarrier_barplot,
    subcarrier_dual_barplot,
)
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator


class Mode(Enum):
    AMP = 0
    PHASE = 1


def plot_mae(df: pl.DataFrame, img_dir: Path):
    # Calculate MAE using Polars native API
    results = df.group_by(["receiver_name", "scale_factor"]).agg(
        (pl.col("scale_factor") - pl.col("scale")).abs().mean().alias("MAE")
    )

    num_colors = results["receiver_name"].n_unique()
    discrete_palette = [palette(i / (num_colors - 1)) for i in range(num_colors)]

    receivers = df.get_column("receiver_name").unique().to_list()
    receivers = [rcv for rcv in RECEIVER_ORDER if rcv in receivers]

    # Create the plot with an appealing color palette
    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=results,
        x="scale_factor",
        y="MAE",
        hue="receiver_name",
        dodge=True,
        saturation=0.95,
        palette=discrete_palette,
        hue_order=receivers,
        zorder=3,
    )

    plt.ylabel("Mean Absolute Error (MAE)", fontsize=22)
    plt.xlabel("Target Scaling", fontsize=22)
    plt.legend(
        title="Device",
        title_fontsize=20,
        fontsize=18,
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        frameon=True,
    )
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig(
        img_dir / "mae.pdf",
        format="pdf",
        bbox_inches="tight",
        dpi=300,
    )

    if SHOW_PLT:
        plt.show()


def plot_topk_mae(df: pl.DataFrame, top_k: int, img_dir: Path):
    receivers = df.get_column("receiver_name").unique().to_list()
    receivers = [rcv for rcv in RECEIVER_ORDER if rcv in receivers]
    # Calculate the MAE using the N smallest deviations
    results = (
        df.with_columns(
            (pl.col("scale_factor") - pl.col("scale")).abs().alias("deviation")
        )
        .sort(
            ["receiver_name", "scale_factor", "deviation"]
        )  # Sort deviations per group
        .group_by(["receiver_name", "scale_factor"], maintain_order=True)
        .agg(
            pl.col("deviation")
            .head(top_k)  # Keep only the smallest N deviations
            .mean()
            .alias("MAE_N_smallest")
        )
    )

    # Plot the results
    num_colors = results["receiver_name"].n_unique()
    discrete_palette = [palette(i / (num_colors - 1)) for i in range(num_colors)]

    plt.figure(figsize=(14, 8))
    sns.barplot(
        data=results,
        x="scale_factor",
        y="MAE_N_smallest",
        hue="receiver_name",
        dodge=True,
        saturation=0.95,
        palette=discrete_palette,
        hue_order=receivers,
        zorder=3,
    )

    plt.ylabel("Mean Absolute Error (MAE)", fontsize=22)
    plt.xlabel("Target Scaling", fontsize=22)
    plt.legend(
        title="Device",
        title_fontsize=20,
        fontsize=18,
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        frameon=True,
    )
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()

    plt.savefig(
        img_dir / f"top_{top_k}_mae.pdf",
        format="pdf",
        bbox_inches="tight",
        dpi=300,
    )

    if SHOW_PLT:
        plt.show()


def plot_linearity(df: pl.DataFrame, img_dir: Path):
    """
    Args:
        r-squared dataframe with columns:
            receiver_name : Name of the device
            modified_idx  : Subcarrier on which scaling was introduced
            r_squared     : r-squared measure how well line could be fit
    """
    # Transform R^2 values to highlight differences closer to 1 (logarithmic transformation)
    df = df.with_columns((-(1 - df["r_squared"]).log10()).alias("log_rsquared_diff"))

    subcarrier_barplot(
        df,
        y="log_rsquared_diff",
        ylabel=r"$- \log{(1-R^2)}$",
        file=img_dir / "linearity.pdf",
    )


def plot_crosstalk(df: pl.DataFrame, img_dir: Path):
    """
    Args:
        crosstalk dataframe with columns:
            receiver_name     : Name of the device
            modified_idx      : Subcarrier on which scaling was introduced
            scale_factor      : The scale factor that was applied to subcarrier modified_idx
            average_crosstalk : Crosstalk (averaged over all subcarriers in a single packet)
    """
    # Transform R^2 values to highlight differences closer to 1 (logarithmic transformation)
    df = df.group_by("receiver_name", "modified_idx").agg(
        pl.col("average_crosstalk").mean()
    )

    subcarrier_barplot(
        df, y="average_crosstalk", ylabel="$Crosstalk$", file=img_dir / "crosstalk.pdf"
    )


def plot_crosstalk_faceted(df: pl.DataFrame, img_dir: Path):
    """
    Args:
        crosstalk dataframe with columns:
            receiver_name     : Name of the device
            modified_idx      : Subcarrier on which scaling was introduced
            scale_factor      : The scale factor that was applied to subcarrier modified_idx
            average_crosstalk : Crosstalk (averaged over all subcarriers in a single packet)
    """
    receivers = df.get_column("receiver_name").unique().to_list()
    receivers = [rcv for rcv in RECEIVER_ORDER if rcv in receivers]

    # Compute the median for each (receiver_name, modified_idx)
    med_crosstalk = df.group_by(["receiver_name", "modified_idx"]).agg(
        pl.col("average_crosstalk").median().alias("med_crosstalk")
    )

    # Normalize the medians and map to colors
    vals = med_crosstalk["med_crosstalk"].to_list()
    vmin: float = min(vals)
    vmax: float = max(vals)
    norm = Normalize(vmin=vmin, vmax=vmax)
    norm = Normalize(
        vmin=vmin,
        vmax=vmax,
    )

    # Create a mapping from (receiver_name, modified_idx) to colors
    color_map = {
        (row["receiver_name"], row["modified_idx"]): fmlp_cmap(
            norm(row["med_crosstalk"])
        )
        for row in med_crosstalk.to_dicts()
    }

    # Create the FacetGrid
    g = sns.FacetGrid(
        data=df.to_pandas(),
        col="receiver_name",
        col_wrap=2,
        height=3,
        aspect=1.5,
        sharey=True,
        col_order=receivers,
    )

    logger.trace(f"Plotting faceted crosstalk for: {df}")

    # Map the boxplot
    g.map_dataframe(
        sns.boxplot,
        x="modified_idx",
        y="average_crosstalk",
        showfliers=False,
        palette=None,
    )

    # Match each patch to the correct modified_idx and receiver_name
    for ax, receiver_name in zip(g.axes.flat, g.col_names):
        # Get unique modified_idx values for this receiver
        mod_idx_values = (
            df.filter(pl.col("receiver_name") == receiver_name)["modified_idx"]
            .unique()
            .to_list()
        )
        for patch, mod_idx in zip(ax.patches, mod_idx_values):
            if (receiver_name, mod_idx) in color_map:
                patch.set_facecolor(color_map[(receiver_name, mod_idx)])

    # Customize axis labels and titles
    g.set_axis_labels("Modified Index", "$Crosstalk$", fontsize=16)
    g.set_titles("Receiver: {col_name}")

    for ax in g.axes.flat:
        ax.title.set_fontsize(18)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Add a slimmer colorbar
    sm = ScalarMappable(norm=norm, cmap=fmlp_cmap)
    sm.set_array([])
    g.figure.colorbar(
        sm,
        ax=g.axes,
        orientation="vertical",
        fraction=0.02,  # Narrower bar
        pad=0.05,  # Padding from the plots
        aspect=40,  # Longer bar (controls height-to-width ratio)
    )

    plt.savefig(
        img_dir / "faceted-crosstalk.pdf", format="pdf", bbox_inches="tight", dpi=300
    )

    if SHOW_PLT:
        plt.show()


def plot_sensitivity(df: pl.DataFrame, img_dir: Path):
    """
    Args:
        df: Dataframe including columns
            `correlation`, `spearman_corr`, and `mutual_info`
    """
    df = df.with_columns(
        pearson_sensitivity=(-(1 - df["correlation"]).log10()),
        spearman_sensitivity=(-(1 - df["spearman_corr"]).log10()),
        slope_sensitivity=1 - ((1 - df["slope"]).abs()),
        mi_sensitivity=df["mutual_info"],  # <— new line
    )

    subcarrier_barplot(
        df,
        y="pearson_sensitivity",
        ylabel="$Pearson\\ Sensitivity$",
        file=img_dir / "pearson-sensitivity.pdf",
    )
    subcarrier_barplot(
        df,
        y="spearman_sensitivity",
        ylabel=r"$\rho$",
        file=img_dir / "spearman-sensitivity.pdf",
    )
    subcarrier_barplot(
        df,
        y="slope_sensitivity",
        ylabel="$Sensitivity$",
        file=img_dir / "slope-sensitivity.pdf",
    )
    # ---- new plot for MI ----
    subcarrier_barplot(
        df,
        y="mi_sensitivity",
        ylabel="MI",
        file=img_dir / "mi-sensitivity.pdf",
    )


def plot_deviations(df: pl.DataFrame, mode: Mode, img_dir: Path):
    """
    Args:
        df: Dataframe including correlation statistics (named `correlation` and `spearman_corr`)
    """
    label = r"$\bar{D}^{phs}_k(H)$" if mode == Mode.PHASE else r"$\bar{D}^{amp}_k(H)$"
    subcarrier_dual_barplot(
        df,
        y="residual",
        ylabel=label,
        file=img_dir / "mean-prd.pdf",
    )


def main(mode: Mode, exp_t: str):
    # See 802.11n-2009 Table 7-25f
    if "80mhz" in exp_t:
        valid_iwl_indices = []  # iwl cant capture 80.
        edge = 122
    elif "40mhz" in exp_t:
        valid_iwl_indices = list(range(-58, 59, 4))
        edge = 58
    else:
        valid_iwl_indices = list(range(-28, -1, 2)) + list(range(-1, 28, 2)) + [28]
        edge = 28

    name = "single_phases" if mode == Mode.PHASE else "single_scs"
    data_dir = Path.cwd() / "data" / name
    img_dir = data_dir / "img" / f"{exp_t}"
    img_dir.mkdir(exist_ok=True, parents=True)

    # --- Plot rsquared as measure of linearity
    df = pl.read_parquet(data_dir / f"sensitivity_{exp_t}.parquet")
    df = df.filter(pl.col("antenna_idx") == 0)

    df = df.filter(
        (df["receiver_name"] != "iwl5300")
        | (df["modified_idx"].is_in(valid_iwl_indices))
    )
    if mode == Mode.PHASE:
        df = df.filter(~pl.col("modified_idx").is_in([-edge, edge]))

    logger.trace("Read data.")

    plot_linearity(df, img_dir)
    logger.trace("Finished rsquared plots")

    plot_sensitivity(df, img_dir)
    logger.trace("Finished sensitivity plots")

    plot_deviations(df, mode, img_dir)
    logger.trace("Finished plotting deviations")

    # --- Plot crosstalk
    # df = pl.read_parquet(data_dir / "crosstalk.parquet")
    # df = df.filter(
    #     (df["receiver_name"] != "iwl5300")
    #     | (df["modified_idx"].is_in(valid_iwl_indices))
    # )
    # if MODE == Mode.PHASE:
    #     df = df.filter(~pl.col("modified_idx").is_in([-edge, edge]))

    # plot_crosstalk(df)
    # logger.trace("Finished crosstalk plot")

    # plot_crosstalk_faceted(df)
    # logger.trace("Finished faceted plot")

    # NOTE: This plot doesnt make sense for the scale sweeps, only when a fixed
    # set is used.
    # --- Plot MAE; deviation between theoretical and measured channel scaling
    # Load the dataframe
    # df = pl.read_parquet(data_dir / "means.parquet")
    # df = df.filter(
    #     (df["receiver_name"] != "iwl5300")
    #     | (df["modified_idx"].is_in(valid_iwl_indices))
    # )
    # plot_mae(df)

    # # Filter out invalid scale_factor values
    # df = df.filter(df["scale_factor"] < 2).filter(pl.col("device") != "ax210")
    # # Modify the "ax210" device's scale
    # df = df.with_columns(
    #     pl.when(df["device"] == "ax210")
    #     .then(df["scale"] * 2 - 1)
    #     .otherwise(df["scale"])
    #     .alias("scale")
    # )
    # plot_topk_mae(df, 20)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    exp_t = "ch11"  # "80mhz"

    main(Mode.AMP, exp_t)
    main(Mode.PHASE, exp_t)
