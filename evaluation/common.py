"""
Common parts for evaluation
"""

from pathlib import Path

import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SHOW_PLT = False

custom_style = {
    "axes.grid": True,  # Enable gridlines
    "grid.color": "0.7",  # Darker gridlines for better visibility
    "grid.linestyle": "--",  # Dashed gridlines
    "axes.facecolor": "white",  # White plot background
    "axes.edgecolor": "0.7",  # Light gray frame
    "axes.linewidth": 0.8,  # Thinner frame lines
    "xtick.color": "0.2",  # Darker tick color
    "ytick.color": "0.2",  # Darker tick color
    "xtick.direction": "out",  # Ticks pointing outward
    "ytick.direction": "out",
    "axes.spines.top": True,  # Show top spine
    "axes.spines.right": True,  # Show right spine
    "axes.spines.left": True,  # Show left spine
    "axes.spines.bottom": True,  # Show bottom spine
    "xtick.major.size": 0,  # Remove small tick lines
    "ytick.major.size": 0,  # Remove small tick lines
}

plt.rcParams.update(custom_style)


RECEIVER_ORDER = [
    "x310",
    "qca",
    "ax210",
    "iwl5300",
    "asus1",
    "asus2",
    "ESP1",
    "ESP2",
]


# ───────────────────────────────────
# 1.  Color palettes
# ───────────────────────────────────

# --> Basic teal over gray to orange color palette
DARK_TEAL = "#0d7d87"
LIGHT_TEAL = "#3aa0a6"
LIGHT_GRAY = "#9c9c9c"
LIGHT_ORAGE = "#ffb84d"
DARK_ORANGE = "#FFA500"
WHITE = "#ffffff"
BLACK = "#000000"

tgo_palette = [
    DARK_TEAL,
    LIGHT_TEAL,  # saturated teal branch
    LIGHT_GRAY,  # neutral light grey
    LIGHT_ORAGE,  # saturated orange branch
    DARK_ORANGE,
]


tgo_cmap = LinearSegmentedColormap.from_list(
    "teal_grey_orange",
    tgo_palette,
    N=256,
)

tgo_cmap_rev = LinearSegmentedColormap.from_list(
    "teal_grey_orange",
    list(reversed(tgo_palette)),
    N=256,
)

# --> Custom grayscale palette for subcarrier shade
custom_gray = LinearSegmentedColormap.from_list("custom_gray", ["#bbbbbb", "#444444"])
palette = custom_gray

# --> Teal white orange palette
fmlp_cmap = LinearSegmentedColormap.from_list(
    "teal_white_orange", [(0.0, DARK_ORANGE), (0.5, WHITE), (1.0, DARK_TEAL)], N=256
)

# --> Vice (legacy)
vice_cmap = LinearSegmentedColormap.from_list(
    "custom_cmap", ["#009999", "#F5F5F5", "#CDA2BE"]
)

# Whatever this is :)
fmlp_cmap_2 = LinearSegmentedColormap.from_list("custom_cmap", ["#009999", "#CDA2BE"])


def subcarrier_barplot(df: pl.DataFrame, y: str, ylabel: str, file: Path):
    """
    Args:
        df: Dataframe
        y: name of y-value column
        title: title of the figure
        filename: filename to save figure to (ending appended automatically)
    """
    # Get unique receivers
    unique_receivers = df.select("receiver_name").unique().to_series().to_list()
    unique_receivers = [rcv for rcv in RECEIVER_ORDER if rcv in unique_receivers]

    # Create the plot
    plt.figure(figsize=(14, 6))

    # Use Seaborn to create a grouped barplot
    sns.barplot(
        data=df,
        x="receiver_name",
        y=y,
        hue="modified_idx",
        palette=palette,
        dodge=True,
        saturation=0.95,
        order=unique_receivers,
        zorder=3,
    )

    plt.ylabel(ylabel=ylabel, fontsize=26)
    plt.xlabel("Receiver Name", fontsize=26)
    plt.legend(
        title="Subcarrier",
        title_fontsize=24,
        fontsize=22,
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        frameon=True,
    )
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.tight_layout()
    plt.savefig(
        file,
        format="pdf",
        bbox_inches="tight",
        dpi=300,
    )

    if SHOW_PLT:
        plt.show()


def subcarrier_dual_barplot(
    df: pl.DataFrame, y: str, ylabel: str, file: Path, outlier: str = "ax210"
):
    """
    Creates two subplots: One with all receiver classes except the specified outlier,
    and another with only the outlier. A shared legend is added to the right.

    Args:
        df: Dataframe
        y: Name of y-value column
        ylabel: Label for the y-axis
        file: Filename to save the figure to
        outlier: The receiver name to be displayed in the right subplot (default: "ax210")
    """

    # Get unique receivers
    unique_receivers = df.select("receiver_name").unique().to_series().to_list()
    unique_receivers = [rcv for rcv in RECEIVER_ORDER if rcv in unique_receivers]

    # Ensure the specified outlier exists in the dataset
    if outlier not in unique_receivers:
        raise ValueError(f"Specified outlier '{outlier}' not found in dataset.")

    # Define the left and right split
    left_receivers = [r for r in unique_receivers if r != outlier]

    # Create subplots with ratio 7:1
    fig, axes = plt.subplots(
        1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [7, 1]}
    )

    # Left plot (all except outlier)
    sns.barplot(
        data=df.filter(pl.col("receiver_name").is_in(left_receivers)),
        x="receiver_name",
        y=y,
        hue="modified_idx",
        palette=palette,
        dodge=True,
        saturation=0.95,
        order=left_receivers,
        ax=axes[0],
        zorder=3,
    )

    axes[0].set_ylabel(ylabel, fontsize=32)
    axes[0].set_xlabel("Receiver Name", fontsize=32)
    axes[0].tick_params(axis="x", rotation=30, labelsize=26)
    axes[0].tick_params(axis="y", labelsize=26)
    axes[0].get_legend().remove()  # Remove subplot legend

    # Right plot (specific outlier)
    sns.barplot(
        data=df.filter(pl.col("receiver_name") == outlier),
        x="receiver_name",
        y=y,
        hue="modified_idx",
        palette=palette,
        dodge=True,
        saturation=0.95,
        ax=axes[1],
        zorder=3,
    )

    axes[1].set_ylabel("")  # No label to avoid redundancy
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=30, labelsize=26)
    axes[1].tick_params(axis="y", labelsize=26)
    # Remove y-ticks for the single-receiver plot
    axes[1].get_legend().remove()  # Remove subplot legend

    # Add a shared legend outside the plots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Subcarrier",
        title_fontsize=22,
        fontsize=20,
        loc="upper right",
        bbox_to_anchor=(1.12, 1),
    )

    plt.subplots_adjust(right=0.85)  # Adjust layout to make space for the legend
    plt.tight_layout()

    # Save figure
    plt.savefig(file, format="pdf", bbox_inches="tight", dpi=300)
