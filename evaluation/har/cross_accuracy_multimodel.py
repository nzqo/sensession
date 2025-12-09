#!/usr/bin/env python3
"""
Ridgeline plot of cross-device accuracies before/after AGC scaling.

For each model, take all off-diagonal (train_receiver, test_receiver) pairs
from:

    <model>_unscaled_cross_std_ablation.csv
    <model>_agcscaled_cross_std_ablation.csv

for a fixed standardization, and plot ridgeline KDEs of the accuracies:
    - x: accuracy
    - one row per model
    - two overlapping densities per row:
        Unscaled (LIGHT_ORANGE), AGC-scaled (LIGHT_TEAL)
"""

from pathlib import Path

import pandas as pd
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from evaluation.common import (
    DARK_GRAY,
    LIGHT_TEAL,
    LIGHT_ORANGE,
)
from matplotlib.patches import Patch

BASE_FONTSIZE = 16  # scale text sizes across all figures


def load_model_accuracy_long_df(
    model: str,
    base_dir: Path,
    standardization: str,
) -> pl.DataFrame:
    """
    Load cross-device accuracies for one model and standardization.
    Returns a long-format DF with columns: model, status, accuracy.
    """
    fname_unscaled = base_dir / f"{model}_unscaled_cross_std_ablation.csv"
    fname_agc = base_dir / f"{model}_agcscaled_cross_std_ablation.csv"

    unscaled = (
        pl.read_csv(fname_unscaled)
        .filter(pl.col("standardization") == standardization)
        .filter(pl.col("train_receiver") != pl.col("test_receiver"))
        .select(
            pl.lit(model).alias("model"),
            pl.lit("Unscaled").alias("status"),
            pl.col("accuracy"),
        )
    )

    agc = (
        pl.read_csv(fname_agc)
        .filter(pl.col("standardization") == standardization)
        .filter(pl.col("train_receiver") != pl.col("test_receiver"))
        .select(
            pl.lit(model).alias("model"),
            pl.lit("AGC-scaled").alias("status"),
            pl.col("accuracy"),
        )
    )

    return pl.concat([unscaled, agc])


def plot_nn_ridgeline_accuracy(
    model_names: list[str],
    base_dir: Path,
    standardization: str = "ZSCORE_ACROSS_EXAMPLES",
    save_path: Path | None = None,
):
    """
    Ridgeline KDE plot:
        - one row per model
        - x = accuracy
        - two overlapping KDEs per row (Unscaled vs AGC-scaled),
          each normalized independently.
    """
    dfs = []
    for m in model_names:
        df_m = load_model_accuracy_long_df(m, base_dir, standardization)
        if df_m.height > 0:
            dfs.append(df_m)

    if not dfs:
        raise ValueError(
            f"No accuracy values found for any model with standardization '{standardization}'."
        )

    df_all = pl.concat(dfs)
    df_pd = df_all.to_pandas()

    # Order models as given in model_names
    df_pd["model"] = pd.Categorical(
        df_pd["model"], categories=model_names, ordered=True
    )

    sns.set_theme(
        style="white",
        rc={
            "axes.facecolor": (0, 0, 0, 0),
            "text.color": DARK_GRAY,
            "axes.labelcolor": DARK_GRAY,
            "axes.edgecolor": DARK_GRAY,
            "xtick.color": DARK_GRAY,
            "ytick.color": DARK_GRAY,
            "font.size": BASE_FONTSIZE,
            "axes.labelsize": BASE_FONTSIZE,
            "xtick.labelsize": BASE_FONTSIZE - 1,
            "ytick.labelsize": BASE_FONTSIZE - 1,
        },
    )

    g = sns.FacetGrid(
        data=df_pd,
        row="model",
        aspect=6,
        height=1.4,
        sharex=True,
        sharey=False,
    )

    g.map_dataframe(
        sns.kdeplot,
        "accuracy",
        bw_adjust=0.5,
        clip_on=False,
        fill=True,
        alpha=0.6,  # slightly see-through
        linewidth=1.2,
        hue="status",
        hue_order=["Unscaled", "AGC-scaled"],
        palette=[LIGHT_ORANGE, LIGHT_TEAL],
        multiple="layer",  # overlay, don't stack
        common_norm=False,  # each hue normalized separately
    )

    # Baseline per ridge
    g.map(plt.axhline, y=0, lw=1.5, clip_on=False, color=DARK_GRAY)

    # Set the subplots to overlap
    g.fig.subplots_adjust(hspace=-0.25)

    # Clean up axes
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.set_xlabels("Cross-device accuracy", color=DARK_GRAY)
    g.despine(bottom=True, left=True)

    # Remove any auto legends seaborn might have created
    for ax in g.axes.flatten():
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    # Manual legend: inside the top (first) axis, upper left, horizontal
    first_ax = g.axes[0, 0]
    legend_handles = [
        Patch(
            facecolor=LIGHT_ORANGE, edgecolor=LIGHT_ORANGE, alpha=0.6, label="Unscaled"
        ),
        Patch(
            facecolor=LIGHT_TEAL, edgecolor=LIGHT_TEAL, alpha=0.6, label="AGC-scaled"
        ),
    ]
    legend = first_ax.legend(
        handles=legend_handles,
        loc="upper left",
        frameon=False,
        borderaxespad=0.3,
        handlelength=1.8,
        handletextpad=0.4,
        fontsize=BASE_FONTSIZE - 1,
        ncol=2,  # horizontal layout
        columnspacing=0.8,
    )
    for text in legend.get_texts():
        text.set_color(DARK_GRAY)

    # Manually label each row with the model name on the left
    for ax, model in zip(g.axes.flatten(), model_names):
        ax.text(
            0.0,
            0.2,
            model,
            fontweight="bold",
            color=DARK_GRAY,
            ha="left",
            va="center",
            transform=ax.transAxes,
            fontsize=BASE_FONTSIZE,
        )

    if save_path is not None:
        g.fig.savefig(save_path, bbox_inches="tight")
        print(f"Ridgeline plot saved as '{save_path}'.")
    else:
        plt.show()


def main():
    data_path = Path("data/har")
    img_path = data_path / "img"
    img_path.mkdir(parents=True, exist_ok=True)

    models = [
        "aril",
        "reconformer",
        "rganet",  # uncomment once rganet data is available
        "wiadn",
    ]
    standardization = "ZSCORE_ACROSS_EXAMPLES"

    out_file = img_path / f"crossdevice-ridgeline-{standardization.lower()}.pdf"

    plot_nn_ridgeline_accuracy(
        model_names=models,
        base_dir=data_path / "standardization_ablations",
        standardization=standardization,
        save_path=out_file,
    )


if __name__ == "__main__":
    main()
