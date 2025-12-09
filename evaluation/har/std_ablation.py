#!/usr/bin/env python3
"""
Ridgeline plot of cross-device accuracies per standardization for a single model.

Expected data layout:

    data/har/multi-model/
        <model>_unscaled_cross_std_ablation.csv
        <model>_agcscaled_cross_std_ablation.csv

where <model> is one of {aril, reconformer, wiadn, rganet}.

For the chosen model, this script:
  - loads both unscaled and AGC-scaled accuracy results
  - for each standardization, collects the distributions of cross-receiver accuracies
  - orders standardizations by the mean AGC-scaled cross-receiver accuracy
  - plots a ridgeline KDE:
        y: standardization (one row per standardization)
        x: cross-receiver accuracy
        two overlapping densities per row:
            Unscaled   -> LIGHT_ORANGE
            AGC-scaled -> LIGHT_TEAL
"""

import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from evaluation.common import DARK_GRAY, LIGHT_TEAL, LIGHT_ORANGE
from matplotlib.patches import Patch

BASE_FONTSIZE = 8

# Axes coordinates: x in [0,1] is inside plot; <0 is left outside, >1 right outside
#                   y in [0,1] is bottom->top inside the subplot
LABEL_X_AXES = -0.25  # horizontal position of standardization labels in axes coords
LABEL_Y_AXES = 0.25  # vertical position of standardization labels in axes coords

# fmt: off
METHOD_LABELS = {
    "ZSCORE_GLOBAL":           "Z-score (global)",
    "ZSCORE_SUBCARRIER":       "Z-score (subcarrier)",
    "ZSCORE_WINDOW":           "Z-score (window)",
    "ZSCORE_ACROSS_EXAMPLES":  "Z-score (examples)",
    "MIN_MAX_GLOBAL":          "Min-max (global)",
    "MIN_MAX_SUBCARRIER":      "Min-max (subcarrier)",
    "MIN_MAX_WINDOW":          "Min-max (window)",
    "MIN_MAX_ACROSS_EXAMPLES": "Min-max (examples)",
}
# fmt: on


def parse_args():
    p = argparse.ArgumentParser(
        description="Ridgeline plot of cross-receiver accuracies per standardization for one model."
    )
    p.add_argument(
        "--model",
        type=str,
        default="aril",
        choices=["aril", "reconformer", "wiadn"],
        help="Model name (default: aril).",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional dataset name to filter on (matches 'dataset' column).",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory for plot (default: data/har/img).",
    )
    return p.parse_args()


def apply_standardization_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add a human-readable label column for standardization methods."""
    df = df.copy()
    df["std_label"] = df["standardization"].map(lambda x: METHOD_LABELS.get(x, x))  # type: ignore
    return df


def load_model_df(model: str, dataset_filter: str | None) -> pd.DataFrame:
    base_dir = Path("data/har/standardization_ablations")

    csv_unscaled = base_dir / f"{model}_unscaled_cross_std_ablation.csv"
    csv_agc = base_dir / f"{model}_agcscaled_cross_std_ablation.csv"

    if not csv_unscaled.exists():
        raise SystemExit(f"CSV not found: {csv_unscaled}")
    if not csv_agc.exists():
        raise SystemExit(f"CSV not found: {csv_agc}")

    df_un = pd.read_csv(csv_unscaled)
    df_agc = pd.read_csv(csv_agc)

    if dataset_filter is not None:
        df_un = df_un[df_un["dataset"] == dataset_filter]
        df_agc = df_agc[df_agc["dataset"] == dataset_filter]

    # cross-device only
    df_un = df_un[df_un["train_receiver"] != df_un["test_receiver"]].copy()
    df_agc = df_agc[df_agc["train_receiver"] != df_agc["test_receiver"]].copy()

    if df_un.empty or df_agc.empty:
        raise SystemExit(
            "No cross-receiver rows after filtering; check dataset and files."
        )

    df_un["status"] = "Unscaled"
    df_agc["status"] = "AGC-scaled"

    df_all = pd.concat([df_un, df_agc], ignore_index=True)
    df_all = apply_standardization_labels(df_all)
    return df_all


def main():
    args = parse_args()

    df = load_model_df(args.model, args.dataset)

    # Order standardizations by mean AGC-scaled accuracy
    df_agc = df[df["status"] == "AGC-scaled"]
    std_means_agc = (
        df_agc.groupby("std_label")["accuracy"].mean().sort_values(ascending=True)
    )
    std_order = std_means_agc.index.tolist()

    df = df[df["std_label"].isin(std_order)].copy()
    df["std_label"] = pd.Categorical(
        df["std_label"], categories=std_order, ordered=True
    )

    if df.empty:
        raise SystemExit("No rows left after filtering to known standardizations.")

    # Output dir: fixed to data/har/img unless overridden
    if args.outdir is not None:
        outdir = args.outdir
    else:
        outdir = Path("data/har/img")
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- Plot style ----
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
        data=df,
        row="std_label",
        aspect=8,
        height=0.5,
        sharex=True,
        sharey=False,
    )

    g.map_dataframe(
        sns.kdeplot,
        "accuracy",
        bw_adjust=0.5,
        clip_on=False,
        fill=True,
        alpha=0.6,
        linewidth=1.2,
        hue="status",
        hue_order=["Unscaled", "AGC-scaled"],
        palette=[LIGHT_ORANGE, LIGHT_TEAL],
        multiple="layer",
        common_norm=False,
    )

    # baseline per ridge
    g.map(plt.axhline, y=0, lw=1.2, clip_on=False, color=DARK_GRAY)

    # Slight overlap of ridges; some space at top for legend
    g.fig.subplots_adjust(hspace=-0.1, left=0.16, top=0.9)

    # clean axes
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.set_xlabels("Cross-receiver accuracy", color=DARK_GRAY)
    g.despine(bottom=True, left=True)

    # remove any auto legends
    for ax in g.axes.flatten():
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    fig = g.fig

    # Figure-level legend above the axes, outside the plot area but close
    legend_handles = [
        Patch(
            facecolor=LIGHT_ORANGE, edgecolor=LIGHT_ORANGE, alpha=0.6, label="Unscaled"
        ),
        Patch(
            facecolor=LIGHT_TEAL, edgecolor=LIGHT_TEAL, alpha=0.6, label="AGC-scaled"
        ),
    ]
    legend = fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        frameon=False,
        handlelength=1.8,
        handletextpad=0.4,
        fontsize=BASE_FONTSIZE - 1,
        ncol=2,
        columnspacing=0.8,
    )
    for text in legend.get_texts():
        text.set_color(DARK_GRAY)

    # Standardization labels: position controlled by LABEL_X_AXES / LABEL_Y_AXES
    for ax, std_label in zip(g.axes.flatten(), std_order):
        ax.text(
            LABEL_X_AXES,
            LABEL_Y_AXES,
            std_label,
            fontweight="bold",
            color=DARK_GRAY,
            ha="left",
            va="center",
            transform=ax.transAxes,
            fontsize=BASE_FONTSIZE,
            rotation=0,
            clip_on=False,
        )

    out_path = outdir / f"{args.model}-standardizations-ridgeline.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved standardization ridgeline plot to {out_path}")


if __name__ == "__main__":
    main()
