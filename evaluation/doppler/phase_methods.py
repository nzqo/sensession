#!/usr/bin/env python3
"""
Phase detrending impact: Δ-error matrix plot

Loads Doppler speed estimates from doppler_estimates.parquet, computes
median absolute error per (receiver, pipeline, phase), and visualizes
the change in error when switching from baseline phase detrending to
LS phase detrending as a PxN heatmap:

    rows    = pipelines (e.g., Raw, AGC-removed, RSSI-scaled)
    columns = receivers
    cell    = Δ|error| = med|err|_LS − med|err|_baseline  [m/s]

Negative values (bluish) -> LS phase improves the estimate.
Positive values (more orange) -> LS phase worsens the estimate.
Values near zero -> negligible impact.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from evaluation.common import RECEIVER_ORDER, tgo_cmap_rev
from matplotlib.colors import TwoSlopeNorm

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

data_dir = Path.cwd() / "data" / "doppler_emulation_slow"
EST_PATH = data_dir / "doppler_estimates.parquet"
GROUNDTRUTH = 1.0


# ---------------------------------------------------------------------------
# Data loading / stats
# ---------------------------------------------------------------------------


def load_estimates(path: Path) -> pl.DataFrame:
    """
    Expected schema of doppler_estimates.parquet:

        receiver       : str
        pipeline       : str
        phase          : str   ("baseline", "phasefit")
        estimated_speed: float
    """
    df = pl.read_parquet(path)
    return df.select(
        pl.col("receiver").cast(pl.Utf8),
        pl.col("pipeline").cast(pl.Utf8),
        pl.col("phase").cast(pl.Utf8),
        pl.col("estimated_speed").cast(pl.Float64),
    )


def receiver_order(df: pl.DataFrame) -> list[str]:
    recs = df.select("receiver").unique()["receiver"].to_list()
    ordered = [r for r in RECEIVER_ORDER if r in recs]
    if not ordered:
        ordered = sorted(recs)
    return ordered


def pipeline_order(df: pl.DataFrame) -> list[str]:
    """
    Infer pipeline labels from the file and order them in a sane way:

        Raw / AGC-removed / RSSI-scaled   (with or without trailing spaces)

    Any extra pipelines are appended at the end.
    """
    present = df.select("pipeline").unique()["pipeline"].to_list()

    # Preferred names (with and without padding)
    pref_groups = [
        ("phase cleaned   ", "phase cleaned"),
        ("+AGC-removed   ", "+AGC-removed"),
        ("+RSSI-scaled",),
    ]

    ordered: list[str] = []

    # Pull in preferred ones in the desired order, if present
    for group in pref_groups:
        for name in group:
            if name in present and name not in ordered:
                ordered.append(name)
                break  # go to next group once one variant is found

    # Append any remaining pipelines that weren't matched
    for p in present:
        if p not in ordered:
            ordered.append(p)

    return ordered


def compute_stats(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute median |error| per (receiver, pipeline, phase).
    """
    return (
        df.with_columns(
            (pl.col("estimated_speed") - GROUNDTRUTH).abs().alias("abs_err")
        )
        .group_by(["receiver", "pipeline", "phase"])
        .agg(
            pl.col("abs_err").median().alias("med_abs_err"),
        )
    )


def compute_delta_matrix(
    stats: pl.DataFrame,
) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Build a P x N matrix of Δ|error|:

        Δ|error| = med|err|_LS − med|err|_baseline

    rows    = pipelines (as stored in the parquet)
    columns = receivers
    """
    recs = receiver_order(stats)
    pls = pipeline_order(stats)

    mat = np.full((len(pls), len(recs)), np.nan, dtype=float)

    for i, p in enumerate(pls):
        for j, r in enumerate(recs):
            base_df = stats.filter(
                (pl.col("receiver") == r)
                & (pl.col("pipeline") == p)
                & (pl.col("phase") == "baseline")
            ).select("med_abs_err")

            pf_df = stats.filter(
                (pl.col("receiver") == r)
                & (pl.col("pipeline") == p)
                & (pl.col("phase") == "phasefit")
            ).select("med_abs_err")

            if base_df.height == 0 or pf_df.height == 0:
                continue  # leave as NaN -> white cell

            base_val = float(base_df["med_abs_err"][0])
            pf_val = float(pf_df["med_abs_err"][0])
            mat[i, j] = pf_val - base_val

    return mat, recs, pls


# ---------------------------------------------------------------------------
# Plot: Δ-error matrix (P x N heatmap, tgo_cmap_rev)
# ---------------------------------------------------------------------------
def plot_delta_matrix(
    delta_mat: np.ndarray,
    receivers: list[str],
    pipelines_labels: list[str],
    *,
    out_file: Path | None = None,
):
    """
    P x N heatmap:

        rows    = pipelines_labels (exact strings from file)
        columns = receivers
        value   = Δ|error| [m/s]

    Color map: tgo_cmap_rev, centered at 0 via TwoSlopeNorm.
    """

    # Pretty row labels: strip padding but keep ordering
    row_labels = [p.strip() for p in pipelines_labels]

    df = pd.DataFrame(
        delta_mat,
        index=row_labels,
        columns=receivers,
    )

    # Symmetric color limits around 0
    if np.isfinite(delta_mat).any():
        max_abs = float(np.nanmax(np.abs(delta_mat)))
        if max_abs == 0:
            max_abs = 1e-6
    else:
        max_abs = 1e-6

    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    sns.set_theme(style="white", context="paper", font_scale=2)

    plt.figure(figsize=(10, 3.5))
    ax = sns.heatmap(
        df,
        annot=True,
        fmt=".3f",
        cmap=tgo_cmap_rev,
        norm=norm,
        square=False,
        cbar_kws={
            "shrink": 0.8,
            "label": r"$\Delta |e|$ [m/s]",
        },
        annot_kws={
            "size": 14,
            "weight": "bold",
            "color": "#4f4f4f",
        },
        linewidths=0,
    )

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", fontsize=16, color="dimgray")
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=16, color="dimgray")

    ax.set_xlabel("", fontsize=18, color="gray", labelpad=10)
    ax.set_ylabel("", fontsize=18, color="gray", labelpad=10)
    ax.tick_params(axis="both", which="both", length=0)

    plt.tight_layout()

    if out_file:
        plt.savefig(out_file, format="pdf", bbox_inches="tight", pad_inches=0.1)
        plt.close()
    else:
        plt.show()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    df = load_estimates(EST_PATH)
    stats = compute_stats(df)
    delta_mat, recs, pls = compute_delta_matrix(stats)

    print("Pipelines in file (in plot order):", pls)

    plot_delta_matrix(
        delta_mat,
        recs,
        pls,
        out_file=data_dir / "phase-delta-matrix.pdf",
    )


if __name__ == "__main__":
    main()
