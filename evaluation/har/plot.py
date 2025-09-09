#!/usr/bin/env python3
"""
Module: visualize_performance_polars.py

This module loads on-device and cross-device performance CSV data using Polars,
reorders the devices in a specified order, and creates publication-quality visualizations
using Plotly Express.

It produces:
  - A horizontal dot plot (scatter plot) showing on-device performance (mean accuracy with error bars),
    optimized for subtle performance comparison.
  - An 8x8 confusion matrix where each cell represents a “Tested on” vs “Trained on” accuracy,
    with the diagonal filled by the on-device training values.

Both figures are exported as high-resolution PDFs.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import plotly.io as pio
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from evaluation.common import (
    DARK_TEAL,
    DARK_ORANGE,
    RECEIVER_ORDER,
    tgo_palette,
    tgo_cmap_rev,
)
from matplotlib.colors import LinearSegmentedColormap

# Disable MathJax in Kaleido's scope to prevent MathJax loading messages
pio.kaleido.scope.mathjax = None


def load_data(
    on_device_path: Path, cross_device_path: Path
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Load CSV data using Polars.
    """
    on_device_df = pl.read_csv(on_device_path)
    cross_device_df = pl.read_csv(cross_device_path)
    return on_device_df, cross_device_df


def reorder_on_device(on_device: pl.DataFrame) -> pl.DataFrame:
    """
    Reorder the on-device DataFrame to match the desired receiver order.
    """
    order_dict = {r: i for i, r in enumerate(RECEIVER_ORDER)}
    order_values = [order_dict.get(x, 999) for x in on_device["receiver"].to_list()]
    return (
        on_device.with_columns([pl.Series("order", order_values)])
        .sort("order")
        .drop("order")
    )


def plot_on_device_performance(df: pl.DataFrame, save_path: Path | None = None):
    """
    Create a horizontal dot plot (scatter) visualization of on-device performance,
    coloring each marker by its mean accuracy (using the custom colormap),
    removing error-bar caps, and preserving the original receiver order.
    """
    # compute error bars
    df = df.with_columns(
        err_minus=pl.col("mean_accuracy") - pl.col("ci99_lower"),
        err_plus=pl.col("ci99_upper") - pl.col("mean_accuracy"),
    )

    # scatter, colored by mean_accuracy
    fig = px.scatter(
        df,
        x="mean_accuracy",
        y="receiver",
        error_x="std",
        color="mean_accuracy",
        template="plotly_white",
        labels={"mean_accuracy": "Mean Accuracy", "receiver": ""},
        color_continuous_scale=list(reversed(tgo_palette)),
        range_color=[df["mean_accuracy"].min(), df["mean_accuracy"].max()],
    )

    # exact same marker & whisker styling, no whisker caps
    fig.update_traces(
        marker=dict(size=45, line=dict(width=2, color="#A9A9A9")),
        error_x=dict(thickness=5, width=0, color="#636363"),
    )

    # layout: preserve device order, reverse so first is on top, hide colorbar
    fig.update_layout(
        width=1920,
        height=700,
        title_font=dict(size=54, family="Arial", color="black"),
        xaxis=dict(
            tickfont=dict(size=50, family="Arial", color="dimgray"),
            title_font=dict(size=52, family="Arial", color="gray"),
            range=[0.88, 0.975],
        ),
        yaxis=dict(
            categoryorder="array",
            categoryarray=RECEIVER_ORDER,
            autorange="reversed",
            tickfont=dict(size=50, family="Arial", color="dimgray"),
            title_font=dict(size=52, family="Arial", color="gray"),
        ),
        showlegend=False,
        coloraxis_showscale=False,
        margin=dict(l=150, r=80, t=0, b=80),
    )

    if save_path:
        fig.write_image(save_path, width=1920, height=700)
        print(f"On-device performance plot saved as '{save_path}'.")
    else:
        fig.show()


def build_conf_matrix(
    cross_device: pl.DataFrame, on_device: pl.DataFrame
) -> np.ndarray:
    """
    Build an 8x8 confusion matrix.

    For each device pair (Tested on, Trained on) in the specified order,
    the corresponding cell is filled with the cross-device accuracy.
    On the diagonal (where Tested on equals Trained on), the cell is filled
    with the on-device accuracy.
    """
    # Create a lookup dictionary for on-device accuracies.
    on_device_map = {
        row["receiver"]: row["mean_accuracy"] for row in on_device.to_dicts()
    }

    matrix = []
    for test_device in RECEIVER_ORDER:
        # Filter cross_device data for the row corresponding to 'test_device'.
        row_df = cross_device.filter(pl.col("Tested on") == test_device)
        row_data = row_df.to_dicts()[0] if row_df.height > 0 else {}
        row_values = []
        for train_device in RECEIVER_ORDER:
            if test_device == train_device:
                # For the diagonal, use on-device accuracy.
                row_values.append(on_device_map.get(test_device, np.nan))
            else:
                col_name = f"Trained on: {train_device}"
                value = row_data.get(col_name, np.nan)
                row_values.append(value)
        matrix.append(row_values)
    return np.array(matrix)


def plot_confusion_matrix(conf_matrix: np.ndarray, out_file: Path | None):
    """
    Generate a minimalist, publication-ready visualization of an 8x8 confusion matrix.

    The matrix is assumed to follow the global RECEIVER_ORDER for both rows and columns.
    Low accuracy values are rendered in red and high values in green, using a softer diverging
    color palette. Numerical values are displayed in each cell with a subtle dark gray, while
    the tick labels (receiver names) and axis labels are enlarged for publication quality.
    The colorbar is sized so it doesn't exceed the heatmap dimensions.

    Parameters:
        conf_matrix : np.ndarray
            An 8x8 numpy array representing the confusion matrix.
        out_filename : Path | None
            Path to output file (if desired)
    """
    # Create a DataFrame that follows the global RECEIVER_ORDER.
    df = pd.DataFrame(conf_matrix, index=RECEIVER_ORDER, columns=RECEIVER_ORDER)

    # Set a minimalist white style with a larger font scale for a paper-quality look.
    sns.set_theme(style="white", context="paper", font_scale=2)

    # Create a square figure.
    plt.figure(figsize=(10, 10))

    # Plot the heatmap with annotations in a subtle dark gray and minimized grid.
    ax = sns.heatmap(
        df,
        annot=True,
        fmt=".2f",
        vmin=0.2,  # Force minimum of color scale
        vmax=1.0,
        cmap=tgo_cmap_rev,
        square=True,
        cbar_kws={"shrink": 0.725, "label": ""},
        annot_kws={"size": 18, "weight": "bold", "color": "#4f4f4f"},
        linewidths=0,  # Remove cell border grid lines.
    )

    # Move x-axis ticks and label to the top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    # Adjust tick labels (receiver names) to be slightly larger and in a muted dark gray.
    plt.setp(ax.get_xticklabels(), rotation=45, fontsize=20, color="dimgray")
    plt.setp(ax.get_yticklabels(), rotation=45, fontsize=20, color="dimgray")

    # Set the axis labels with a larger font and a subtle gray color.
    ax.set_xlabel("Trained on", fontsize=24, color="gray", labelpad=20)
    ax.set_ylabel("Tested on", fontsize=24, color="gray", labelpad=20)

    # Remove the tick marks to maintain a clean look.
    ax.tick_params(axis="both", which="both", length=0)

    # Ensure a tight, neat layout.
    plt.tight_layout()

    # Save the figure as a PDF.
    if out_file:
        plt.savefig(out_file, format="pdf", bbox_inches="tight", pad_inches=0.1)
    else:
        plt.show()
    plt.close()


def plot_comparison(
    unscaled_df: pl.DataFrame,
    scaled_df: pl.DataFrame,
    save_path: Path | None = None,
    dodge: float = 0.18,  # vertical offset for clarity
):
    """
    Plot mean accuracies of unscaled vs scaled data for each receiver together
    with error bars in a dodged plot
    """
    group_spacing = 1.5

    # add row numbers and y positions
    df_u = (
        unscaled_df.reverse()
        .with_row_index("row_nr")
        .with_columns((pl.col("row_nr") * group_spacing - dodge).alias("y"))
    )
    df_s = (
        scaled_df.reverse()
        .with_row_index("row_nr")
        .with_columns((pl.col("row_nr") * group_spacing + dodge).alias("y"))
    )

    df_u = df_u.with_columns(
        err_minus=pl.col("mean_accuracy") - pl.col("ci99_lower"),
        err_plus=pl.col("ci99_upper") - pl.col("mean_accuracy"),
    )
    df_s = df_s.with_columns(
        err_minus=pl.col("mean_accuracy") - pl.col("ci99_lower"),
        err_plus=pl.col("ci99_upper") - pl.col("mean_accuracy"),
    )
    receivers = df_s["receiver"].to_list()

    # build figure
    fig = go.Figure()
    marker_cfg = dict(size=35, line=dict(width=2, color="#A9A9A9"))
    error_cfg = dict(type="data", thickness=5, width=12, color="#636363")

    for df, name, symbol, color in [
        (df_u, "Raw", "circle", "#66c2a5"),
        (df_s, "AGC-scaled", "diamond", "#fc8d62"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=df["mean_accuracy"].to_list(),
                y=df["y"].to_list(),
                mode="markers",
                name=name + "      ",
                marker={**marker_cfg, "symbol": symbol, "color": color},
                error_x={
                    **error_cfg,
                    "array": df["err_plus"].to_list(),
                    "arrayminus": df["err_minus"].to_list(),
                },
            )
        )

    # layout
    fig.update_layout(
        width=1920,
        height=1600,
        template="plotly_white",
        xaxis=dict(
            title="Mean Accuracy",
            range=[0.88, 0.975],
            title_font=dict(size=54, family="Arial", color="gray"),
            tickfont=dict(size=50, family="Arial", color="dimgray"),
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=[i * group_spacing for i in range(len(receivers))],
            ticktext=receivers,
            title_font=dict(size=54, family="Arial", color="gray"),
            tickfont=dict(size=50, family="Arial", color="dimgray"),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.98,
            xanchor="right",
            x=0.4,
            tracegroupgap=50,
            font=dict(size=54, family="Arial", color="gray"),
        ),
        margin=dict(l=180, r=80, t=70, b=80),
    )

    if save_path:
        fig.write_image(save_path, width=1920, height=1600)
    else:
        fig.show()


def plot_comparison_compact(
    unscaled_df: pl.DataFrame,
    scaled_df: pl.DataFrame,
    *,
    save_path: Path | None = None,
    dodge: float = 0.20,  # horizontal offset between the two series
    # Series appearance (easy to tweak)
    names: tuple[str, str] = ("Raw", "AGC-scaled"),
    colors: tuple[str, str] = (DARK_TEAL, DARK_ORANGE),  # ("#66c2a5", "#fc8d62"),
    symbols: tuple[str, str] = ("circle", "diamond"),
    # Layout & typography
    height: int = 600,
    width: int = 1600,
    receiver_bands: bool = True,
    method_lanes: bool = False,
    inner_labels: bool = True,  # receiver names drawn inside at the top
    label_y: float = 0.98,  # vertical position for names (paper coords 0..1)
    label_clearance: float = 0.02,  # keep top-of-data at least this far below label_y (paper coords)
    min_top_pad_frac: float = 0.12,  # minimum top headroom as fraction of data range
    bottom_pad_frac: float = 0.04,  # small bottom pad so nothing clips
    y_title: str = "Mean Accuracy",
    y_title_standoff: int = 32,
    y_title_font_size: int = 48,
    y_tick_font_size: int = 40,
    y_nticks: int = 6,  # cleaned-up y ticks
    y_range: tuple[float, float]
    | None = None,  # set manually if you want (overrides auto headroom)
):
    """
    Flipped comparison plot (mean ± CI99) with two series (unscaled vs scaled):
    - Values on Y (mean accuracy), receivers on X with a small horizontal dodge.
    - Receiver names rendered *inside* the plotting area near the top.
    - Legend inside bottom-right to save vertical space.
    """

    group_spacing = 1.5

    # ---------- helpers
    def to_dict(df: pl.DataFrame) -> dict[str, tuple[float, float, float]]:
        # receiver -> (mean, err_minus, err_plus)
        out: dict[str, tuple[float, float, float]] = {}
        for r in df.iter_rows(named=True):
            mean = float(r["mean_accuracy"])
            em = float(mean - r["ci99_lower"])
            ep = float(r["ci99_upper"] - mean)
            out[str(r["receiver"])] = (mean, em, ep)
        return out

    # Keep the original top-to-bottom order you had (reverse, then read 'receiver')
    receivers = scaled_df.reverse()["receiver"].to_list()
    # Build quick lookup dicts
    data_u = to_dict(unscaled_df)
    data_s = to_dict(scaled_df)

    # Numerical x position for each receiver (centre column)
    base_x = {rcv: i * group_spacing for i, rcv in enumerate(receivers)}
    offsets = (-dodge, +dodge)  # two series

    # ---------- figure + styles
    fig = go.Figure()
    marker_cfg = dict(size=35, line=dict(width=2, color="#A9A9A9"))
    error_cfg = dict(type="data", thickness=5, width=0, color="#636363")

    # Subtle alternating bands per receiver
    if receiver_bands and receivers:
        half = group_spacing / 2
        for i, rcv in enumerate(receivers):
            if i % 2 == 0:
                fig.add_shape(
                    type="rect",
                    x0=base_x[rcv] - half,
                    x1=base_x[rcv] + half,
                    y0=0,
                    y1=1,
                    yref="paper",
                    line=dict(width=0),
                    fillcolor="rgba(0,0,0,0.035)",
                    layer="below",
                )

    # Optional vertical guide lanes (one for each series offset)
    if method_lanes:
        for k, off in enumerate(offsets):
            for rcv in receivers:
                fig.add_vline(
                    x=base_x[rcv] + off,
                    line_width=1,
                    line_dash="dot",
                    line_color="rgba(0,0,0,0.15)",
                    layer="below",
                )

    # ---------- traces (values on Y)
    series = [
        (data_u, names[0], symbols[0], colors[0]),
        (data_s, names[1], symbols[1], colors[1]),
    ]

    ymins, ymaxs = [], []
    for k, (data_map, name, symbol, color) in enumerate(series):
        xs, ys, ems, eps = [], [], [], []
        for rcv in receivers:
            mean, em, ep = data_map.get(rcv, (np.nan, 0.0, 0.0))
            xs.append(base_x[rcv] + offsets[k])
            ys.append(mean)
            ems.append(em)
            eps.append(ep)
            if np.isfinite(mean):
                ymins.append(mean - em)
                ymaxs.append(mean + ep)

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                name=name,
                marker={**marker_cfg, "symbol": symbol, "color": color},
                error_y={**error_cfg, "array": eps, "arrayminus": ems},  # up & down
            )
        )

    if not ymins or not ymaxs:
        raise ValueError("No finite data to plot.")

    # ---------- y-range with headroom (so inside labels never overlap)
    data_min = float(np.nanmin(ymins))
    data_max = float(np.nanmax(ymaxs))
    data_rng = max(data_max - data_min, 1e-12)
    bot_pad = data_rng * float(bottom_pad_frac)

    if y_range is None:
        # Ensure top-of-data (paper coords) <= label_y - label_clearance
        L = max(0.01, min(0.99, float(label_y) - float(label_clearance)))
        needed_top_pad = ((1.0 - L) / L) * (data_rng + bot_pad)
        top_pad = max(data_rng * float(min_top_pad_frac), needed_top_pad)
        yrange = [data_min - bot_pad, data_max + top_pad]
    else:
        yrange = list(y_range)

    # ---------- layout
    fig.update_layout(
        width=width,
        height=height,
        template="plotly_white",
        margin=dict(l=120, r=40, t=0, b=0, pad=0),  # zero top/bottom margin
        legend=dict(  # INSIDE bottom-right
            orientation="h",
            x=0.98,
            xanchor="right",
            y=0.02,
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.6)",
            bordercolor="rgba(0,0,0,0.05)",
            borderwidth=1,
            font=dict(size=44, family="Arial", color="gray"),
        ),
    )

    # X axis: receiver columns; hide tick labels if we're drawing inside labels
    tickvals = [base_x[r] for r in receivers]
    fig.update_xaxes(
        title=None,
        tickmode="array",
        tickvals=tickvals,
        ticktext=[] if inner_labels else receivers,
        showticklabels=not inner_labels,
        ticks="",
    )

    # Y axis: values with tidy ticks and proper standoff
    fig.update_yaxes(
        title=y_title,
        range=yrange,
        title_font=dict(size=y_title_font_size, family="Arial", color="gray"),
        tickfont=dict(size=y_tick_font_size, family="Arial", color="dimgray"),
        nticks=int(y_nticks),
        title_standoff=y_title_standoff,
        automargin=True,
    )

    # make markers & errors pop
    fig.update_traces(line_width=5, marker_line_width=2, error_y_thickness=5)

    # Receiver labels inside (top)
    if inner_labels:
        for rcv in receivers:
            fig.add_annotation(
                x=base_x[rcv],
                xref="x",
                y=float(label_y),
                yref="paper",
                text=str(rcv),
                showarrow=False,
                xanchor="center",
                yanchor="top",
                font=dict(size=44, family="Arial", color="dimgray"),
                bgcolor="rgba(255,255,255,0.6)",
                bordercolor="rgba(0,0,0,0.05)",
                borderwidth=1,
            )

    if save_path:
        fig.write_image(save_path, width=width, height=height)
    else:
        fig.show()


# ---------------------------------------------------------------------------
# 1)  Off-diagonal distribution plot  (Figure 1 in the paper)
# ---------------------------------------------------------------------------
def plot_offdiag_distribution(
    cross_device_map: dict[str, pl.DataFrame],
    save_path: Path | None = None,
):
    """
    Create a horizontal lollipop chart per method, showing the full off-diagonal
    accuracy range (min → max) as a thick grey line (matching error‐bar thickness),
    with a large dot at the median. Methods are sorted by median accuracy and
    colored according to the “teal→orange” colormap you provided.
    """
    # 1) Gather off-diagonal accuracies
    records: list[dict[str, float | str]] = []
    for method, df in cross_device_map.items():
        for test_dev in RECEIVER_ORDER:
            row = df.filter(pl.col("Tested on") == test_dev).to_dicts()
            if not row:
                continue
            row = row[0]
            for train_dev in RECEIVER_ORDER:
                if train_dev == test_dev:
                    continue
                v = row.get(f"Trained on: {train_dev}")
                if v is not None and not np.isnan(v):
                    records.append({"Method": method, "Accuracy": float(v)})

    # 2) Compute summary stats and sort by median
    dist_df = pl.from_dicts(records).to_pandas()
    summary = (
        dist_df.groupby("Method")["Accuracy"]
        .agg(["min", "median", "max"])
        .sort_values("median")
    )
    methods_order = summary.index.tolist()

    # 3) Build colormap and sample one color per method
    def _rgba_to_hex(rgba):
        r, g, b, _ = rgba
        return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))

    n = len(methods_order)
    colors = [_rgba_to_hex(tgo_cmap_rev(i / (n - 1))) for i in range(n)]
    color_map = dict(zip(methods_order, colors))

    # 4) Styling to match your other plots
    marker_size = 50
    marker_line_cfg = dict(width=2, color="#A9A9A9")
    line_cfg = dict(color="#636363", width=5)  # matches error bar thickness

    # 5) Build lollipop figure
    fig = go.Figure()
    for method in methods_order:
        mn, md, mx = summary.loc[method, ["min", "median", "max"]]

        # range line
        fig.add_trace(
            go.Scatter(
                x=[mn, mx],
                y=[method, method],
                mode="lines",
                line=line_cfg,
                showlegend=False,
            )
        )
        # median dot
        fig.add_trace(
            go.Scatter(
                x=[md],
                y=[method],
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=color_map[method],
                    line=dict(
                        width=marker_line_cfg["width"],
                        color=marker_line_cfg["color"],
                    ),
                ),
                showlegend=False,
            )
        )

    # 6) Layout tweaks (consistent fonts, margins)
    fig.update_layout(
        template="plotly_white",
        width=1920,
        height=1150,
        margin=dict(l=320, r=80, t=0, b=50),
        xaxis=dict(
            title="Cross-device accuracy",
            range=[0.15, 1.0],
            title_font=dict(size=52, family="Arial", color="gray"),
            tickfont=dict(size=52, family="Arial", color="dimgray"),
        ),
        yaxis=dict(
            categoryorder="array",
            categoryarray=methods_order,
            tickfont=dict(size=52, family="Arial", color="dimgray"),
        ),
    )

    # 7) Boost strokes so lines & markers pop
    fig.update_traces(
        line_width=line_cfg["width"], marker_line_width=marker_line_cfg["width"]
    )

    # 8) Save or show
    if save_path:
        fig.write_image(save_path, width=1920, height=1150)
    else:
        fig.show()


# ---------------------------------------------------------------------------
# 2)  Composite “winner-gap” heat-map
# ---------------------------------------------------------------------------
def plot_winner_gap_heatmap(
    cross_device_map: dict[str, pl.DataFrame],
    save_path: Path | None = None,
):
    """
    Composite 8x8 heatmap of winning-method gaps:
      • annotation = best method for each train→test  (or “NA” if none)
      • color      = difference between best and runner-up
    Diagonal cells are left blank.
    """
    methods = list(cross_device_map.keys())

    devices = RECEIVER_ORDER

    # Prepare empty DataFrames
    gap_df = pd.DataFrame(np.nan, index=devices, columns=devices)
    annot_df = pd.DataFrame("", index=devices, columns=devices)
    missing = []

    # Fill gap_df and annot_df
    for test in devices:
        for train in devices:
            if test == train:
                continue
            # collect (method, score) pairs
            vals = []
            for m in methods:
                df = cross_device_map[m]
                row = df.filter(pl.col("Tested on") == test).to_dicts()
                if not row:
                    continue
                v = row[0].get(f"Trained on: {train}", None)
                if v is not None and not np.isnan(v):
                    method = (
                        r"$\ell_1$" if m == "ℓ₁" else r"$\ell_2$" if m == "ℓ₂" else m
                    )
                    vals.append((method, v))
            if not vals:
                missing.append((train, test))
                annot_df.at[test, train] = "NA"
            else:
                # sort descending
                vals.sort(key=lambda x: x[1], reverse=True)
                best_m, best_v = vals[0]
                second_v = vals[1][1] if len(vals) > 1 else best_v
                gap_df.at[test, train] = best_v - second_v
                annot_df.at[test, train] = best_m

    # Report true off-diagonal gaps
    if missing:
        msg = ", ".join(f"({t}←{r})" for r, t in missing)
        print(f"\n[plot_winner_gap_heatmap] ⚠️  Missing scores for: {msg}\n")

    # Mask diagonal
    mask = np.eye(len(devices), dtype=bool)

    # Plot
    sns.set_theme(style="white", context="paper", font_scale=2)
    cmap = LinearSegmentedColormap.from_list(
        "something", ["#9c9c9c", "#ffb84d", "#FFA500"], N=256
    )
    vmax = np.nanmax(gap_df.to_numpy()) if np.isfinite(gap_df.to_numpy()).any() else 1.0

    plt.figure(figsize=(10, 10))
    ax = sns.heatmap(
        gap_df,
        annot=annot_df,
        fmt="",
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
        mask=mask,
        square=True,
        linewidths=0,
        cbar_kws={"shrink": 0.775, "label": "Gap to runner-up"},
        annot_kws={"size": 13, "weight": "bold", "color": "#4f4f4f"},
    )
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    plt.setp(ax.get_xticklabels(), rotation=45, fontsize=20, color="dimgray")
    plt.setp(ax.get_yticklabels(), rotation=45, fontsize=20, color="dimgray")
    ax.set_xlabel("Trained on", fontsize=24, color="gray", labelpad=20)
    ax.set_ylabel("Tested on", fontsize=24, color="gray", labelpad=20)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight", pad_inches=0.1)
        plt.close()
    else:
        plt.show()


def read_csv(filename: Path) -> pl.DataFrame:
    data = pl.read_csv(filename)
    return reorder_on_device(data)


def main():
    # Direct comparison of unscaled and agcscaled
    for pre in ["un", "agc"]:
        # Define paths for CSV data.
        on_device_csv = data_path / f"on-device/{pre}scaled_botonghar_kfold_summary.csv"
        cross_device_csv = data_path / f"cross-device/{pre}scaled_botonghar.csv"

        # Load CSV files.
        on_device, cross_device = load_data(on_device_csv, cross_device_csv)

        # Reorder the on-device DataFrame.
        on_device = reorder_on_device(on_device)

        # Create and export the on-device performance visualization.
        plot_on_device_performance(
            on_device, save_path=img_path / f"{pre}scaled-ondevice.pdf"
        )

        # Build the 8x8 confusion matrix and export its visualization.
        cm_matrix = build_conf_matrix(cross_device, on_device)
        plot_confusion_matrix(
            cm_matrix, out_file=img_path / f"{pre}scaled-confusion.pdf"
        )

    # 1) distribution figure
    plot_offdiag_distribution(
        cross_device_map, save_path=img_path / "offdiag-distribution.pdf"
    )

    # 2) composite heat-map
    plot_winner_gap_heatmap(
        cross_device_map, save_path=img_path / "winnter-gap-heatmap.pdf"
    )


if __name__ == "__main__":
    # Read data
    # fmt: off

    data_path = Path("data/har")
    img_path  = data_path / "img"
    a = read_csv(data_path / "on-device/unscaled_botonghar_kfold_summary.csv")
    b = read_csv(data_path / "on-device/agcscaled_botonghar_kfold_summary.csv")

    cross_device_map = {
        # Rescaling for recovery or explicit removal of AGC
        "raw":             pl.read_csv(data_path / "cross-device/unscaled_botonghar.csv"),    # Raw data, not preprocessed
        "ℓ₁":              pl.read_csv(data_path / "cross-device/agcscaled_botonghar.csv"),   # Divided by l1-norm, i.e. mean of absolute values \Cref{eq:amp-norm}
        "ℓ₂":              pl.read_csv(data_path / "cross-device/rms_botonghar.csv"),         # Divided by l2-norm, i.e. CSI power \cite{ratnam2024optimal, gaussian2020}
        "RSSI":            pl.read_csv(data_path / "cross-device/rssi_botonghar.csv"),        # Rescaled using RSSI (\Cref{eq:rssi-rescale})
        "DBSCN":           pl.read_csv(data_path / "cross-device/dbscan_botonghar.csv"),      # DBScan \cite{liu2021wiphone}
        "GINC":            pl.read_csv(data_path / "cross-device/ratnam_1_botonghar.csv"),    # Gain Increment Clustering, Algorithm 1 from \cite{ratnam2024optimal}
        "λ-grid":          pl.read_csv(data_path / "cross-device/ratnam_2_botonghar.csv"),    # Uniform grid AGC ML-based optimization, Algorithm 3 from \cite{ratnam2024optimal}
        # The next four are all "smoothing" with different filters, since that is often done
        "hampel":          pl.read_csv(data_path / "cross-device/hampel_botonghar.csv"),      # Detect outliers based on consistent MAD, replace with running Median. E.g. \cite{liu2015sleep,wisleep2014,deman2015}
        "median":          pl.read_csv(data_path / "cross-device/rollingmed_botonghar.csv"),  # Rolling median filter \cite{jie2018,seare2020,wisign2017}
        "savgol":          pl.read_csv(data_path / "cross-device/savgol_botonghar.csv"),      # Savitzky-Golay \cite{yang2021,zeng2018fullbreathe, dou2021full}
        "wavelet":         pl.read_csv(data_path / "cross-device/wavelet_botonghar.csv"),     # Wavelet denoising \cite{falldefi2018, neuralwave2018, rttwd2017, yangmotion2017,wisleep2014}
        # Using derived features that are amplitude invariant to begin with
        "morph":           pl.read_csv(data_path / "cross-device/morphology_botonghar.csv"),  # Morphological feature, from \cite{chen2023}
        "ratio":           pl.read_csv(data_path / "cross-device/doublediff_botonghar.csv"),  # Double ratio, from \cite{yi2024enabling}
    }
    # fmt: on

    plot_comparison_compact(a, b, save_path=img_path / "agc-comparison.pdf")

    main()
