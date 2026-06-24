#!/usr/bin/env python3
# pylint: disable=too-many-lines,too-many-locals,too-many-statements

"""
Visualize ARIL on-device and cross-device performance.

This script:
  - Loads on-device k-fold summaries (NONE and Z-SCORE)
  - Loads cross-device accuracy matrices for a set of methods.
  - Produces publication-quality PDF plots:
      * on-device accuracy with std-dev error bars (single method),
      * on-device accuracy comparison with BCa intervals (unscaled vs AGC-scaled),
      * cross-device "confusion" heatmaps,
      * off-diagonal cross-device accuracy distributions per method,
      * winner-gap heatmap (best method vs runner up per train/test pair).
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
    LIGHT_TEAL,
    LIGHT_ORANGE,
    RECEIVER_ORDER,
    tgo_palette,
    tgo_cmap_rev,
)
from matplotlib.colors import LinearSegmentedColormap

# Disable MathJax in Kaleido's scope to prevent MathJax loading messages
pio.defaults.mathjax = None

# ---------------------------------------------------------------------------
# Global plotting config
# ---------------------------------------------------------------------------
PLOT_WIDTH_WIDE = 1920
PLOT_HEIGHT_SHORT = 660
PLOT_HEIGHT_MEDIUM = 1150
PLOT_HEIGHT_TALL = 1600

FONT_FAMILY = "Arial"
TICK_FONT_SIZE_LARGE = 50
TITLE_FONT_SIZE_LARGE = 52
LEGEND_FONT_SIZE_LARGE = 54

MARKER_SIZE_LARGE = 45
MARKER_LINE_WIDTH = 2
MARKER_LINE_COLOR = "#A9A9A9"

ERROR_BAR_THICKNESS = 5
ERROR_BAR_COLOR = "#636363"

HEATMAP_ANNOT_COLOR = "#4f4f4f"

HEATMAP_LOW = "#EE3377"
HEATMAP_HIGH = "#009988"


# ---------------------------------------------------------------------------
# Basic loading / utilities
# ---------------------------------------------------------------------------
def reorder_by_receiver_order(on_device: pl.DataFrame) -> pl.DataFrame:
    """
    Reorder an on-device DataFrame to match the global RECEIVER_ORDER.
    """
    order_index_by_receiver = {
        receiver: idx for idx, receiver in enumerate(RECEIVER_ORDER)
    }
    order_values = [
        order_index_by_receiver.get(receiver_name, 999)
        for receiver_name in on_device["receiver"].to_list()
    ]

    return (
        on_device.with_columns([pl.Series("order", order_values)])
        .sort("order")
        .drop("order")
    )


def load_on_device_summary_reordered(summary_path: Path) -> pl.DataFrame:
    """
    Convenience wrapper: read an on-device summary CSV and reorder it
    according to RECEIVER_ORDER.
    """
    return reorder_by_receiver_order(pl.read_csv(summary_path))


def compute_bca_stats_by_receiver(
    data: pl.DataFrame,
    mean_col: str = "mean_accuracy",
    lower_col: str = "ci99_lower",
    upper_col: str = "ci99_upper",
) -> dict[str, tuple[float, float, float]]:
    """
    Compute BCa stats per receiver: mean, lower error, upper error.
    Returns: {receiver -> (mean, err_minus, err_plus)}.
    """
    stats_by_receiver: dict[str, tuple[float, float, float]] = {}
    for row in data.iter_rows(named=True):
        mean = float(row[mean_col])
        err_minus = float(mean - row[lower_col])
        err_plus = float(row[upper_col] - mean)
        stats_by_receiver[str(row["receiver"])] = (mean, err_minus, err_plus)
    return stats_by_receiver


# ---------------------------------------------------------------------------
# 1) Single-method on-device accuracy (std-dev error bars)
# ---------------------------------------------------------------------------
def plot_on_device_accuracy_with_std(
    on_device_df: pl.DataFrame,
    save_path: Path | None = None,
    *,
    x_range: tuple[float, float] = (0.88, 0.975),
) -> None:
    """
    Horizontal dot plot (scatter) of on-device performance for a *single* method.

    - X-axis: mean_accuracy
    - Y-axis: receiver (in RECEIVER_ORDER)
    - Error bars: standard deviation ('std' column)
    - Marker color: mean_accuracy using the tgo_palette colormap
    """

    # ---- config ----
    marker_cfg = dict(
        size=MARKER_SIZE_LARGE,
        line=dict(width=MARKER_LINE_WIDTH, color=MARKER_LINE_COLOR),
    )
    error_cfg = dict(thickness=ERROR_BAR_THICKNESS, width=0, color=ERROR_BAR_COLOR)

    # 1) Scatter plot with std-dev error bars
    figure = px.scatter(
        on_device_df,
        x="mean_accuracy",
        y="receiver",
        error_x="std",
        color="mean_accuracy",
        template="plotly_white",
        labels={"mean_accuracy": "Mean Accuracy", "receiver": ""},
        color_continuous_scale=list(reversed(tgo_palette)),
        range_color=[
            on_device_df["mean_accuracy"].min(),
            on_device_df["mean_accuracy"].max(),
        ],
    )

    # 2) Style markers and error bars
    figure.update_traces(
        marker=marker_cfg,
        error_x=error_cfg,
    )

    # 3) Layout config
    figure.update_layout(
        width=PLOT_WIDTH_WIDE,
        height=PLOT_HEIGHT_SHORT,
        title_font=dict(size=54, family=FONT_FAMILY, color="black"),
        xaxis=dict(
            tickfont=dict(
                size=TICK_FONT_SIZE_LARGE, family=FONT_FAMILY, color="dimgray"
            ),
            title_font=dict(
                size=TITLE_FONT_SIZE_LARGE, family=FONT_FAMILY, color="gray"
            ),
            range=list(x_range),
        ),
        yaxis=dict(
            categoryorder="array",
            categoryarray=RECEIVER_ORDER,
            autorange="reversed",
            tickfont=dict(
                size=TICK_FONT_SIZE_LARGE, family=FONT_FAMILY, color="dimgray"
            ),
            title_font=dict(
                size=TITLE_FONT_SIZE_LARGE, family=FONT_FAMILY, color="gray"
            ),
        ),
        showlegend=False,
        coloraxis_showscale=False,
        margin=dict(l=150, r=80, t=0, b=130),
    )

    figure.update_xaxes(
        automargin=True,
        title_standoff=5,  # keep it from being pushed too far down
    )
    # 4) Save or show
    if save_path:
        figure.write_image(save_path, width=PLOT_WIDTH_WIDE, height=PLOT_HEIGHT_SHORT)
        print(f"On-device (std) accuracy plot saved to '{save_path}'.")
    else:
        figure.show()


# ---------------------------------------------------------------------------
# 2) Confusion-matrix-like cross-device accuracy plots
# ---------------------------------------------------------------------------
def build_cross_device_confusion_matrix(
    cross_device_df: pl.DataFrame,
    on_device_df: pl.DataFrame,
) -> np.ndarray:
    """
    Build an 8x8 "confusion matrix" of cross-device accuracies.

    For each device pair (Tested on, Trained on) in RECEIVER_ORDER:

      - Off-diagonal: cross-device accuracy from 'cross_device_df'
                      (filtered by train_receiver / test_receiver).
      - Diagonal:     on-device accuracy from 'on_device_df'.
    """
    on_device_accuracy_by_receiver = {
        row["receiver"]: row["mean_accuracy"] for row in on_device_df.to_dicts()
    }

    matrix_rows: list[list[float]] = []

    for tested_device in RECEIVER_ORDER:
        row_values: list[float] = []
        for trained_device in RECEIVER_ORDER:
            if tested_device == trained_device:
                row_values.append(
                    on_device_accuracy_by_receiver.get(tested_device, np.nan)
                )
            else:
                value_series = cross_device_df.filter(
                    (pl.col("train_receiver") == trained_device)
                    & (pl.col("test_receiver") == tested_device)
                )["accuracy"]

                if value_series.is_empty():
                    row_values.append(np.nan)
                else:
                    row_values.append(float(value_series.item()))

        matrix_rows.append(row_values)

    return np.array(matrix_rows)


def plot_cross_device_confusion_heatmap(
    confusion_matrix: np.ndarray,
    save_path: Path | None = None,
    *,
    vmin: float = 0.2,
    vmax: float = 1.0,
) -> None:
    """
    Visualize an 8x8 cross-device accuracy array as a heatmap.

    Rows: Tested on
    Columns: Trained on
    """
    # ---- config ----
    sns.set_theme(style="white", context="paper", font_scale=2)

    # --- transparent figure/axes + custom pink->teal colormap ---
    plt.rcParams["savefig.transparent"] = True
    plt.rcParams["figure.facecolor"] = (0, 0, 0, 0)
    plt.rcParams["axes.facecolor"] = (0, 0, 0, 0)

    heatmap_cmap = LinearSegmentedColormap.from_list(
        "aril_pink_teal", [HEATMAP_LOW, HEATMAP_HIGH]
    )
    LIGHT_GRAY = "#c9c9c9"

    heatmap_df = pd.DataFrame(
        confusion_matrix, index=RECEIVER_ORDER, columns=RECEIVER_ORDER
    )

    # 1) Create figure
    plt.figure(figsize=(10, 10), facecolor=(0, 0, 0, 0))
    plt.gca().set_facecolor((0, 0, 0, 0))

    # 2) Heatmap with annotations
    axis = sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".2f",
        vmin=vmin,
        vmax=vmax,
        cmap=heatmap_cmap,
        square=True,
        cbar_kws={"shrink": 0.725, "label": ""},
        annot_kws={"size": 18, "weight": "bold", "color": "white"},
        linewidths=0,
    )

    # --- colorbar tick/outline styling ---
    cbar = axis.collections[0].colorbar
    cbar.ax.tick_params(colors=LIGHT_GRAY, labelsize=18)
    cbar.outline.set_edgecolor(LIGHT_GRAY)

    # 3) Axis styling
    axis.xaxis.tick_top()
    axis.xaxis.set_label_position("top")

    plt.setp(axis.get_xticklabels(), rotation=45, fontsize=20, color=LIGHT_GRAY)
    plt.setp(axis.get_yticklabels(), rotation=45, fontsize=20, color=LIGHT_GRAY)

    axis.set_xlabel("Trained on", fontsize=24, color=LIGHT_GRAY, labelpad=20)
    axis.set_ylabel("Tested on", fontsize=24, color=LIGHT_GRAY, labelpad=20)
    axis.tick_params(axis="both", which="both", length=0, colors=LIGHT_GRAY)

    # --- hide axes frame/spines ---
    for spine in axis.spines.values():
        spine.set_visible(False)

    plt.tight_layout()

    # 4) Save or show
    if save_path:
        plt.savefig(
            save_path,
            format="pdf",
            bbox_inches="tight",
            pad_inches=0.1,
            transparent=True,
            facecolor="none",
            edgecolor="none",
        )
        print(f"Cross-device confusion heatmap saved to '{save_path}'.")
        plt.close()
    else:
        plt.show()


# ---------------------------------------------------------------------------
# 4) Compact two-method comparison with BCa intervals (Z-SCORE summaries)
# ---------------------------------------------------------------------------
def plot_accuracy_comparison_with_bca_intervals(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    unscaled_df: pl.DataFrame,
    scaled_df: pl.DataFrame,
    *,
    save_path: Path | None = None,
    dodge: float = 0.20,
    names: tuple[str, str] = ("Raw", "AGC-scaled"),
    colors: tuple[str, str] = (LIGHT_ORANGE, LIGHT_TEAL),
    symbols: tuple[str, str] = ("circle", "diamond"),
    height: int = 600,
    width: int = 1600,
    receiver_bands: bool = True,
    method_lanes: bool = False,
    inner_labels: bool = True,
    label_y: float = 0.98,
    label_clearance: float = 0.02,
    min_top_pad_frac: float = 0.12,
    bottom_pad_frac: float = 0.04,
    y_title: str = "Mean Accuracy",
    y_title_standoff: int = 32,
    y_title_font_size: int = 48,
    y_tick_font_size: int = 40,
    y_nticks: int = 6,
    y_range: tuple[float, float] | None = None,
) -> None:
    """
    Compact comparison plot (mean +/- BCa CI99) with two series (unscaled vs scaled),
    using Z-SCORE k-fold summaries.

    - Y-axis: mean_accuracy (with BCa intervals)
    - X-axis: receivers (columns), slightly offset per method (dodge)
    """

    # ---- config ----
    group_spacing = 1.5
    marker_cfg = dict(
        size=35, line=dict(width=MARKER_LINE_WIDTH, color=MARKER_LINE_COLOR)
    )
    error_cfg = dict(
        type="data", thickness=ERROR_BAR_THICKNESS, width=0, color=ERROR_BAR_COLOR
    )

    # 1) Prepare receiver order and BCa stats
    receivers = scaled_df.reverse()["receiver"].to_list()
    unscaled_stats = compute_bca_stats_by_receiver(unscaled_df)
    scaled_stats = compute_bca_stats_by_receiver(scaled_df)

    x_center_by_receiver = {
        rcv: idx * group_spacing for idx, rcv in enumerate(receivers)
    }
    offsets = (-dodge, +dodge)

    # 2) Create base figure and optional background bands / lanes
    figure = go.Figure()

    if receiver_bands and receivers:
        half_spacing = group_spacing / 2
        for index, receiver in enumerate(receivers):
            if index % 2 == 0:
                figure.add_shape(
                    type="rect",
                    x0=x_center_by_receiver[receiver] - half_spacing,
                    x1=x_center_by_receiver[receiver] + half_spacing,
                    y0=0,
                    y1=1,
                    yref="paper",
                    line=dict(width=0),
                    fillcolor="rgba(0,0,0,0.035)",
                    layer="below",
                )

    if method_lanes:
        for lane_index, offset in enumerate(offsets):
            _ = lane_index  # retained for potential debugging
            for receiver in receivers:
                figure.add_vline(
                    x=x_center_by_receiver[receiver] + offset,
                    line_width=1,
                    line_dash="dot",
                    line_color="rgba(0,0,0,0.15)",
                    layer="below",
                )

    # 3) Add scatter traces (markers + error bars) for both methods
    series_definitions = [
        (unscaled_stats, names[0], symbols[0], colors[0]),
        (scaled_stats, names[1], symbols[1], colors[1]),
    ]

    ymins: list[float] = []
    ymaxs: list[float] = []

    for series_index, (stats_map, series_name, symbol, color) in enumerate(
        series_definitions
    ):
        xs: list[float] = []
        ys: list[float] = []
        err_minus_list: list[float] = []
        err_plus_list: list[float] = []

        for receiver in receivers:
            mean, err_minus, err_plus = stats_map.get(receiver, (np.nan, 0.0, 0.0))
            xs.append(x_center_by_receiver[receiver] + offsets[series_index])
            ys.append(mean)
            err_minus_list.append(err_minus)
            err_plus_list.append(err_plus)

            if np.isfinite(mean):
                ymins.append(mean - err_minus)
                ymaxs.append(mean + err_plus)

        figure.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                name=series_name,
                marker={**marker_cfg, "symbol": symbol, "color": color},
                error_y={
                    **error_cfg,
                    "array": err_plus_list,
                    "arrayminus": err_minus_list,
                },
            )
        )

    if not ymins or not ymaxs:
        raise ValueError("No finite data available to plot BCa intervals.")

    # 4) Determine y-range with headroom so labels don't overlap markers
    data_min = float(np.nanmin(ymins))
    data_max = float(np.nanmax(ymaxs))
    data_range = max(data_max - data_min, 1e-12)
    bottom_padding = data_range * float(bottom_pad_frac)

    if y_range is None:
        safe_label_y = max(0.01, min(0.99, float(label_y) - float(label_clearance)))
        needed_top_pad = ((1.0 - safe_label_y) / safe_label_y) * (
            data_range + bottom_padding
        )
        top_padding = max(data_range * float(min_top_pad_frac), needed_top_pad)
        resolved_y_range: tuple[float, float] = (
            data_min - bottom_padding,
            data_max + top_padding,
        )
    else:
        resolved_y_range = y_range

    # 5) Layout & axes
    figure.update_layout(
        width=width,
        height=height,
        template="plotly_white",
        margin=dict(l=120, r=40, t=0, b=0, pad=0),
        legend=dict(
            orientation="h",
            x=0.98,
            xanchor="right",
            y=0.02,
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.6)",
            bordercolor="rgba(0,0,0,0.05)",
            borderwidth=1,
            font=dict(size=44, family=FONT_FAMILY, color="gray"),
        ),
    )

    tickvals = [x_center_by_receiver[receiver] for receiver in receivers]
    figure.update_xaxes(
        title=None,
        tickmode="array",
        tickvals=tickvals,
        ticktext=[] if inner_labels else receivers,
        showticklabels=not inner_labels,
        ticks="",
    )

    figure.update_yaxes(
        title=y_title,
        range=resolved_y_range,
        title_font=dict(size=y_title_font_size, family=FONT_FAMILY, color="gray"),
        tickfont=dict(size=y_tick_font_size, family=FONT_FAMILY, color="dimgray"),
        nticks=int(y_nticks),
        title_standoff=y_title_standoff,
        automargin=True,
    )

    figure.update_traces(
        line_width=5, marker_line_width=2, error_y_thickness=ERROR_BAR_THICKNESS
    )

    # 6) Receiver labels drawn inside the plot area near the top
    if inner_labels:
        for receiver in receivers:
            figure.add_annotation(
                x=x_center_by_receiver[receiver],
                xref="x",
                y=float(label_y),
                yref="paper",
                text=str(receiver),
                showarrow=False,
                xanchor="center",
                yanchor="top",
                font=dict(size=44, family=FONT_FAMILY, color="dimgray"),
                bgcolor="rgba(255,255,255,0.6)",
                bordercolor="rgba(0,0,0,0.05)",
                borderwidth=1,
            )

    # 7) Save or show
    if save_path:
        figure.write_image(save_path, width=width, height=height)
        print(f"Compact BCa comparison plot saved to '{save_path}'.")
    else:
        figure.show()


# ---------------------------------------------------------------------------
# 5) Off-diagonal cross-device accuracy distribution (per method)
# ---------------------------------------------------------------------------
def plot_crossdevice_distribution_per_method(
    cross_device_map: dict[str, pl.DataFrame],
    save_path: Path | None = None,
) -> None:
    """
    Horizontal lollipop chart per method, summarizing off-diagonal
    cross-device accuracies:

      - For each method, gather all accuracies where Trained on != Tested on.
      - Compute min, median, max.
      - Draw min->max as a line and median as a large dot.
    """
    # 1) Gather off-diagonal accuracies into a flat list
    records: list[dict[str, float | str]] = []
    for method_name, method_df in cross_device_map.items():
        for tested_device in RECEIVER_ORDER:
            for trained_device in RECEIVER_ORDER:
                if trained_device == tested_device:
                    continue

                value_series = method_df.filter(
                    (pl.col("test_receiver") == tested_device)
                    & (pl.col("train_receiver") == trained_device)
                )["accuracy"]

                if value_series.is_empty():
                    continue

                records.append(
                    {"Method": method_name, "Accuracy": float(value_series.item())}
                )

    # 2) Compute per-method min/median/max and sort by median
    distribution_df = pl.from_dicts(records).to_pandas()
    summary_stats = (
        distribution_df.groupby("Method")["Accuracy"]
        .agg(["min", "median", "max"])
        .sort_values("median")
    )
    methods_in_plot_order = summary_stats.index.tolist()

    # ---- config ----
    def rgba_to_hex(rgba: tuple[float, float, float, float]) -> str:
        r, g, b, _ = rgba
        return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))

    n_methods = len(methods_in_plot_order)
    colors = [rgba_to_hex(tgo_cmap_rev(i / (n_methods - 1))) for i in range(n_methods)]
    color_by_method = dict(zip(methods_in_plot_order, colors))

    marker_size = 50
    marker_line_cfg = dict(width=MARKER_LINE_WIDTH, color=MARKER_LINE_COLOR)
    line_cfg = dict(color=ERROR_BAR_COLOR, width=ERROR_BAR_THICKNESS)

    # 3) Build figure
    figure = go.Figure()

    for method in methods_in_plot_order:
        min_val, median_val, max_val = summary_stats.loc[
            method, ["min", "median", "max"]
        ]

        # line from min to max
        figure.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[method, method],
                mode="lines",
                line=line_cfg,
                showlegend=False,
            )
        )

        # median dot
        figure.add_trace(
            go.Scatter(
                x=[median_val],
                y=[method],
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=color_by_method[method],
                    line=dict(
                        width=marker_line_cfg["width"],
                        color=marker_line_cfg["color"],
                    ),
                ),
                showlegend=False,
            )
        )

    figure.update_layout(
        template="plotly_white",
        width=PLOT_WIDTH_WIDE,
        height=PLOT_HEIGHT_MEDIUM,
        margin=dict(l=320, r=80, t=0, b=50),
        xaxis=dict(
            title="Cross-device accuracy",
            range=[0.15, 1.0],
            title_font=dict(
                size=TITLE_FONT_SIZE_LARGE, family=FONT_FAMILY, color="gray"
            ),
            tickfont=dict(
                size=TITLE_FONT_SIZE_LARGE, family=FONT_FAMILY, color="dimgray"
            ),
        ),
        yaxis=dict(
            categoryorder="array",
            categoryarray=methods_in_plot_order,
            tickfont=dict(
                size=TITLE_FONT_SIZE_LARGE, family=FONT_FAMILY, color="dimgray"
            ),
        ),
    )

    figure.update_traces(
        line_width=line_cfg["width"],
        marker_line_width=marker_line_cfg["width"],
    )

    if save_path:
        figure.write_image(save_path, width=PLOT_WIDTH_WIDE, height=PLOT_HEIGHT_MEDIUM)
        print(f"Off-diagonal accuracy distribution plot saved to '{save_path}'.")
    else:
        figure.show()


# ---------------------------------------------------------------------------
# 7) Cross-device map loader
# ---------------------------------------------------------------------------
def load_cross_device_results_map(
    agc_ablations_df: pl.DataFrame,
) -> dict[str, pl.DataFrame]:
    """
    Split cross-device accuracies per dataset/agcmethod
    {method_name -> Polars DataFrame}.
    """
    # Map "method name" -> dataset key used in the flat CSV
    # fmt: off
    dataset_map = {
        # Rescaling for recovery or explicit removal of AGC
        "raw":    "unscaled",    # Raw data, not preprocessed
        "ℓ₁":     "agcscaled",   # Divided by l1-norm (mean abs values)
        "ℓ₂":     "rms",         # Divided by l2-norm (CSI power)
        "RSSI":   "rssi",        # Rescaled using RSSI
        "DBSCN":  "dbscan",      # DBScan
        "GINC":   "ratnam_1",    # Gain Increment Clustering
        "λ-grid": "ratnam_2",    # Uniform grid AGC ML-based optimization

        # Smoothing-based methods
        "hampel":  "hampel",     # Hampel filter
        "median":  "rollingmed", # Rolling median filter
        "savgol":  "savgol",     # Savitzky-Golay
        "wavelet": "wavelet",    # Wavelet denoising

        # Amplitude-invariant feature methods
        "morph": "morphology",   # Morphological features
        "ratio": "doublediff",   # Double ratio
    }
    # fmt: on

    cross_device_map: dict[str, pl.DataFrame] = {}
    for method_name, dataset_name in dataset_map.items():
        cross_device_map[method_name] = agc_ablations_df.filter(
            pl.col("dataset") == dataset_name
        )

    return cross_device_map


# ---------------------------------------------------------------------------
# 8) Orchestration
# ---------------------------------------------------------------------------
def generate_all_aril_figures(base_data_path: Path) -> None:
    """
    Orchestrate loading data and generating all ARIL plots.
    """
    on_device_dir = base_data_path / "aril_ondevice_accuracy"
    agc_ablations_dir = base_data_path / "aril_agcnorm_ablations"
    image_output_dir = base_data_path / "img"
    image_output_dir.mkdir(parents=True, exist_ok=True)

    # On-device summaries
    on_device_unscaled_none = load_on_device_summary_reordered(
        on_device_dir / "aril_unscaled_NONE_kfold_summary.csv"
    )
    on_device_unscaled_zscore = load_on_device_summary_reordered(
        on_device_dir / "aril_unscaled_ZSCORE_ACROSS_EXAMPLES_kfold_summary.csv"
    )
    on_device_agc_zscore = load_on_device_summary_reordered(
        on_device_dir / "aril_agcscaled_ZSCORE_ACROSS_EXAMPLES_kfold_summary.csv"
    )

    # 1) Single per-device accuracy (std) using NONE summary
    plot_on_device_accuracy_with_std(
        on_device_unscaled_none,
        save_path=image_output_dir / "aril-unscaled-ondevice.pdf",
    )

    # 2) Compact BCa comparison (Z-SCORE summaries: unscaled vs AGC-scaled)
    plot_accuracy_comparison_with_bca_intervals(
        on_device_unscaled_zscore,
        on_device_agc_zscore,
        save_path=image_output_dir / "aril-agc-comparison.pdf",
    )

    # Cross-device results for confusion-style heatmaps (unscaled + AGC)
    agc_ablations_file = (
        agc_ablations_dir / "aril_ZSCORE_ACROSS_EXAMPLES_cross_dataset_ablation.csv"
    )
    agc_ablations_df = pl.read_csv(agc_ablations_file)
    cross_device_unscaled = agc_ablations_df.filter(pl.col("dataset") == "unscaled")
    cross_device_agcscaled = agc_ablations_df.filter(pl.col("dataset") == "agcscaled")

    # 3) Confusion matrices (using Z-SCORE on-device diagonals)
    plot_cross_device_confusion_heatmap(
        build_cross_device_confusion_matrix(
            cross_device_unscaled,
            on_device_unscaled_zscore,
        ),
        save_path=image_output_dir / "aril-unscaled-confusion.pdf",
    )

    plot_cross_device_confusion_heatmap(
        build_cross_device_confusion_matrix(
            cross_device_agcscaled,
            on_device_agc_zscore,
        ),
        save_path=image_output_dir / "aril-agcscaled-confusion.pdf",
    )

    # 4) Ablation between agcnorms
    cross_device_map = load_cross_device_results_map(agc_ablations_df)

    plot_crossdevice_distribution_per_method(
        cross_device_map,
        save_path=image_output_dir / "aril-agcnorm-ablation.pdf",
    )


def main() -> None:
    """
    Entry point: call the ARIL figure generator with the default data path.
    """
    data_path = Path("data/har")
    generate_all_aril_figures(data_path)


if __name__ == "__main__":
    main()
