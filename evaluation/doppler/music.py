#!/usr/bin/env python3
"""
Doppler estimation and plotting pipeline.

This script supports three subcommands:

  1. calculate
       Loads all CSI data variants (scaled, unscaled, RSSI-scaled),
       computes Doppler estimates for the baseline and phasefit pipelines,
       and caches ALL estimates in a single parquet file.
       No plots are produced during calculation.

  2. plot_distributions
       Loads cached Doppler estimates and produces per-method
       distribution plots.

  3. plot_estimates
       Loads the cached Doppler estimates and produces a compact
       multi-method comparison plot.

All functions, documentation, and utilities from the original module
are preserved unless explicitly removed by request (e.g. plot_speed_boxplot).

All ASUS CSI visualizations are kept but their calls remain commented out.
"""

import argparse
from typing import Mapping, Sequence
from pathlib import Path

import numpy as np
import polars as pl
import seaborn as sns
import plotly.io as pio
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from evaluation.common import (
    LIGHT_GRAY,
    LIGHT_TEAL,
    LIGHT_ORANGE,
    RECEIVER_ORDER,
    fmlp_cmap_2,
    tgo_palette,
)
from matplotlib.colors import Normalize

from sensession.campaign.processor import CampaignProcessor

# Required by kaleido when exporting plotly figures
pio.defaults.mathjax = None


# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------

data_dir = Path.cwd() / "data" / "doppler_emulation_slow"
SHOW: bool = False


# ---------------------------------------------------------------------------
# CSI loading + preprocessing functions
# ---------------------------------------------------------------------------


def regroup(df: pl.DataFrame):
    """
    Regroup CSI rows by capture number, preserving order.
    """
    return df.group_by("capture_num", maintain_order=True).agg(
        pl.col("meta_id").first(),
        pl.col("receiver_name").first(),
        pl.col("timestamp").first(),
        pl.col("sequence_number").first(),
        "csi_abs",
        "csi_phase",
        "subcarrier_idxs",
        pl.col("rssi").first(),
    )


def load_csi_data(
    scale: bool = True,
    equalize: bool = False,
    rescale_rssi: bool = False,
    phase_fit: bool = False,
) -> pl.DataFrame:
    """
    Load CSI recordings and apply a selected preprocessing chain.
    """
    data = pl.read_parquet(data_dir / "csi.parquet")
    meta = pl.read_parquet(data_dir / "meta.parquet")

    proc = (
        CampaignProcessor(data, meta, lazy=False)
        .correct_rssi_by_agc()
        .unwrap()
        .filter("antenna_idxs", 0)
    )

    if phase_fit:
        proc = proc.detrend_phase_ls()
    else:
        proc = proc.detrend_phase(pin_edges=False)

    if scale:
        proc = proc.scale_magnitude()

    if rescale_rssi:
        proc = proc.rescale_csi_by_rssi(exclude_expr=pl.col("receiver_name") == "x310")

    if equalize:
        proc = proc.equalize_magnitude().equalize_phase()

    proc = proc.drop_contains("collection_name", "warmup")

    if not isinstance(proc.csi, pl.DataFrame):
        raise ValueError("CSI must be a concrete DataFrame.")

    return regroup(proc.csi)


# ---------------------------------------------------------------------------
# MUSIC Doppler estimation functions
# ---------------------------------------------------------------------------
def compute_music_spectrum(
    csi_slice: np.ndarray,
    gap_time: float,
    wavelength: float,
    candidate_speeds: np.ndarray,
    num_packets: int,
    num_targets: int,
) -> np.ndarray:
    """
    Compute the normalized MUSIC spectrum for a CSI block.

    For each candidate speed v, the pseudo-spectrum is defined as

        Spectrum(v) = 1 / ( s(v)^H · P_noise · s(v) )

    where

        s(v)      = exp(2j·π·gap_time·v·(n / wavelength))  for n = 0,...,num_packets-1
        P_noise   = U · U^H  is the projection onto the noise subspace
        U         collects the eigenvectors associated with the smallest
                   eigenvalues of the space-time covariance matrix

        R = csi_slice · csi_slice^H.

    The output is converted to decibels and normalized to the range [0, 1].
    """

    # ------------------------------------------------------------------
    # 1) Covariance matrix and noise projection matrix
    # ------------------------------------------------------------------
    # R = X X^H, where X = csi_slice (packets x subcarriers)
    covariance = csi_slice @ csi_slice.conj().T

    # Eigen-decompose R. Columns of `eigenvectors` are eigenvectors.
    _, eigenvectors = np.linalg.eigh(covariance)

    # Take the eigenvectors corresponding to the smallest eigenvalues
    # as an estimate of the noise subspace.
    noise_subspace = eigenvectors[:, : eigenvectors.shape[0] - num_targets]

    # P_noise = U U^H
    noise_projection = noise_subspace @ noise_subspace.conj().T

    # ------------------------------------------------------------------
    # 2) Steering matrix s(v) for all candidate speeds v
    # ------------------------------------------------------------------
    # For each packet index n (0..num_packets-1), build the term n / wavelength.
    comb = np.arange(num_packets) / wavelength

    # Each row of `steering_matrix` is
    #   s(v) = exp(2j·π·gap_time·v·(n / wavelength))
    steering_matrix = np.exp(2j * np.pi * gap_time * candidate_speeds[:, None] * comb)

    # ------------------------------------------------------------------
    # 3) MUSIC pseudo-spectrum:
    #       Spectrum(v) = 1 / ( s(v)^H · P_noise · s(v) )
    # ------------------------------------------------------------------
    projected = noise_projection @ steering_matrix.T
    spectrum = 1.0 / np.sum(steering_matrix.conj() * projected.T, axis=1)

    # ------------------------------------------------------------------
    # 4) Convert the spectrum to decibels and normalize to [0, 1]
    # ------------------------------------------------------------------
    spectrum_db = 10.0 * np.log10(np.abs(spectrum))
    spectrum_db -= spectrum_db.min()
    max_val = spectrum_db.max()
    if max_val:
        spectrum_db /= max_val

    return spectrum_db


def estimate_speed(spectrum: np.ndarray, candidate_speeds: np.ndarray) -> float:
    """
    Return the candidate speed that maximizes the MUSIC spectrum.
    """
    return candidate_speeds[np.argmax(spectrum)]


def doppler_estimates(
    data: pl.DataFrame,
    figure_name: str,
    plot_distributions: bool = False,
) -> dict[str, list[float]]:
    """
    Compute Doppler estimates blockwise for each receiver.

    Returns:
        { receiver_name : [estimated speeds] }

    If plot_distributions=True, immediately calls plot_estimate_distributions().
    """
    num_packets = 50
    center_frequency = 2.462e9
    light_speed = 299_792_458
    wavelength = light_speed / center_frequency
    candidate_speeds = np.arange(0, 5.01, 0.0001)

    results: dict[str, list[float]] = {}
    time_indices: dict[str, np.ndarray] = {}

    for receiver_key, group in data.group_by("receiver_name", maintain_order=True):
        receiver_name = str(receiver_key[0])

        # ------------------------------------------------------------------
        # Timestamps / time base
        # ------------------------------------------------------------------
        # Convert timestamps from sequence numbers to a consistent float
        # timebase. Conceptually there are two options:
        #
        #   (A) "Ideal" timestamps based on prior knowledge that the TX
        #       sends regularly at ~500 Hz:
        #
        #       seq_nums = np.unwrap(
        #           np.array(group["sequence_number"].to_list()),
        #           period=4096,
        #       )
        #       ideal_timestamps = seq_nums / 500.0
        #
        #   (B) Use the reported wall-clock timestamps from the capture
        #       (what we actually use below).
        timestamps = np.array([ts.timestamp() for ts in group["timestamp"]])

        # ------------------------------------------------------------------
        # CSI: magnitude/phase -> complex, zero-mean per subcarrier
        # ------------------------------------------------------------------
        # `csi_abs` and `csi_phase` are Polars list columns; convert them
        # to proper 2D NumPy arrays [num_packets, num_subcarriers].
        csi_abs = np.array(group["csi_abs"].to_list())
        csi_phase = np.array(group["csi_phase"].to_list())

        csi_complex = csi_abs * np.exp(1j * csi_phase)
        csi_complex -= np.mean(csi_complex, axis=0, keepdims=True)

        total_packets = csi_complex.shape[0]
        num_blocks = total_packets // num_packets

        estimated_speeds: list[float] = []
        block_times: list[float] = []

        for block_idx in range(num_blocks):
            start = block_idx * num_packets
            end = (block_idx + 1) * num_packets
            csi_slice = csi_complex[start:end]

            # Effective time spacing between packets in this block.
            gap_time = (timestamps[end - 1] - timestamps[start]) / (num_packets - 1)

            spectrum = compute_music_spectrum(
                csi_slice=csi_slice,
                gap_time=gap_time,
                wavelength=wavelength,
                candidate_speeds=candidate_speeds,
                num_packets=num_packets,
                num_targets=1,
            )

            speed = estimate_speed(spectrum, candidate_speeds)
            estimated_speeds.append(speed)

            # Store elapsed time since beginning of capture (seconds)
            block_times.append(timestamps[end - 1] - timestamps[0])

        results[receiver_name] = estimated_speeds
        time_indices[receiver_name] = np.array(block_times, dtype=np.float64)

    if plot_distributions:
        plot_estimate_distributions(results, figure_name)

    return results


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------
def plot_estimate_distributions(results: dict[str, list[float]], figure_name: str):
    """Plot the distribution of speed estimates per receiver as a box plot.
    Args:
        results: Dictionary mapping receiver names to lists of estimated speeds.
    """
    # Redo this.. I feel like there is a simpler way.
    estimates = pl.DataFrame(
        {
            "Receiver": results.keys(),
            "Estimated Speed": results.values(),
        }
    ).explode("Estimated Speed")

    # Identify which receivers are actually present
    receivers = estimates.unique("Receiver").get_column("Receiver").to_list()
    receivers = [r for r in RECEIVER_ORDER if r in receivers]

    plt.figure(figsize=(10, 4))
    boxplot = sns.boxplot(
        y="Receiver",
        x="Estimated Speed",
        data=estimates,
        linewidth=1.5,
        width=0.6,
        showfliers=False,
        boxprops={"edgecolor": "black"},
        whiskerprops={"linewidth": 1.5},
        capprops={"linewidth": 1.5},
        medianprops={"color": "black", "linewidth": 1.5},
        patch_artist=True,
        order=receivers,
    )

    # Redo this. This function doesnt exist anymore.
    stats = estimates.group_by("Receiver").agg(
        pl.col("Estimated Speed").median().alias("Median Speed")
    )

    # NOTE: 1.0 is hardcoded ground truth.
    norm = Normalize(stats.select("Median Speed").min().item(), 1.0)

    # Color each box by its median speed
    for patch, receiver in zip(boxplot.patches, receivers):
        median_value = stats.filter(pl.col("Receiver") == receiver)[
            "Median Speed"
        ].item()
        patch.set_facecolor(fmlp_cmap_2(norm(median_value)))

    sm = plt.cm.ScalarMappable(cmap=fmlp_cmap_2, norm=norm)
    cbar = plt.colorbar(sm, ax=plt.gca(), pad=0.02)
    cbar.set_label("Median Speed (m/s)", fontsize=16, color="gray")
    cbar.ax.tick_params(labelsize=16)

    plt.ylabel("Receiver", fontsize=19)
    plt.xlabel("Estimated Speed (m/s)", fontsize=19)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()

    # plt.xlim(0.99, 1.01)
    plt.savefig(
        data_dir / f"{figure_name}.pdf",
        format="pdf",
        bbox_inches="tight",
        dpi=300,
    )
    if SHOW:
        plt.show()


# ---------------------------------------------------------------------------
# Compact pointrange plot
# ---------------------------------------------------------------------------
def plot_speed_pointrange_compact(
    speed_samples_by_method: Mapping[str, Mapping[str, Sequence[float]]],
    *,
    save_path: Path | None = None,
    dodge: float = 0.27,  # horizontal offset between methods (within a receiver column)
    inner_labels: bool = True,  # put receiver names inside the figure (top)
    show_receiver_bands: bool = True,  # light background strips per receiver
    show_method_lanes: bool = False,  # faint vertical guides at each method x-position
    height: int = 600,
    width: int = 1600,
    # Label placement + headroom tuning
    label_y: float = 0.99,  # vertical position for receiver names (paper coords 0..1)
    label_clearance: float = 0.01,  # keep top-of-data at least this far below label_y (paper coords)
    min_top_pad_frac: float = 0.12,  # min top padding as fraction of data range
    bottom_pad_frac: float = 0.04,  # small bottom pad so markers/error bars don't clip
    # Fonts / ticks
    receiver_label_font_size: int = 44,
    legend_font_size: int = 44,
    y_tick_font_size: int = 36,
    y_title_font_size: int = 46,
    y_title_standoff: int = 32,
    y_nticks: int = 6,
) -> None:
    """
    Compact point-range plot (median +/- IQR) with values on Y and receivers on X.

    - Each method is shown as points with vertical error bars (median +/- IQR).
    - Receivers form groups along the X axis.
    - Labels for receivers can be drawn inside the plot at the top (inner_labels=True).
    """

    # -------------------------------------------------------------------------
    # Helper: summarize samples into median and IQR-based error bars
    # -------------------------------------------------------------------------
    def summarize_samples(samples: Sequence[float]) -> tuple[float, float, float]:
        """
        Return (median, iqr_minus, iqr_plus) for a sequence of samples.

        iqr_minus = median - Q1
        iqr_plus  = Q3 - median
        """
        if not samples:
            return np.nan, 0.0, 0.0

        q1, q3 = np.percentile(samples, (25, 75))
        median = float(np.median(samples))

        iqr_minus = median - q1
        iqr_plus = q3 - median

        return median, iqr_minus, iqr_plus

    # -------------------------------------------------------------------------
    # Basic validation + extraction of names
    # -------------------------------------------------------------------------
    method_names = list(speed_samples_by_method.keys())
    num_methods = len(method_names)

    if num_methods < 2:
        raise ValueError("Need at least two methods for a comparison plot.")

    # Collect all receiver names that appear in any method
    receiver_names: list[str] = sorted(
        {
            receiver_name
            for method_data in speed_samples_by_method.values()
            for receiver_name in method_data.keys()
        }
    )

    if not receiver_names:
        raise ValueError("No receivers found in the input data.")

    # -------------------------------------------------------------------------
    # Compute X positions for each receiver group and offsets per method
    # -------------------------------------------------------------------------
    group_spacing = 1.5  # horizontal distance between receiver groups

    # Base X position for the center of each receiver group
    receiver_x_position: dict[str, float] = {
        receiver_name: group_index * group_spacing
        for group_index, receiver_name in enumerate(receiver_names)
    }

    # Horizontal offsets so methods are side-by-side within each receiver group
    # e.g. for 3 methods, offsets might be [-d, 0, +d]
    method_offsets = np.linspace(
        start=-(num_methods - 1),
        stop=(num_methods - 1),
        num=num_methods,
    ) * (dodge / 2.0)

    # -------------------------------------------------------------------------
    # Figure and style configuration
    # -------------------------------------------------------------------------
    figure = go.Figure()

    # Marker style (size and outline)
    marker_style = dict(
        size=35,
        line=dict(width=2, color="#A9A9A9"),
    )

    # Error-bar style
    error_bar_style = dict(
        type="data",
        thickness=5,
        width=0,
        color="#636363",
    )

    palette = [
        LIGHT_GRAY,
        LIGHT_TEAL,
        LIGHT_ORANGE,
    ]

    marker_symbols = [
        "circle",
        "square",
        "diamond",
        "triangle-up",
        "x",
        "cross",
        "star",
        "triangle-down",
    ]

    # -------------------------------------------------------------------------
    # Optional alternating receiver background bands
    # -------------------------------------------------------------------------
    if show_receiver_bands and receiver_names:
        half_group_width = group_spacing / 2.0

        for receiver_index, receiver_name in enumerate(receiver_names):
            # Shade every second receiver band
            if receiver_index % 2 != 0:
                continue

            x_center = receiver_x_position[receiver_name]

            figure.add_shape(
                type="rect",
                x0=x_center - half_group_width,
                x1=x_center + half_group_width,
                y0=0.0,
                y1=1.0,
                yref="paper",  # stretch from bottom to top of plotting area
                line=dict(width=0),
                fillcolor="rgba(0,0,0,0.035)",
                layer="below",
            )

    # -------------------------------------------------------------------------
    # Optional vertical guide lines for each method lane
    # -------------------------------------------------------------------------
    if show_method_lanes:
        for method_index in range(num_methods):
            method_offset = method_offsets[method_index]

            for receiver_name in receiver_names:
                x_position = receiver_x_position[receiver_name] + method_offset

                figure.add_vline(
                    x=x_position,
                    line_width=1,
                    line_dash="dot",
                    line_color="rgba(0,0,0,0.15)",
                    layer="below",
                )

    # -------------------------------------------------------------------------
    # Main traces: one scatter per method (values on Y, receivers on X)
    # -------------------------------------------------------------------------
    y_min_values: list[float] = []
    y_max_values: list[float] = []

    for method_index, method_name in enumerate(method_names):
        method_data = speed_samples_by_method[method_name]

        x_values: list[float] = []
        medians: list[float] = []
        error_minus_values: list[float] = []
        error_plus_values: list[float] = []

        for receiver_name in receiver_names:
            receiver_samples = method_data.get(receiver_name, [])

            median, iqr_minus, iqr_plus = summarize_samples(receiver_samples)

            x_position = (
                receiver_x_position[receiver_name] + method_offsets[method_index]
            )

            x_values.append(x_position)
            medians.append(median)
            error_minus_values.append(iqr_minus)
            error_plus_values.append(iqr_plus)

            if np.isfinite(median):
                y_min_values.append(median - iqr_minus)
                y_max_values.append(median + iqr_plus)

        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=medians,
                mode="markers",
                name=method_name,
                marker=dict(
                    **marker_style,
                    symbol=marker_symbols[method_index % len(marker_symbols)],
                    color=palette[method_index % len(palette)],
                ),
                error_y=dict(
                    **error_bar_style,
                    array=error_plus_values,
                    arrayminus=error_minus_values,
                ),
            )
        )

    if not y_min_values or not y_max_values:
        raise ValueError("No finite data to plot.")

    # -------------------------------------------------------------------------
    # Compute Y-range with bottom padding and enough top headroom
    # so data does not collide with receiver labels.
    # -------------------------------------------------------------------------
    data_min = float(np.nanmin(y_min_values))
    data_max = float(np.nanmax(y_max_values))

    data_range = max(data_max - data_min, 1e-12)
    bottom_padding = data_range * float(bottom_pad_frac)

    # Convert label position and clearance into a required padding above data.
    # L is the fraction of the plotting area used for the data (up to the label zone).
    label_floor = float(label_y) - float(label_clearance)
    label_floor = max(0.01, min(0.99, label_floor))

    # Derived from the mapping between data coordinates and paper coordinates.
    required_top_padding = ((1.0 - label_floor) / label_floor) * (
        data_range + bottom_padding
    )
    top_padding_from_fraction = data_range * float(min_top_pad_frac)

    top_padding = max(top_padding_from_fraction, required_top_padding)

    figure.update_yaxes(range=[data_min - bottom_padding, data_max + top_padding])

    # -------------------------------------------------------------------------
    # Layout configuration (axes, legend, margins)
    # -------------------------------------------------------------------------
    x_tick_positions = [receiver_x_position[name] for name in receiver_names]

    if inner_labels:
        x_tick_labels = ["" for _ in receiver_names]
        show_x_tick_labels = False
    else:
        x_tick_labels = receiver_names
        show_x_tick_labels = True

    figure.update_layout(
        width=width,
        height=height,
        template="plotly_white",
        xaxis=dict(
            title=None,
            tickmode="array",
            tickvals=x_tick_positions,
            ticktext=x_tick_labels,
            showticklabels=show_x_tick_labels,
            ticks="",
            tickfont=dict(
                size=receiver_label_font_size,
                family="Arial",
                color="dimgray",
            ),
        ),
        yaxis=dict(
            title="Estimated speed [m/s]",
            tickfont=dict(
                size=y_tick_font_size,
                family="Arial",
                color="dimgray",
            ),
            title_font=dict(
                size=y_title_font_size,
                family="Arial",
                color="gray",
            ),
            nticks=int(y_nticks),
        ),
        legend=dict(
            orientation="h",  # <-- FIX: must be 'h' or 'v', not 'horizontal'
            x=0.98,
            xanchor="right",
            y=0.02,
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.6)",
            bordercolor="rgba(0,0,0,0.05)",
            borderwidth=1,
            font=dict(
                size=legend_font_size,
                family="Arial",
                color="gray",
            ),
        ),
        margin=dict(
            l=120,
            r=40,
            t=0,
            b=0,
            pad=0,
        ),
    )

    figure.update_traces(
        line_width=5,
        marker_line_width=2,
        error_y_thickness=5,
    )

    figure.update_yaxes(
        title_standoff=y_title_standoff,
    )

    # Reference line at y=1
    figure.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="#b4b4b4",
        line_width=3,
        layer="below",
    )

    # -------------------------------------------------------------------------
    # Inner receiver labels at the top of the plot area
    # -------------------------------------------------------------------------
    if inner_labels:
        for receiver_name in receiver_names:
            x_position = receiver_x_position[receiver_name]

            figure.add_annotation(
                x=x_position,
                xref="x",
                y=float(label_y),
                yref="paper",
                text=receiver_name,
                showarrow=False,
                xanchor="center",
                yanchor="top",
                font=dict(
                    size=receiver_label_font_size,
                    family="Arial",
                    color="dimgray",
                ),
                bgcolor="rgba(255,255,255,0.6)",
                bordercolor="rgba(0,0,0,0.05)",
                borderwidth=1,
            )

    # -------------------------------------------------------------------------
    # Output: save to file or show
    # -------------------------------------------------------------------------
    if save_path is not None:
        figure.write_image(str(save_path), width=width, height=height)
    else:
        figure.show()


def plot_error_compact(
    methods: dict[str, dict[str, list[float]]],
    *,
    groundtruth: float = 1.0,
    save_path: Path | None = None,
    dodge: float = 0.27,
    height: int = 800,
    width: int = 1600,
):
    """
    Compact plot of |groundtruth - median(speed)| per receiver and method.

    For each receiver + method:
      - median_estimate = median(speed)
      - center_error    = |groundtruth - median_estimate|
      - errors          = |groundtruth - speed|
      - q1_err, q3_err  = 25th / 75th percentiles of errors

    Marker = center_error.
    Error bars span min(q1_err, center_error) .. max(q3_err, center_error).
    """
    # methods keys may include padding; map to canonical labels
    canonical_to_label: dict[str, str] = {}
    for label in methods.keys():
        canonical = label.strip()
        if canonical not in canonical_to_label:
            canonical_to_label[canonical] = label

    canonical_order = ["phase cleaned", "+RSSI-scaled", "+AGC-removed"]
    canonical_methods = [m for m in canonical_order if m in canonical_to_label]
    if not canonical_methods:
        raise ValueError(
            "No known methods found. Expected labels with .strip() in "
            "{'Raw', 'RSSI-scaled', 'AGC-removed'}."
        )

    # Colors and marker symbols from shared palette
    color_map = {
        "phase cleaned": tgo_palette[2],  # light grey
        "+RSSI-scaled": tgo_palette[1],  # teal
        "+AGC-removed": tgo_palette[3],  # orange
    }
    symbol_map = {
        "phase cleaned": "circle",
        "+RSSI-scaled": "square",
        "+AGC-removed": "diamond",
    }

    # Receivers in RECEIVER_ORDER, restricted to those that appear anywhere
    all_receivers: set[str] = set()
    for per_receiver in methods.values():
        all_receivers.update(per_receiver.keys())

    receivers = [r for r in RECEIVER_ORDER if r in all_receivers]
    if not receivers:
        raise ValueError("No receivers found in any method dictionary.")

    group_spacing = 1.5
    base_x = {rcv: i * group_spacing for i, rcv in enumerate(receivers)}
    n_methods = len(canonical_methods)
    offsets = np.linspace(-(n_methods - 1), n_methods - 1, n_methods) * (dodge / 2)

    def _summary_error(samples: Sequence[float]):
        """
        Center = |gt - median(speed)|.
        Bars span min(q1_err, center) .. max(q3_err, center),
        where q1_err/q3_err are quantiles of |gt - speed|.
        Returns (center, err_minus, err_plus).
        """
        if not samples:
            return np.nan, 0.0, 0.0, 0.0, 0.0

        arr = np.asarray(samples, dtype=float)
        median_estimate = float(np.median(arr))
        center_error = abs(groundtruth - median_estimate)

        errors = np.abs(arr - groundtruth)
        q1_err, q3_err = np.percentile(errors, (25, 75))

        bottom = min(q1_err, center_error)
        top = max(q3_err, center_error)

        err_minus = max(center_error - bottom, 0.0)
        err_plus = max(top - center_error, 0.0)

        return center_error, err_minus, err_plus, bottom, top

    fig = go.Figure()
    marker_cfg = dict(size=30, line=dict(width=2, color="#A9A9A9"))
    error_cfg = dict(type="data", thickness=4, width=0)

    all_y_min: list[float] = []
    all_y_max: list[float] = []

    for k, canonical in enumerate(canonical_methods):
        label = canonical_to_label[canonical]
        per_receiver = methods[label]

        xs: list[float] = []
        centers: list[float] = []
        err_minus: list[float] = []
        err_plus: list[float] = []

        for rcv in receivers:
            center_error, minus, plus, bottom, top = _summary_error(
                per_receiver.get(rcv, [])
            )
            x_pos = base_x[rcv] + offsets[k]

            xs.append(x_pos)
            centers.append(center_error)
            err_minus.append(minus)
            err_plus.append(plus)

            if np.isfinite(bottom):
                all_y_min.append(bottom)
            if np.isfinite(top):
                all_y_max.append(top)

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=centers,
                mode="markers",
                name=canonical,
                marker={
                    **marker_cfg,
                    "symbol": symbol_map.get(canonical, "circle"),
                    "color": color_map.get(canonical, tgo_palette[0]),
                },
                error_y={
                    **error_cfg,
                    "array": err_plus,
                    "arrayminus": err_minus,
                    "color": color_map.get(canonical, tgo_palette[0]),
                },
            )
        )

    # Y-axis range: no clipping, a bit of padding around full [min, max] of bars
    if all_y_min and all_y_max:
        data_min = float(np.nanmin(all_y_min))
        data_max = float(np.nanmax(all_y_max))
        if not np.isfinite(data_min):
            data_min = 0.0
        if not np.isfinite(data_max):
            data_max = 1.0

        span = max(data_max - data_min, 1e-6)
        pad = max(0.08 * span, 1e-3)
        y_min = data_min - pad
        y_max = data_max + pad
    else:
        y_min, y_max = 0.0, 1.0

    fig.update_yaxes(
        range=[y_min, y_max],
        title=f"|v̂_median - {groundtruth:.2f}| [m/s]",
        tickfont=dict(size=32, family="Arial", color="dimgray"),
        title_font=dict(size=36, family="Arial", color="gray"),
    )

    fig.update_layout(
        width=width,
        height=height,
        template="plotly_white",
        xaxis=dict(
            title=None,
            tickmode="array",
            tickvals=[base_x[r] for r in receivers],
            ticktext=receivers,
            showticklabels=True,
            ticks="",
            tickfont=dict(size=32, family="Arial", color="dimgray"),
        ),
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top",
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="rgba(0,0,0,0.05)",
            borderwidth=1,
            font=dict(size=32, family="Arial", color="gray"),
        ),
        margin=dict(l=120, r=40, t=80, b=40, pad=0),
    )

    fig.update_traces(marker_line_width=2, error_y_thickness=4)

    if save_path is not None:
        fig.write_image(save_path, width=width, height=height)
    else:
        fig.show()


# ---------------------------------------------------------------------------
# Flattening + caching utilities
# ---------------------------------------------------------------------------


def _methods_to_df(
    methods: dict[str, dict[str, list[float]]],
    phase_label: str,
) -> pl.DataFrame:
    """
    Flatten nested storage into a single Polars DataFrame.
    """
    rows = []
    for pipeline_name, per_receiver in methods.items():
        for receiver_name, samples in per_receiver.items():
            for est in samples:
                rows.append(
                    {
                        "receiver": receiver_name,
                        "pipeline": pipeline_name.strip(),
                        "phase": phase_label,
                        "estimated_speed": float(est),
                    }
                )

    if not rows:
        return pl.DataFrame(
            {
                "receiver": pl.Series([], dtype=pl.Utf8),
                "pipeline": pl.Series([], dtype=pl.Utf8),
                "phase": pl.Series([], dtype=pl.Utf8),
                "estimated_speed": pl.Series([], dtype=pl.Float64),
            }
        )

    return pl.DataFrame(rows)


def save_results(methods_baseline, methods_phasefit, path: Path):
    """
    Save all Doppler estimates (baseline and phasefit) to parquet.
    """
    df_base = _methods_to_df(methods_baseline, phase_label="baseline")
    df_pf = _methods_to_df(methods_phasefit, phase_label="phasefit")
    df_all = pl.concat([df_base, df_pf], how="vertical_relaxed")
    df_all.write_parquet(path)


def load_cached_results(path: Path):
    """
    Load cached Doppler estimates and reconstruct nested dicts:

        baseline_methods, phasefit_methods
    """
    df = pl.read_parquet(path)

    def build(phase: str):
        df_sub = df.filter(pl.col("phase") == phase)
        nested: dict = {}
        for pipeline, g1 in df_sub.group_by("pipeline"):
            pipeline_name = pipeline[0]
            nested[pipeline_name] = {}
            for receiver, g2 in g1.group_by("receiver"):
                receiver_name = receiver[0]
                nested[pipeline_name][receiver_name] = g2["estimated_speed"].to_list()
        return nested

    return build("baseline"), build("phasefit")


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Doppler estimation pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "calculate", help="Compute Doppler estimates and save to parquet."
    )
    subparsers.add_parser(
        "plot_distributions", help="Plot per-method distributions from cached parquet."
    )
    subparsers.add_parser(
        "plot_estimates", help="Plot compact pointrange plot from cached parquet."
    )

    args = parser.parse_args()

    estimates_path = data_dir / "doppler_estimates.parquet"

    # ------------------------------------------------------
    # calculate
    # ------------------------------------------------------
    if args.command == "calculate":
        print("[INFO] Loading and processing CSI...")

        data_scaled = load_csi_data(scale=True)
        data_unscaled = load_csi_data(scale=False)
        data_rssiscaled = load_csi_data(scale=True, rescale_rssi=True)

        results_scaled = doppler_estimates(data_scaled, "scaled")
        results_unscaled = doppler_estimates(data_unscaled, "unscaled")
        results_rssiscaled = doppler_estimates(data_rssiscaled, "rssiscaled")

        data_scaled_pf = load_csi_data(scale=True, phase_fit=True)
        data_unscaled_pf = load_csi_data(scale=False, phase_fit=True)
        data_rssiscaled_pf = load_csi_data(
            scale=True, rescale_rssi=True, phase_fit=True
        )

        results_scaled_pf = doppler_estimates(data_scaled_pf, "scaled-phasefit")
        results_unscaled_pf = doppler_estimates(data_unscaled_pf, "unscaled-phasefit")
        results_rssiscaled_pf = doppler_estimates(
            data_rssiscaled_pf, "rssiscaled-phasefit"
        )

        methods = {
            "phase cleaned   ": results_unscaled,
            "+AGC-removed   ": results_scaled,
            "+RSSI-scaled": results_rssiscaled,
        }

        methods_phasefit = {
            "phase cleaned   ": results_unscaled_pf,
            "+AGC-removed   ": results_scaled_pf,
            "+RSSI-scaled": results_rssiscaled_pf,
        }

        save_results(
            methods,
            methods_phasefit,
            estimates_path,
        )

        print(f"[OK] Saved Doppler estimates to {estimates_path}")
        return

    # ------------------------------------------------------
    # plot_distributions
    # ------------------------------------------------------
    if args.command == "plot_distributions":
        if not estimates_path.exists():
            raise FileNotFoundError("Run `calculate` first; no cached parquet found.")

        baseline_methods, _ = load_cached_results(estimates_path)

        print("[INFO] Creating distribution plots...")
        for method_name, per_receiver in baseline_methods.items():
            sanitized = method_name.replace(" ", "_")
            print(f"[PLOT] {method_name}")
            plot_estimate_distributions(
                per_receiver,
                figure_name=f"distribution_{sanitized}",
            )

        print("[OK] Distribution plots generated.")
        return

    # ------------------------------------------------------
    # plot_estimates
    # ------------------------------------------------------
    if args.command == "plot_estimates":
        if not estimates_path.exists():
            raise FileNotFoundError("Run `calculate` first; no cached parquet found.")

        _, baseline_methods = load_cached_results(estimates_path)

        # Rebuild methods dict in the exact order + labels you used originally
        methods_for_plot = {
            "phase cleaned   ": baseline_methods.get("phase cleaned", {}),
            "+RSSI-scaled": baseline_methods.get("+RSSI-scaled", {}),
            "+AGC-removed   ": baseline_methods.get("+AGC-removed", {}),
        }

        plot_speed_pointrange_compact(
            methods_for_plot,
            save_path=data_dir / "methods-compared.pdf",
        )
        print("[OK] Saved methods-compared.pdf")
        return


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
