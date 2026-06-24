#!/usr/bin/env python3
"""
CLI for time-of-flight (ToF) estimation on CSI data and generation of paper plots:
ground truth vs estimates, normalization, PDP waterfalls, and phase-delta matrices.
"""

from __future__ import annotations

import argparse
from typing import Any
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import plotly.io as pio
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from loguru import logger
from matplotlib.lines import Line2D
from evaluation.common import LIGHT_TEAL, LIGHT_ORANGE, RECEIVER_ORDER, tgo_cmap_rev
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import interp1d
from evaluation.tof.pdp import compute_pdp, compute_delays
from evaluation.tof.tof import ToFStats, TOFConfig, TOFProcessor

from sensession.campaign.processor import CampaignProcessor

# Required by kaleido when exporting plotly figures
pio.defaults.mathjax = None


# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------
data_dir = Path.cwd() / "data" / "tof_linear_doppler"
CSI_PATH = data_dir / "csi.parquet"
META_PATH = data_dir / "meta.parquet"

STATS_PATH = data_dir / "tof_stats.parquet"

CARRIER_SPACING = 312.5e3  # Hz
PDP_RANGE_NS = (-1000.0, 1000.0)  # ns
PAD_FACTOR = 1

SHOW = False  # set to True for interactive plt.show()

NUM_WIREFRAME = 50
PDP_CUTOFF = 0.4
SEQ_PERIOD = 4096  # sequence number period

# Phase-cleaning comparison: (raw, equalized) x (phase variant)
PHASE_PIPELINES = ["phase cleaned", "+Equalized"]


# ===========================================================================
# 1. ToF-specific helpers: DC imputation + CSI extraction
# ===========================================================================


def impute_missing_subcarriers(channel: np.ndarray) -> np.ndarray:
    """
    Impute the missing DC subcarrier in a (56, N) CSI matrix and return a (57, N) matrix.

    Assumes tones for indices [-28..-1, 1..28] and interpolates DC (0).
    """
    known = np.concatenate((np.arange(-28, 0), np.arange(1, 29)))
    missing = np.array([0])

    interp_real = interp1d(
        known,
        channel.real,
        axis=0,
        kind="linear",
        fill_value="extrapolate",  # type: ignore
    )
    interp_imag = interp1d(
        known,
        channel.imag,
        axis=0,
        kind="linear",
        fill_value="extrapolate",  # type: ignore
    )
    imputed = interp_real(missing) + 1j * interp_imag(missing)

    m = int(known.min())
    full_channel = np.zeros(
        (len(known) + len(missing), channel.shape[1]),
        dtype=channel.dtype,
    )
    full_channel[missing - m] = imputed
    full_channel[known - m] = channel

    return full_channel


def regroup(df: pl.DataFrame) -> pl.DataFrame:
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


def load_csi_data(normalize: bool, phase_variant: str = "ls") -> pl.DataFrame:
    """
    Load CSI and meta, apply phase detrending and optional equalization, and return a CSI DataFrame.
    """
    if not CSI_PATH.exists():
        raise FileNotFoundError(f"Missing CSI parquet at {CSI_PATH}")
    if not META_PATH.exists():
        raise FileNotFoundError(f"Missing meta parquet at {META_PATH}")

    csi = pl.read_parquet(CSI_PATH)
    meta = pl.read_parquet(META_PATH)

    proc = CampaignProcessor(csi, meta, lazy=False).unwrap().filter("antenna_idxs", 0)

    if phase_variant == "ls":
        proc = proc.detrend_phase_ls()
    elif phase_variant == "pads":
        proc = proc.detrend_phase()
    else:
        raise ValueError(f"Unknown phase_variant {phase_variant!r}")

    if normalize:
        proc = proc.equalize_magnitude().equalize_phase()

    proc = proc.drop_contains("collection_name", "warmup")

    if not isinstance(proc.csi, pl.DataFrame):
        raise ValueError("CampaignProcessor.csi must be a concrete DataFrame.")

    return regroup(proc.csi)


def csi_per_receiver(df: pl.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Convert a CSI DataFrame into per-receiver (channel, sequence) pairs.

    Returned channel shape: (subcarrier, sample)
    """
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for (rx_name,), group in df.group_by("receiver_name", maintain_order=True):
        rx_name = str(rx_name)

        abs_vals = np.array(group["csi_abs"].to_list())
        phs_vals = np.array(group["csi_phase"].to_list())
        csi_complex = abs_vals * np.exp(1j * phs_vals)

        # Use first antenna / stream.
        csi_complex = csi_complex[:, 0, 0, :]

        # impute DC subcarrier
        channel = csi_complex.transpose()  # (subcarrier, sample)
        channel = impute_missing_subcarriers(channel)

        # Drop final 2 glitch samples
        if channel.shape[1] >= 2:
            channel = channel[:, :-2]

        seq = np.unwrap(
            np.array(group["sequence_number"].to_list(), dtype=np.int64),
            period=SEQ_PERIOD,
        )
        if seq.shape[0] >= 2:
            seq = seq[:-2]

        result[rx_name] = (channel, seq)

    return result


# ===========================================================================
# 2. ToF estimation + stats
# ===========================================================================
def compute_tof_stats_for_receivers(
    channels: dict[
        str, tuple[np.ndarray, np.ndarray]
    ],  # {receiver : (CSI, sequence_nums)}
    pad_factor: int = PAD_FACTOR,
    carrier_spacing: float = CARRIER_SPACING,
) -> tuple[list[ToFStats], dict[str, TOFConfig]]:
    """
    Compute ToFStats and optimized TOFConfig for each receiver.
    """
    stats: list[ToFStats] = []
    configs: dict[str, TOFConfig] = {}

    def rx_sort(name: str) -> int:
        return RECEIVER_ORDER.index(name) if name in RECEIVER_ORDER else 999

    for rx_name in sorted(channels.keys(), key=rx_sort):
        channel, seq = channels[rx_name]

        logger.info(
            f"Receiver={rx_name}, subcarriers={channel.shape[0]}, samples={channel.shape[1]}"
        )

        cfg, best_dev = TOFProcessor.optimize_hyperparameters(
            csi=channel,
            sequence_numbers=seq,
            pad_factor=pad_factor,
            carrier_spacing=carrier_spacing,
        )
        logger.info(
            f"Optimized config for {rx_name}: {cfg} ({best_dev:.2f} ns deviation)"
        )

        configs[rx_name] = cfg
        processor = TOFProcessor(cfg)

        pdp = compute_pdp(channel, pad_factor=pad_factor)
        delays_ns = compute_delays(channel.shape[0], pad_factor, carrier_spacing) * 1e9

        tof_ns = processor.estimate_tof_difference(pdp, delays_ns)
        stat = processor.get_tof_stats(
            computed_tof_ns=tof_ns,
            sequence_numbers=seq,
            receiver_name=rx_name,
            carrier_spacing=carrier_spacing,
        )
        stats.append(stat)

    return stats, configs


def compute_stats_block(
    *,
    normalize: bool,
    phase_variant: str,
    pipeline_label: str,
    phase_label: str,
) -> pl.DataFrame:
    """
    Compute ToF stats and hyperparameters for a single (pipeline, phase) setting
    and return them as a DataFrame block in the unified schema.
    """
    df_csi = load_csi_data(normalize=normalize, phase_variant=phase_variant)
    channels = csi_per_receiver(df_csi)

    stats, configs = compute_tof_stats_for_receivers(
        channels,
        pad_factor=PAD_FACTOR,
        carrier_spacing=CARRIER_SPACING,
    )

    rows: list[dict[str, Any]] = []
    for s in stats:
        cfg = configs[s.receiver_name]
        rows.append(
            {
                "receiver_name": s.receiver_name,
                "pipeline": pipeline_label,
                "phase": phase_label,
                "seq_numbers": s.seq_numbers.tolist(),
                "computed_ns": s.computed_ns.tolist(),
                "ground_truth_ns": s.ground_truth_ns.tolist(),
                "avg_error_ns": float(s.avg_error_ns),
                "std_error_ns": float(s.std_error_ns),
                "min_peak_distance": int(cfg.min_peak_distance),
                "min_peak_strength": float(cfg.min_peak_strength),
                "pad_factor": int(cfg.pad_factor),
            }
        )

    return pl.DataFrame(rows)


def load_stats_from_all(
    pipeline: str,
    phase: str = "clean_phase",
) -> list[ToFStats]:
    """
    Load ToFStats objects for a given (pipeline, phase) from the unified stats file.
    """
    if not STATS_PATH.exists():
        raise FileNotFoundError("Run `calculate` first; no ToF stats found.")

    df = pl.read_parquet(STATS_PATH).filter(
        (pl.col("pipeline") == pipeline) & (pl.col("phase") == phase)
    )

    out: list[ToFStats] = []
    for row in df.iter_rows(named=True):
        out.append(
            ToFStats(
                receiver_name=row["receiver_name"],
                seq_numbers=np.asarray(row["seq_numbers"], dtype=np.int64),
                computed_ns=np.asarray(row["computed_ns"], dtype=float),
                ground_truth_ns=np.asarray(row["ground_truth_ns"], dtype=float),
                avg_error_ns=float(row["avg_error_ns"]),
                std_error_ns=float(row["std_error_ns"]),
            )
        )
    return out


def stats_to_summary_df(stats: list[ToFStats]) -> pl.DataFrame:
    """
    Reduce a list of ToFStats to a summary frame for normalization plots.
    """
    return pl.DataFrame(
        {
            "receiver_name": [s.receiver_name for s in stats],
            "avg_error_ns": [float(s.avg_error_ns) for s in stats],
            "std_error_ns": [float(s.std_error_ns) for s in stats],
        }
    )


# ===========================================================================
# 3. PDPConfig + black/gray 3D waterfall plots
# ===========================================================================
@dataclass
class PDPConfig:
    """
    PDP over time for a single receiver.
    """

    # fmt: off
    delays : np.ndarray  # candidate delays (ns)
    times  : np.ndarray  # evaluated times (s)
    pdp    : np.ndarray  # PDP values
    name   : str = ""    # receiver name
    # fmt: on


def build_pdp_configs(
    channels: dict[str, tuple[np.ndarray, np.ndarray]],
    pad_factor: int = PAD_FACTOR,
) -> list[PDPConfig]:
    """
    Build PDPConfig list from per-receiver channels.
    """
    configs: list[PDPConfig] = []

    for rx_name in RECEIVER_ORDER:
        if rx_name not in channels:
            continue

        channel, _ = channels[rx_name]
        pdp = compute_pdp(channel, pad_factor=pad_factor)
        delays_ns = compute_delays(channel.shape[0], pad_factor, CARRIER_SPACING) * 1e9

        mask = (delays_ns >= PDP_RANGE_NS[0]) & (delays_ns <= PDP_RANGE_NS[1])
        pdp_clipped = pdp[mask]

        times = np.linspace(0.0, 5.0, pdp.shape[1])

        configs.append(
            PDPConfig(
                delays=delays_ns[mask],
                times=times,
                pdp=pdp_clipped,
                name=rx_name,
            )
        )

    return configs


def plot_pdp_waterfall_on_ax_lines(
    ax,
    delays: np.ndarray,
    time_steps: np.ndarray,
    pdp: np.ndarray,
) -> None:
    """
    Render a PDP waterfall plot as a series of 3D curves.
    """
    total_points = len(time_steps)
    if total_points > NUM_WIREFRAME:
        indices = np.linspace(0, total_points - 1, NUM_WIREFRAME, dtype=int)
    else:
        indices = np.arange(total_points)

    pdp = np.clip(pdp, 0, PDP_CUTOFF)

    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.zaxis.pane.set_visible(False)
    ax.grid(False)

    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.zaxis.set_major_locator(MaxNLocator(2))

    tick_label_size = 16
    ax.tick_params(axis="x", labelsize=tick_label_size)
    ax.tick_params(axis="y", labelsize=tick_label_size)
    ax.zaxis.set_tick_params(labelsize=tick_label_size)

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        for tick in axis.get_ticklines():
            tick.set_color("lightgray")
            tick.set_linewidth(1.0)

    ax.set_box_aspect([1.5, 1.2, 1.0])

    for t in indices:
        ax.plot(
            delays,
            [time_steps[t]] * len(delays),
            pdp[:, t],
            color="darkgray",
            linewidth=1.5,
            alpha=0.8,
        )

    ax.set_xlabel("Delay (ns)", fontsize=15, labelpad=10)
    ax.set_ylabel("Time (s)", fontsize=15, labelpad=10)
    ax.set_zlabel("PDP", fontsize=15, labelpad=10)
    ax.set_zlim(0, PDP_CUTOFF)
    ax.view_init(elev=35, azim=-115)


def plot_all_pdp_waterfall_lines(
    configs: list[PDPConfig],
    output_file: Path | str | None = None,
) -> None:
    """
    Create a faceted grid of 3D PDP waterfall plots using manual axes positioning.
    """
    if output_file is None:
        output_file = data_dir / "all-pdp-waterfall.pdf"

    num_devices = len(configs)
    if num_devices == 0:
        return

    cols = 4
    rows = (num_devices + cols - 1) // cols

    left_margin = 0.03
    right_margin = 0.99
    bottom_margin = 0.015
    top_margin = 1.0

    avail_width = right_margin - left_margin
    avail_height = top_margin - bottom_margin
    subplot_width = avail_width / cols
    subplot_height = avail_height / rows

    fig = plt.figure(figsize=(20, 10))

    for idx, config in enumerate(configs):
        col = idx % cols
        row = rows - 1 - (idx // cols)
        gap_x = 0.0
        gap_y = 0.0

        ax_left = left_margin + col * subplot_width + gap_x
        ax_bottom = bottom_margin + row * subplot_height + gap_y
        ax_width = subplot_width - 2 * gap_x
        ax_height = subplot_height - 2 * gap_y

        ax = fig.add_axes([ax_left, ax_bottom, ax_width, ax_height], projection="3d")  # type: ignore[call-overload]
        plot_pdp_waterfall_on_ax_lines(ax, config.delays, config.times, config.pdp)

        ax.text(
            3.0,
            -9.0,
            3.3,
            config.name,
            transform=ax.transAxes,
            fontsize=24,
            ha="center",
            va="bottom",
        )

    plt.savefig(output_file, format="pdf", dpi=300, transparent=True)
    if SHOW:
        plt.show()
    plt.close(fig)


def plot_single_pdp_waterfall_lines(
    config: PDPConfig,
    output_file: Path | str | None = None,
) -> None:
    """
    Create a single 3D PDP waterfall lines plot (per receiver).
    """
    if output_file is None:
        output_file = data_dir / f"pdp-waterfall-line-{config.name}.pdf"

    left_margin = 0.08
    right_margin = 0.99
    bottom_margin = 0.08
    top_margin = 0.99

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_axes(  # type: ignore[call-overload]
        [
            left_margin,
            bottom_margin,
            right_margin - left_margin,
            top_margin - bottom_margin,
        ],
        projection="3d",
    )

    plot_pdp_waterfall_on_ax_lines(ax, config.delays, config.times, config.pdp)

    plt.savefig(output_file, format="pdf", dpi=300, transparent=True)

    if SHOW:
        plt.show()
    plt.close(fig)


# ===========================================================================
# 4. Ground-truth + normalization plots
# ===========================================================================
def plot_all_ground_truth_top(
    ground_truth_list: list[ToFStats],
    output_file: Path | str | None = None,
) -> None:
    """
    Multi-receiver ground truth vs computed ToF plot with avg deviation row on top.
    """
    if output_file is None:
        output_file = data_dir / "all-ground-truth.pdf"

    sns.set_theme(
        style="whitegrid",
        context="talk",
        font_scale=1,
        rc={
            "grid.color": "0.9",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
        },
    )

    rows = []
    for d in ground_truth_list:
        for kind, vals in [
            ("Computed ToF (ns)", d.computed_ns),
            ("Ground Truth ToF (ns)", d.ground_truth_ns),
        ]:
            rows.append(
                pd.DataFrame(
                    {
                        "sequence_number": d.seq_numbers,
                        "ToF": vals,
                        "Type": kind,
                        "rx": d.receiver_name,
                        "avg_dev": d.avg_error_ns,
                    }
                )
            )
    df = pd.concat(rows, ignore_index=True)
    df["Type"] = pd.Categorical(
        df["Type"], ["Computed ToF (ns)", "Ground Truth ToF (ns)"], ordered=True
    )
    df["rx"] = pd.Categorical(df["rx"], RECEIVER_ORDER, ordered=True)
    df.sort_values("rx", inplace=True)

    facet_w, facet_h = 4.5, 2.0
    bar_h = 2.5
    extra_h = 0.5

    fig_w = 2 * facet_w
    fig_h = bar_h + 4 * facet_h + extra_h

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[bar_h, 4 * facet_h],
        hspace=0.1,
        left=0.12,
        right=0.98,
        top=0.95,
        bottom=0.06,
    )

    ax_bar = fig.add_subplot(outer[0])
    dev_df = (
        df[["rx", "avg_dev"]]
        .drop_duplicates("rx")
        .set_index("rx")
        .loc[RECEIVER_ORDER]
        .reset_index()
    )
    sns.barplot(x="rx", y="avg_dev", data=dev_df, color="#636363", ax=ax_bar)
    ax_bar.set_ylabel("Average Deviation (ns)", color="#636363", labelpad=6)
    ax_bar.set_xlabel("")
    ax_bar.set_ylim(0, 1.8)

    ax_bar.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_bar.tick_params(axis="y", labelcolor="#636363", colors="#636363")

    bar_label_fontsize = 15
    ylim_top = ax_bar.get_ylim()[1]
    offset = 0.03 * ylim_top

    for patch, rx_label in zip(ax_bar.patches, dev_df["rx"]):
        height = patch.get_height()  # type: ignore[attr-defined]
        x = patch.get_x() + patch.get_width() / 2.0  # type: ignore[attr-defined]
        y = height + offset
        ax_bar.text(
            x,
            y,
            str(rx_label),
            ha="center",
            va="bottom",
            rotation=90,
            color="#3b3b3b",
            fontsize=bar_label_fontsize,
            fontweight="bold",
            clip_on=False,
        )

    palette = {"Computed ToF (ns)": "#FFA500", "Ground Truth ToF (ns)": "#000000"}
    dashes = {"Computed ToF (ns)": (1, 0), "Ground Truth ToF (ns)": (5, 2)}

    inner = outer[1].subgridspec(4, 2, hspace=0.25, wspace=0.08)
    axes: list[plt.Axes] = []
    for idx, rx in enumerate(RECEIVER_ORDER):
        r, c = divmod(idx, 2)
        ax = fig.add_subplot(
            inner[r, c],
            sharex=axes[0] if axes else None,
            sharey=axes[0] if axes else None,
        )
        axes.append(ax)

        sub = df[df["rx"] == rx]
        if sub.empty:
            ax.set_visible(False)
            continue

        sns.lineplot(
            data=sub,
            x="sequence_number",
            y="ToF",
            hue="Type",
            style="Type",
            palette=palette,
            dashes=dashes,
            linewidth=2,
            ax=ax,
            legend=False,
        )

        avg = sub["avg_dev"].iloc[0]
        ax.set_title(rf"$\bf{{{rx}:\,\Delta = {avg:.2f}\,ns}}$", color="#3b3b3b", pad=4)

        if r < 3:
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.set_xlabel("Sequence Number", color="#636363")

        if c == 1:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)
        else:
            ax.set_ylabel("ToF (ns)", color="#636363")

        ax.tick_params(
            axis="both", which="both", labelcolor="#636363", colors="#636363"
        )

    handles = [
        Line2D([0], [0], color=palette["Computed ToF (ns)"], lw=3),
        Line2D(
            [0],
            [0],
            color=palette["Ground Truth ToF (ns)"],
            lw=3,
            dashes=dashes["Ground Truth ToF (ns)"],
        ),
    ]
    labels = ["Computed ToF (ns)", "Ground Truth ToF (ns)"]

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        frameon=False,
        fontsize=20,
    )

    fig.savefig(output_file, bbox_inches="tight")
    plt.close(fig)


def plot_normalization_comparison(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    raw_df: pl.DataFrame,
    scaled_df: pl.DataFrame,
    output_file: Path | str | None = None,
    dodge: float = 0.2,
    # series appearance
    names: tuple[str, str] = ("Not equalized", "Equalized"),
    colors: tuple[str, str] = (LIGHT_ORANGE, LIGHT_TEAL),
    symbols: tuple[str, str] = ("circle", "diamond"),
    # layout & typography
    height: int = 650,
    width: int = 1600,
    receiver_bands: bool = True,
    inner_labels: bool = True,
    label_y: float = 0.98,
    label_clearance: float = 0.02,
    min_top_pad_frac: float = 0.12,
    bottom_pad_frac: float = 0.04,
    y_title: str = "ToF Deviation (ns)",
    y_title_standoff: int = 32,
    y_tick_font_size: int = 38,
    y_nticks: int = 6,
    y_range: tuple[float, float] | None = None,
    legend_loc: str = "top-right",
    legend_font_size: int = 44,
):
    """
    Plot normalization comparison (raw vs equalized) with error bars.
    """
    if output_file is None:
        output_file = data_dir / "normalization-comp.pdf"

    group_spacing = 1.5

    receivers = scaled_df.reverse()["receiver_name"].to_list()

    def to_dict(df: pl.DataFrame) -> dict[str, tuple[float, float]]:
        out: dict[str, tuple[float, float]] = {}
        for r in df.iter_rows(named=True):
            out[str(r["receiver_name"])] = (
                float(r["avg_error_ns"]),
                float(r["std_error_ns"]),
            )
        return out

    data_raw = to_dict(raw_df)
    data_eq = to_dict(scaled_df)

    base_x = {rcv: i * group_spacing for i, rcv in enumerate(receivers)}
    offsets = (-dodge, +dodge)
    fig = go.Figure()
    marker_cfg = dict(size=35, line=dict(width=2, color="#A9A9A9"))
    error_cfg = dict(type="data", thickness=5, width=0, color="#636363")

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

    series = [
        (data_raw, names[0], symbols[0], colors[0]),
        (data_eq, names[1], symbols[1], colors[1]),
    ]

    ymins, ymaxs = [], []
    for k, (dmap, label, symbol, color) in enumerate(series):
        xs, ys, err = [], [], []
        for rcv in receivers:
            mean, std = dmap.get(rcv, (np.nan, 0.0))
            xs.append(base_x[rcv] + offsets[k])
            ys.append(mean)
            err.append(std)
            if np.isfinite(mean):
                ymins.append(mean - std)
                ymaxs.append(mean + std)

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                name=label,
                marker={**marker_cfg, "symbol": symbol, "color": color},
                error_y={**error_cfg, "array": err, "arrayminus": err},
            )
        )

    if not ymins or not ymaxs:
        raise ValueError("No finite data to plot.")

    data_min = float(np.nanmin(ymins))
    data_max = float(np.nanmax(ymaxs))
    data_rng = max(data_max - data_min, 1e-12)
    bot_pad = data_rng * float(bottom_pad_frac)

    if y_title is None:
        y_title = ""

    if y_tick_font_size is None:
        y_tick_font_size = 10

    if y_nticks is None:
        y_nticks = 6

    if legend_loc is None:
        legend_loc = "top-right"

    if y_range is None:
        L = max(0.01, min(0.99, float(label_y) - float(label_clearance)))
        needed_top_pad = ((1.0 - L) / L) * (data_rng + bot_pad)
        top_pad = max(data_rng * float(min_top_pad_frac), needed_top_pad)
        yrange = [data_min - bot_pad, data_max + top_pad]
    else:
        yrange = list(y_range)

    if legend_loc == "top-right":
        leg_x, leg_y, leg_xa, leg_ya = 0.98, 0.88, "right", "top"
    else:
        leg_x, leg_y, leg_xa, leg_ya = 0.98, 0.02, "right", "bottom"

    tickvals = [base_x[r] for r in receivers]

    fig.update_layout(
        width=width,
        height=height,
        template="plotly_white",
        margin=dict(l=120, r=40, t=0, b=0, pad=0),
        legend=dict(
            orientation="h",
            x=leg_x,
            xanchor=leg_xa,
            y=leg_y,
            yanchor=leg_ya,
            bgcolor="#ffffff",
            bordercolor="rgba(0,0,0,0.12)",
            borderwidth=1,
            font=dict(size=legend_font_size, family="Arial", color="gray"),
        ),
    )

    fig.update_xaxes(
        title=None,
        tickmode="array",
        tickvals=tickvals,
        ticktext=[] if inner_labels else receivers,
        showticklabels=not inner_labels,
        ticks="",
    )

    fig.update_yaxes(
        title=y_title,
        range=yrange,
        title_font=dict(size=48, family="Arial", color="gray"),
        tickfont=dict(size=y_tick_font_size, family="Arial", color="dimgray"),
        nticks=int(y_nticks),
        title_standoff=y_title_standoff,
        automargin=True,
        zeroline=True,
        zerolinecolor="#9e9e9e",
        zerolinewidth=2,
    )

    fig.update_traces(line_width=5, marker_line_width=2, error_y_thickness=5)

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

    fig.write_image(output_file, width=width, height=height)


def compute_phase_delta_matrix(
    stats: pl.DataFrame,
) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Build a 2xN matrix of delta_abs_error [ns]:

        delta_abs_error = avg_error_ns_clean_phase - avg_error_ns_clean_phase_pads

    rows    = PHASE_PIPELINES (phase cleaned, +Equalized)
    columns = receivers (RECEIVER_ORDER)
    """
    receivers = RECEIVER_ORDER

    mat = np.full((len(PHASE_PIPELINES), len(receivers)), np.nan, dtype=float)

    for i, pipeline_label in enumerate(PHASE_PIPELINES):
        for j, r in enumerate(receivers):
            pads_df = stats.filter(
                (pl.col("receiver_name") == r)
                & (pl.col("pipeline") == pipeline_label)
                & (pl.col("phase") == "clean_phase_pads")
            ).select("avg_error_ns")

            cf_df = stats.filter(
                (pl.col("receiver_name") == r)
                & (pl.col("pipeline") == pipeline_label)
                & (pl.col("phase") == "clean_phase")
            ).select("avg_error_ns")

            pads_val = float(pads_df["avg_error_ns"][0])
            cf_val = float(cf_df["avg_error_ns"][0])
            mat[i, j] = cf_val - pads_val

    return mat, receivers, PHASE_PIPELINES


def plot_phase_delta_matrix(
    delta_mat: np.ndarray,
    receivers: list[str],
    pipeline_labels: list[str],
    output_file: Path | str | None = None,
) -> None:
    """
    Plot 2xN heatmap of delta_abs_error per pipeline and receiver.
    """
    if output_file is None:
        output_file = data_dir / "tof-phase-delta-matrix.pdf"

    df = pd.DataFrame(
        delta_mat,
        index=pipeline_labels,
        columns=receivers,
    )

    if np.isfinite(delta_mat).any():
        max_abs = float(np.nanmax(np.abs(delta_mat)))
        if max_abs == 0:
            max_abs = 1e-6
    else:
        max_abs = 1e-6

    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    sns.set_theme(style="white", context="paper", font_scale=2)

    plt.figure(figsize=(10, 2.5))
    ax = sns.heatmap(
        df,
        annot=True,
        fmt=".3f",
        cmap=tgo_cmap_rev,
        norm=norm,
        square=False,
        cbar_kws={
            "shrink": 0.8,
            "label": r"$\Delta |e|$ [ns]",
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

    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="left",
        fontsize=16,
        color="dimgray",
    )
    plt.setp(
        ax.get_yticklabels(),
        rotation=0,
        fontsize=16,
        color="dimgray",
    )

    ax.set_xlabel("", fontsize=18, color="gray", labelpad=10)
    ax.set_ylabel("", fontsize=18, color="gray", labelpad=10)
    ax.tick_params(axis="both", which="both", length=0)

    plt.tight_layout()
    plt.savefig(output_file, format="pdf", bbox_inches="tight", pad_inches=0.1)
    plt.close()


# ===========================================================================
# 5. CLI orchestration
# ===========================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="ToF estimation pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("calculate", help="Compute ToF stats and save to parquet.")
    subparsers.add_parser(
        "plot_groundtruth", help="Plot ground truth vs ToF estimates."
    )
    subparsers.add_parser("plot_normalization", help="Plot normalization comparison.")
    subparsers.add_parser("plot_pdps", help="Plot PDP waterfalls.")
    subparsers.add_parser(
        "plot_phase_delta",
        help="Plot delta-error matrix for different phase preprocessing.",
    )

    args = parser.parse_args()

    # ------------------------------------------------------
    # calculate
    # ------------------------------------------------------
    if args.command == "calculate":
        if not CSI_PATH.exists() or not META_PATH.exists():
            raise FileNotFoundError("Missing CSI or meta parquet in data directory.")

        logger.info("Computing ToF stats for all pipelines and phase variants...")

        phase_map: dict[str, str] = {
            "clean_phase": "ls",
            "clean_phase_pads": "pads",
        }

        blocks: list[pl.DataFrame] = []

        for pipeline_label, normalize in [
            ("phase cleaned", False),
            ("+Equalized", True),
        ]:
            for phase_label, phase_variant in phase_map.items():
                logger.info(
                    f"pipeline={pipeline_label}, phase={phase_label}, normalize={normalize}"
                )
                block = compute_stats_block(
                    normalize=normalize,
                    phase_variant=phase_variant,
                    pipeline_label=pipeline_label,
                    phase_label=phase_label,
                )
                blocks.append(block)

        all_stats = pl.concat(blocks, how="vertical")
        all_stats.write_parquet(STATS_PATH)

        logger.info(f"Done. Stats written to: {STATS_PATH}")
        return

    # ------------------------------------------------------
    # plot_groundtruth
    # ------------------------------------------------------
    if args.command == "plot_groundtruth":
        stats_eq = load_stats_from_all(pipeline="+Equalized", phase="clean_phase")
        logger.info("Plotting ground truth vs ToF estimates (equalized pipeline)...")
        plot_all_ground_truth_top(stats_eq)
        logger.info("Ground-truth figure generated.")
        return

    # ------------------------------------------------------
    # plot_normalization
    # ------------------------------------------------------
    if args.command == "plot_normalization":
        stats_raw = load_stats_from_all(pipeline="phase cleaned", phase="clean_phase")
        stats_eq = load_stats_from_all(pipeline="+Equalized", phase="clean_phase")

        raw_df = stats_to_summary_df(stats_raw)
        eq_df = stats_to_summary_df(stats_eq)

        logger.info("Plotting normalization comparison (raw vs equalized)...")
        plot_normalization_comparison(raw_df, eq_df)
        logger.info("Normalization comparison plot generated.")
        return

    # ------------------------------------------------------
    # plot_pdps
    # ------------------------------------------------------
    if args.command == "plot_pdps":
        logger.info("Loading CSI (normalize=True) for PDP plots...")
        df_eq = load_csi_data(normalize=True)
        ch_eq = csi_per_receiver(df_eq)

        logger.info("Computing PDP configs...")
        configs = build_pdp_configs(ch_eq, pad_factor=PAD_FACTOR)

        logger.info("Plotting all PDP waterfalls (lines, black/gray)...")
        plot_all_pdp_waterfall_lines(configs)

        logger.info("Plotting per-receiver PDP line waterfalls...")
        for cfg in configs:
            plot_single_pdp_waterfall_lines(cfg)

        logger.info("PDP plots generated.")
        return

    # ------------------------------------------------------
    # plot_phase_delta
    # ------------------------------------------------------
    if args.command == "plot_phase_delta":
        logger.info("Loading ToF stats for phase variants...")
        if not STATS_PATH.exists():
            raise FileNotFoundError("Run `calculate` first; no ToF stats found.")
        stats_phase = pl.read_parquet(STATS_PATH)

        delta_mat, recs, pls = compute_phase_delta_matrix(stats_phase)

        logger.info(f"Receivers (columns): {recs}")
        logger.info(f"Pipelines (rows): {pls}")

        logger.info("Plotting phase delta-error matrix...")
        plot_phase_delta_matrix(delta_mat, recs, pls)
        logger.info("Phase delta-error matrix plot generated.")
        return


if __name__ == "__main__":
    main()
