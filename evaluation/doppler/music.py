#!/usr/bin/env python3
from typing import Mapping, Sequence
from pathlib import Path

import numpy as np
import polars as pl
import seaborn as sns
import plotly.io as pio
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from evaluation.common import RECEIVER_ORDER, fmlp_cmap_2
from matplotlib.colors import Normalize
from matplotlib.transforms import Bbox
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from sensession.campaign.processor import CampaignProcessor

pio.kaleido.scope.mathjax = None

data_dir = Path.cwd() / "data" / "doppler_emulation_slow"
SHOW: bool = False


def regroup(df: pl.DataFrame):
    """
    Regroup CSI
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
    scale: bool = True, equalize: bool = False, rescale_rssi: bool = False
) -> pl.DataFrame:
    """Load CSI data without filtering receivers."""
    data = pl.read_parquet(data_dir / "csi.parquet")
    meta = pl.read_parquet(data_dir / "meta.parquet")

    proc = (
        CampaignProcessor(
            data,
            meta,
            lazy=False,
        )
        .correct_rssi_by_agc()
        .unwrap()
        .filter("antenna_idxs", 0)
        .detrend_phase(pin_edges=False)
    )

    if scale:
        proc = proc.scale_magnitude()

    if rescale_rssi:
        proc = proc.rescale_csi_by_rssi(
            exclude_expr=(pl.col("receiver_name") == "x310")
        )

    if equalize:
        proc = proc.equalize_magnitude().equalize_phase()

    proc = proc.drop_contains("collection_name", "warmup")

    if not isinstance(proc.csi, pl.DataFrame):
        raise ValueError("CSI must be instantiated DataFrame.")
    data = regroup(proc.csi)
    return data


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

    For each candidate speed v, the spectrum is defined as:
      Spectrum(v) = 1 / ( s(v)^H · P · s(v) )

    where:
      s(v) = exp(2j·π·gap_time·v·(n/wavelength)) for n = 0,...,num_packets-1,
      P    = U · U^H                             is the noise projection matrix derived from ...
      U    = csi_slice · (csi_slice)^H.          ... the covariance matrix
    The output is converted to decibels and normalized to [0, 1].
    """

    # Compute covariance matrix and derive the noise projection matrix.
    covariance = csi_slice @ csi_slice.conj().T
    _, eigenvectors = np.linalg.eigh(covariance)
    noise_subspace = eigenvectors[:, : eigenvectors.shape[0] - num_targets]
    noise_proj = noise_subspace @ noise_subspace.conj().T

    # Compute the steering matrix for all candidate speeds. Every row is of the form:
    #   s(v) = exp(2j·π·gap_time·v·(n/wavelength)), n = 0, ..., num_packets-1.
    comb = np.arange(num_packets) / wavelength
    steering_matrix = np.exp(2j * np.pi * gap_time * candidate_speeds[:, None] * comb)

    # Compute the quadratic form for each speed v:
    #   s(v)^H · (P_noise · s(v))
    temp = noise_proj @ steering_matrix.T
    spectrum = 1 / np.sum(steering_matrix.conj() * temp.T, axis=1)

    # Convert the spectrum to decibels and normalize to the range [0, 1].
    spec_db = 10 * np.log10(np.abs(spectrum))
    spec_db -= spec_db.min()
    max_val = spec_db.max()
    if max_val:
        spec_db /= max_val

    return spec_db


def estimate_speed(spectrum: np.ndarray, candidate_speeds: np.ndarray) -> float:
    """Return the candidate speed corresponding to the peak of the MUSIC spectrum."""
    return candidate_speeds[np.argmax(spectrum)]


def compute_and_print_stats(df: pl.DataFrame, groundtruth: float = 1.0) -> pl.DataFrame:
    """Compute speed stats, print results, and return summary for plotting."""

    stats_df = df.group_by("Receiver").agg(
        [
            pl.col("Estimated Speed").mean().alias("Mean Speed"),
            pl.col("Estimated Speed").median().alias("Median Speed"),
        ]
    )
    print("\nErrors (Mean and Median):")
    print(
        stats_df.with_columns(
            mean_err=(groundtruth - pl.col("Mean Speed")),
            median_err=(groundtruth - pl.col("Median Speed")),
        )
    )

    print("\nSquared Errors:")
    print(
        df.group_by("Receiver").agg(
            [
                ((groundtruth - pl.col("Estimated Speed")) ** 2)
                .median()
                .alias("Median Squared Error"),
                ((groundtruth - pl.col("Estimated Speed")) ** 2)
                .mean()
                .alias("Mean Squared Error"),
            ]
        )
    )

    return stats_df


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


def doppler_estimates(data: pl.DataFrame, figure_name: str) -> dict[str, list[float]]:
    num_packets = 50
    f_center = 2.462e9
    light_speed = 299_792_458
    wavelength = light_speed / f_center
    candidate_speeds = np.arange(0, 5.01, 0.0001)

    results = {}
    time_indices = {}

    for receiver_name, group in data.group_by("receiver_name", maintain_order=True):
        name = receiver_name[0]  # Extract the receiver name

        # Convert timestamps from datetime to float (seconds)
        # This is the ideal timestamp, taken from knowledge that the TX transmits
        # very regularly.
        sequence_nums = np.array(group.get_column("sequence_number").to_list())
        sequence_nums = np.unwrap(sequence_nums, period=4096)
        timestamps = sequence_nums * 1 / 500

        # Or use reported timestamps
        timestamps = np.array(
            [ts.timestamp() for ts in group.get_column("timestamp").to_list()]
        )

        csi_abs = np.array(group.get_column("csi_abs").to_list())
        csi_phs = np.array(group.get_column("csi_phase").to_list())
        csi_arr = csi_abs * np.exp(1j * csi_phs)
        csi_arr -= np.mean(csi_arr, axis=0, keepdims=True)

        total_packets = csi_arr.shape[0]
        num_blocks = total_packets // num_packets
        estimated_speed_list = []
        block_time_stamps = []

        for block_idx in range(num_blocks):
            start = block_idx * num_packets
            end = (block_idx + 1) * num_packets
            csi_slice = csi_arr[start:end]

            gap_time = (timestamps[end - 1] - timestamps[start]) / (num_packets - 1)

            spectrum = compute_music_spectrum(
                csi_slice,
                gap_time,
                wavelength,
                candidate_speeds,
                num_packets,
                num_targets=1,
            )
            est_speed = estimate_speed(spectrum, candidate_speeds)
            estimated_speed_list.append(est_speed)

            # Store elapsed time in seconds
            block_time_stamps.append(timestamps[end - 1] - timestamps[0])

        time_indices[name] = np.array(block_time_stamps, dtype=np.float64)
        results[name] = estimated_speed_list

    # plot_results_together(results, time_indices)
    plot_estimate_distributions(results, figure_name)
    return results


def plot_speed_boxplot(
    unscaled: Mapping[str, Sequence[float]],
    scaled: Mapping[str, Sequence[float]],
    *,
    save_path: Path | None = None,
    dodge: float = 0.18,
):
    """
    Horizontal, dodged box-plots (Raw vs AGC-scaled) per receiver,
    with whiskers capped at the 1.5xIQR fences and *no* outlier points.
    """

    # ---------- helper: discard points outside 1.5xIQR ------------------------
    def _trim(samples: Sequence[float]) -> list[float]:
        if len(samples) == 0:
            return []
        q1, q3 = np.percentile(samples, (25, 75))
        iqr = q3 - q1
        low, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        return [x for x in samples if low <= x <= hi]

    def _to_df(data: Mapping[str, Sequence[float]], label: str) -> pl.DataFrame:
        rows = []
        for rcv, s in data.items():
            rows.append({"receiver": rcv, "samples": _trim(s)})
        return (
            pl.DataFrame(rows)
            .explode("samples")
            .with_columns(pl.lit(label).alias("kind"))
        )

    df_u = _to_df(unscaled, "Raw")
    df_s = _to_df(scaled, "AGC-scaled")

    # stable paper-friendly order
    receivers = sorted(set(df_u["receiver"]))
    order_df = pl.DataFrame(
        {"receiver": receivers, "_row": list(range(len(receivers)))}
    )

    group_spacing = 1.5
    df_u = df_u.join(order_df, on="receiver").with_columns(
        ((pl.col("_row") * group_spacing) - dodge).alias("y")
    )
    df_s = df_s.join(order_df, on="receiver").with_columns(
        ((pl.col("_row") * group_spacing) + dodge).alias("y")
    )

    # ---------- build figure --------------------------------------------------
    fig = go.Figure()
    base_marker = dict(line=dict(width=2, color="#A9A9A9"))

    for df, colour in [(df_u, "#66c2a5"), (df_s, "#fc8d62")]:
        fig.add_trace(
            go.Box(
                x=df["samples"].to_list(),
                y=df["y"].to_list(),
                orientation="h",
                name=df["kind"][0],
                boxpoints=False,  # <- no outlier dots
                quartilemethod="exclusive",
                marker={**base_marker, "color": colour},
                line_color=colour,
                hovertemplate="%{x:.3f} m/s<extra></extra>",
                width=0.3,
            )
        )

    fig.update_traces(line_width=5, whiskerwidth=1)

    # ---------- layout (from your reference) ----------------------------------
    fig.update_layout(
        width=1920,
        height=1600,
        template="plotly_white",
        xaxis=dict(
            title="Estimated speed [m/s]",
            tickfont=dict(size=52, family="Arial", color="dimgray"),
            title_font=dict(size=56, family="Arial", color="gray"),
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=[i * group_spacing for i in range(len(receivers))],
            ticktext=receivers,
            tickfont=dict(size=52, family="Arial", color="dimgray"),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.98,
            xanchor="right",
            x=0.4,
            font=dict(size=52, family="Arial", color="gray"),
        ),
        margin=dict(l=180, r=80, t=70, b=80),
    )

    if save_path:
        fig.write_image(save_path, width=1920, height=1600)
    else:
        fig.show()


def plot_speed_pointrange_compact(
    methods: dict[str, dict[str, list[float]]],
    *,
    save_path: Path | None = None,
    dodge: float = 0.27,  # horizontal offset between neighbouring methods (within a receiver column)
    inner_labels: bool = True,  # put receiver names inside the figure (top)
    receiver_bands: bool = True,  # light background strips per receiver
    method_lanes: bool = False,  # faint vertical guides at each method x-position
    height: int = 800,
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
):
    """Compact point–range plot (median ± IQR) with values on Y and receivers on X.
    Inside-top receiver labels; legend inside bottom-right; y-range padded to avoid overlap/clipping.
    """

    # ----------------------------- helpers
    def _summary(samples: Sequence[float]) -> tuple[float, float, float]:
        """Return median, iqr_minus, iqr_plus."""
        if not samples:
            return np.nan, 0.0, 0.0
        q1, q3 = np.percentile(samples, (25, 75))
        med = float(np.median(samples))
        return med, med - q1, q3 - med

    # ----------------------------- tidy up
    method_labels = list(methods.keys())
    n_methods = len(method_labels)
    if n_methods < 2:
        raise ValueError("Need at least two methods for a comparison plot.")

    receivers = sorted({rcv for data in methods.values() for rcv in data.keys()})
    if not receivers:
        raise ValueError("No receivers found.")

    group_spacing = 1.5
    base_x = {rcv: i * group_spacing for i, rcv in enumerate(receivers)}
    offsets = np.linspace(-(n_methods - 1), n_methods - 1, n_methods) * (dodge / 2)

    # ----------------------------- figure + styles
    fig = go.Figure()
    marker_cfg = dict(size=35, line=dict(width=2, color="#A9A9A9"))
    error_cfg = dict(type="data", thickness=5, width=0, color="#636363")

    palette = [
        "#66c2a5",
        "#fc8d62",
        "#8da0cb",
        "#e78ac3",
        "#a6d854",
        "#ffd92f",
        "#e5c494",
        "#b3b3b3",
    ]
    symbols = [
        "circle",
        "diamond",
        "square",
        "triangle-up",
        "x",
        "cross",
        "star",
        "triangle-down",
    ]

    # Alternating receiver bands
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

    # Optional vertical method guide lines
    if method_lanes:
        for k in range(n_methods):
            for rcv in receivers:
                fig.add_vline(
                    x=base_x[rcv] + offsets[k],
                    line_width=1,
                    line_dash="dot",
                    line_color="rgba(0,0,0,0.15)",
                    layer="below",
                )

    # ----------------------------- traces (values on Y)
    ymins, ymaxs = [], []
    for k, label in enumerate(method_labels):
        data = methods[label]
        xs, meds, err_minus, err_plus = [], [], [], []
        for rcv in receivers:
            med, m_minus, m_plus = _summary(data.get(rcv, []))
            xs.append(base_x[rcv] + offsets[k])
            meds.append(med)
            err_minus.append(m_minus)
            err_plus.append(m_plus)
            if np.isfinite(med):
                ymins.append(med - m_minus)
                ymaxs.append(med + m_plus)

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=meds,
                mode="markers",
                name=label,
                marker={
                    **marker_cfg,
                    "symbol": symbols[k % len(symbols)],
                    "color": palette[k % len(palette)],
                },
                error_y={
                    **error_cfg,
                    "array": err_plus,
                    "arrayminus": err_minus,
                },  # up & down
            )
        )

    if not ymins or not ymaxs:
        raise ValueError("No finite data to plot.")

    # ----------------------------- y-range with headroom (top) and safety (bottom)
    data_min = float(np.nanmin(ymins))
    data_max = float(np.nanmax(ymaxs))
    data_rng = max(data_max - data_min, 1e-12)

    # bottom pad (prevents visual clipping at the floor)
    bot_pad = data_rng * float(bottom_pad_frac)

    # ensure top-of-data stays below the label line by label_clearance (paper coords)
    L = max(0.01, min(0.99, float(label_y) - float(label_clearance)))
    needed_top_pad = ((1.0 - L) / L) * (
        data_rng + bot_pad
    )  # derived from paper-coordinate constraint
    top_pad = max(data_rng * float(min_top_pad_frac), needed_top_pad)

    fig.update_yaxes(range=[data_min - bot_pad, data_max + top_pad])

    # ----------------------------- layout & cosmetics
    fig.update_layout(
        width=width,
        height=height,
        template="plotly_white",
        xaxis=dict(
            title=None,
            tickmode="array",
            tickvals=[base_x[r] for r in receivers],
            ticktext=["" for _ in receivers] if inner_labels else receivers,
            showticklabels=not inner_labels,  # hide to remove any bottom gap
            ticks="",
            tickfont=dict(
                size=receiver_label_font_size, family="Arial", color="dimgray"
            ),
        ),
        yaxis=dict(
            title="Estimated speed [m/s]",
            tickfont=dict(size=y_tick_font_size, family="Arial", color="dimgray"),
            title_font=dict(size=y_title_font_size, family="Arial", color="gray"),
            nticks=int(y_nticks),
        ),
        legend=dict(  # INSIDE bottom-right
            orientation="h",
            x=0.98,
            xanchor="right",
            y=0.02,
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.6)",
            bordercolor="rgba(0,0,0,0.05)",
            borderwidth=1,
            font=dict(size=legend_font_size, family="Arial", color="gray"),
        ),
        margin=dict(l=120, r=40, t=0, b=0, pad=0),  # no top or bottom margin
    )

    # make things pop
    fig.update_traces(line_width=5, marker_line_width=2, error_y_thickness=5)
    fig.update_yaxes(title_standoff=y_title_standoff)

    # ground-truth line
    fig.add_hline(
        y=1, line_dash="dash", line_color="#b4b4b4", line_width=3, layer="below"
    )

    # ----------------------------- receiver labels inside (top)
    if inner_labels:
        for rcv in receivers:
            fig.add_annotation(
                x=base_x[rcv],
                xref="x",
                y=float(label_y),
                yref="paper",
                text=rcv,
                showarrow=False,
                xanchor="center",
                yanchor="top",
                font=dict(
                    size=receiver_label_font_size, family="Arial", color="dimgray"
                ),
                bgcolor="rgba(255,255,255,0.6)",
                bordercolor="rgba(0,0,0,0.05)",
                borderwidth=1,
            )

    # ----------------------------- output
    if save_path:
        fig.write_image(save_path, width=width, height=height)
    else:
        fig.show()


def plot_speed_pointrange(
    methods: dict[str, dict[str, list[float]]],
    *,
    save_path: Path | None = None,
    dodge: float = 0.27,  # half the total spread across 4 methods
):
    """
    Point-range plot (median ± IQR) for multiple processing methods.

    Parameters
    ----------
    methods
        Dict keyed by *method-label* → { receiver → list of speed estimates }.
        Example: { "Raw":   results_unscaled,
                   "AGC":   results_scaled,
                   "DPF":   results_dpf,
                   "XYZ":   results_xyz }
    save_path
        If given, write PDF/PNG via Kaleido; otherwise call `fig.show()`.
    dodge
        Horizontal offset (in y-axis units) between neighbouring methods.
        Defaults to ±0.27 so the four markers fill a range ≈ 1.6 units wide.
    """

    # ------------------------------------------------------------------ helpers
    def _summary(samples: Sequence[float]) -> tuple[float, float, float]:
        """Return median, iqr_minus, iqr_plus."""
        if not samples:
            return np.nan, 0, 0
        q1, q3 = np.percentile(samples, (25, 75))
        med = float(np.median(samples))
        return med, med - q1, q3 - med

    # ------------------------------------------------------------------ tidy up
    method_labels = list(methods.keys())
    n_methods = len(method_labels)
    if n_methods < 2:
        raise ValueError("Need at least two methods for a comparison plot.")

    # Collect receiver set once; order alphabetically for paper-stable layout
    receivers = sorted({rcv for data in methods.values() for rcv in data.keys()})

    # Numerical y position for each receiver (centre line)
    group_spacing = 1.5
    base_y = {rcv: i * group_spacing for i, rcv in enumerate(receivers)}

    # Pre-compute per-method offset so they’re symmetrically dodged
    # e.g. for 4 methods → offsets = [-3d, -d, +d, +3d] with d = dodge
    offsets = np.linspace(-(n_methods - 1), n_methods - 1, n_methods) * dodge / 2

    # ------------------------------------------------------------------ figure
    fig = go.Figure()
    marker_cfg = dict(size=35, line=dict(width=2, color="#A9A9A9"))
    error_cfg = dict(type="data", thickness=5, width=0, color="#636363")

    # Palette + symbols
    palette = [
        "#66c2a5",
        "#fc8d62",
        "#8da0cb",
        "#e78ac3",
        "#a6d854",
        "#ffd92f",
        "#e5c494",
        "#b3b3b3",
    ]
    symbols = [
        "circle",
        "diamond",
        "square",
        "triangle-up",
        "x",
        "cross",
        "star",
        "triangle-down",
    ]

    for k, label in enumerate(method_labels):
        data = methods[label]
        rows = []
        for rcv in receivers:
            med, m_minus, m_plus = _summary(data.get(rcv, []))
            rows.append((med, m_minus, m_plus, base_y[rcv] + offsets[k]))
        meds, err_minus, err_plus, ys = zip(*rows)

        fig.add_trace(
            go.Scatter(
                x=list(meds),
                y=list(ys),
                mode="markers",
                name=label,
                marker={
                    **marker_cfg,
                    "symbol": symbols[k % len(symbols)],
                    "color": palette[k % len(palette)],
                },
                error_x={
                    **error_cfg,
                    "array": list(err_plus),
                    "arrayminus": list(err_minus),
                },
            )
        )

    # ------------------------------------------------------------------ layout
    fig.update_layout(
        width=1920,
        height=1600,
        template="plotly_white",
        xaxis=dict(
            title="Estimated speed [m/s]",
            tickfont=dict(size=48, family="Arial", color="dimgray"),
            title_font=dict(size=50, family="Arial", color="gray"),
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=[base_y[r] for r in receivers],
            ticktext=receivers,
            tickfont=dict(size=48, family="Arial", color="dimgray"),
            title_font=dict(size=50, family="Arial", color="gray"),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.98,
            xanchor="right",
            x=0.4,
            font=dict(size=48, family="Arial", color="gray"),
        ),
        margin=dict(l=180, r=80, t=70, b=80),
    )

    # Thicker strokes so points pop even after journal down-sampling
    fig.update_traces(line_width=5, marker_line_width=2, error_x_thickness=5)

    fig.add_vline(
        x=1,  # ground-truth speed
        line_dash="dash",
        line_color="#b4b4b4",  # soft grey so it doesn’t dominate
        line_width=3,
        layer="below",  # keeps markers/error bars on top
    )

    # ------------------------------------------------------------------ output
    if save_path:
        fig.write_image(save_path, width=1920, height=1600)
    else:
        fig.show()


def main():
    data_scaled = load_csi_data(scale=True)
    data_unscaled = load_csi_data(scale=False)
    data_rssiscaled = load_csi_data(scale=True, rescale_rssi=True)

    results_scaled: dict[str, list[float]] = doppler_estimates(data_scaled, "scaled")
    results_unscaled: dict[str, list[float]] = doppler_estimates(
        data_unscaled, "unscaled"
    )
    results_rssiscaled: dict[str, list[float]] = doppler_estimates(
        data_rssiscaled, "rssiscaled"
    )

    plot_speed_boxplot(
        results_unscaled, results_scaled, save_path=data_dir / "improvement.pdf"
    )
    methods = {
        "Raw    ": results_unscaled,
        "AGC-removed    ": results_scaled,
        "RSSI-scaled": results_rssiscaled,
    }

    plot_speed_pointrange_compact(methods, save_path=data_dir / "methods-compared.pdf")
    # plot_asus_csi_timeseries(data, 5000)
    # plot_asus_csi_packet_overlays(data)
    # plot_asus_csi_3d(data)


if __name__ == "__main__":
    main()


# ---
def plot_asus_csi_timeseries(data: pl.DataFrame):
    """
    Plot the CSI amplitude timeseries for the receivers 'asus1' and 'asus2'
    in a single figure with two subplots.

    For each receiver:
      - Only the first n captures are plotted (if n is provided).
      - Every 10th subcarrier is plotted as a separate line.
      - Only every 5th time point is plotted to reduce clutter.

    Parameters:
      data (pl.DataFrame): The processed CSI data.
      n (int, optional): The number of time points to plot per receiver.
                         If None, all available captures are used.
    """
    # Filter data to include only rows from 'asus1' and 'asus2'
    asus_data = data.filter(pl.col("receiver_name").is_in(["asus1", "asus2"]))
    receivers = ["asus1", "asus2"]

    # Create a figure with 2 subplots (one per receiver)
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    for ax, receiver in zip(axs, receivers):
        rec_df = asus_data.filter(pl.col("receiver_name") == receiver)
        if rec_df.height == 0:
            print(f"No data found for receiver: {receiver}")
            continue

        # If n is provided, only use the first n captures
        rec_df = rec_df.head(20000)
        rec_df = rec_df.tail(1000)

        # Convert timestamps to seconds (float) and select every 5th time point
        timestamps = np.array(rec_df.get_column("sequence_number").to_list())

        # 'csi_abs' is stored as a list per capture; stack them to get a 2D array.
        # Then select every 5th time sample to reduce the number of points plotted.
        csi_abs_list = rec_df.get_column("csi_abs").to_list()
        csi_abs_array = np.stack(csi_abs_list, axis=0)  # [::5]

        # Determine subcarrier indices to plot (every 10th one)
        num_subcarriers = csi_abs_array.shape[1]
        subcarrier_indices = np.arange(0, num_subcarriers, 10)

        # Plot the amplitude for each selected subcarrier
        for idx in subcarrier_indices:
            ax.plot(timestamps, csi_abs_array[:, idx], label=f"Subcarrier {idx}")

        ax.set_title(f"CSI Amplitude Timeseries for Receiver {receiver}")
        ax.set_ylabel("CSI Amplitude")
        ax.grid(True)
        ax.legend(fontsize=8)

    axs[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    if SHOW:
        plt.show()


def plot_asus_csi_packet_overlays(data: pl.DataFrame, num_overlays: int = 3):
    """
    Plot a time series of the average CSI amplitude per packet for receivers 'asus1' and 'asus2',
    and overlay a few insets that show the full packet (amplitude vs. subcarrier index)
    at time points evenly spaced across the data. Each inset is connected to the main plot
    with an arrow.

    Parameters:
      data (pl.DataFrame): The processed CSI data.
      num_overlays (int): The number of packet overlays (inset plots) to display.
    """
    asus_data = data.filter(pl.col("receiver_name").is_in(["asus1", "asus2"]))
    receivers = ["asus1", "asus2"]

    fig, axs = plt.subplots(len(receivers), 1, figsize=(12, 6), sharex=True)
    if len(receivers) == 1:
        axs = [axs]

    for ax, receiver in zip(axs, receivers):
        rec_df = asus_data.filter(pl.col("receiver_name") == receiver)
        if rec_df.height == 0:
            print(f"No data found for receiver: {receiver}")
            continue

        rec_df = rec_df.head(20000).tail(1000)
        sequence_numbers = np.array(rec_df.get_column("sequence_number").to_list())
        csi_abs_list = rec_df.get_column("csi_abs").to_list()
        csi_abs_array = np.stack(csi_abs_list, axis=0)
        avg_amplitudes = np.mean(csi_abs_array, axis=1)

        ax.plot(sequence_numbers, avg_amplitudes, label="Avg Amplitude", color="blue")
        ax.set_title(f"CSI Time Series and Packet Overlays for {receiver}")
        ax.set_xlabel("Sequence Number")
        ax.set_ylabel("Avg CSI Amplitude")
        ax.grid(True)

        num_packets = len(sequence_numbers)
        overlay_indices = np.linspace(0, num_packets - 1, num_overlays, dtype=int)
        dx = (sequence_numbers[-1] - sequence_numbers[0]) * 0.05
        dy = (np.max(avg_amplitudes) - np.min(avg_amplitudes)) * 0.1

        # Pre-calculate a bounding box size for the insets.
        bbox_width = (sequence_numbers[-1] - sequence_numbers[0]) * 0.1
        bbox_height = (np.max(avg_amplitudes) - np.min(avg_amplitudes)) * 0.1

        for idx in overlay_indices:
            time_point = sequence_numbers[idx]
            avg_amp = avg_amplitudes[idx]
            packet_amplitude = csi_abs_array[idx]
            num_subcarriers = packet_amplitude.shape[0]
            subcarrier_indices = np.arange(num_subcarriers)

            # Create a Bbox for the inset's anchor
            bbox = Bbox.from_bounds(
                time_point + dx, avg_amp + dy, bbox_width, bbox_height
            )

            ax_inset = inset_axes(
                ax,
                width="20%",
                height="20%",
                bbox_to_anchor=bbox,
                bbox_transform=ax.transData,
                loc="upper left",
                borderpad=1,
            )

            ax_inset.plot(
                subcarrier_indices,
                packet_amplitude,
                marker="o",
                linestyle="-",
                color="red",
            )
            ax_inset.set_title(f"Packet {idx}", fontsize=8)
            ax_inset.set_xlim(0, num_subcarriers - 1)
            ax_inset.tick_params(axis="both", which="major", labelsize=6)
            ax_inset.grid(True)

            ax.annotate(
                "",
                xy=(time_point, avg_amp),
                xycoords="data",
                xytext=(time_point + dx, avg_amp + dy),
                textcoords="data",
                arrowprops=dict(arrowstyle="->", color="gray"),
            )

    plt.tight_layout()
    if SHOW:
        plt.show()


def plot_results(results, time_indices):
    """Plot estimated speed for each receiver in a single big subplot."""
    fig, axes = plt.subplots(4, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (receiver, speeds) in enumerate(results.items()):
        axes[idx].plot(time_indices[receiver], speeds, "*")
        axes[idx].set_title(receiver)
        axes[idx].set_xlabel("Time (s)")
        axes[idx].set_ylabel("Estimated Speed (m/s)")
        axes[idx].set_ylim(0, 5)
        axes[idx].grid(True)

    plt.tight_layout()
    if SHOW:
        plt.show()


def plot_results_together(results, time_indices):
    """Plot estimated speed for all receivers in one figure using a seaborn scatter plot."""

    styles = ["-", "--", "-.", ":", "-", "--", "-.", ":"]
    markers = ["o", "s", "D", "*", "x", "^", "v", "p"]

    plt.figure(figsize=(12, 6))
    for (receiver, speeds), style, marker in zip(results.items(), styles, markers):
        plt.plot(
            time_indices[receiver],
            speeds,
            linestyle=style,
            marker=marker,
            label=receiver,
        )

    plt.xlabel("Time (s)")
    plt.ylabel("Estimated Speed (m/s)")
    plt.title("Estimated Speed for All Receivers")
    plt.legend(title="Receiver")
    plt.grid(True)
    # plt.ylim(0.7, 1.3)
    if SHOW:
        plt.show()
