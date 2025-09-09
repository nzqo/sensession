"""
Base experiment (unchanged frame transmission) evaluation plots
"""

import itertools
from pathlib import Path

import numpy as np
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from evaluation.common import RECEIVER_ORDER, fmlp_cmap, tgo_palette
from matplotlib.colors import Normalize

from sensession.campaign import CampaignProcessor


def preprocess(csi: pl.DataFrame, meta: pl.DataFrame):
    """
    Preprocessing steps
    """

    proc = CampaignProcessor(
        csi,
        meta,
        lazy=False,
    )

    # fmt: off
    proc = (proc
        .correct_rssi_by_agc()
        .unwrap()
    )

    proc.csi = proc.csi.filter(
        pl.col("antenna_idxs")
        == pl.col("receiver_name").str.contains("asus").cast(pl.Int8)
    )

    proc = (proc
        #.filter("antenna_idxs", 0)
        #.drop_interpolated_iwl_subcarriers()
        .remove_guard_subcarriers()
        .remove_dc_subcarrier()
        .scale_magnitude(column_alias="csi_abs_scaled")
        .detrend_phase(column_alias="csi_phase_detrended")
        .equalize_magnitude(normed_column="csi_abs_scaled", column_alias="csi_abs_eq")
        .equalize_phase(normed_column="csi_phase_detrended", column_alias="csi_phase_eq")
    )

    proc.meta = proc.meta.with_columns(
        (pl.col("collection_start") - pl.duration(seconds=(pl.col("collection_start").dt.timestamp() // 1_000_000) % 600))
        .dt.truncate("1m")
        .dt.strftime("%Y-%m-%d")
    )

    proc = proc.meta_attach("collection_start")
    # fmt: on

    csi, meta = proc.get()

    # Remove unused columns for simpler working here
    meta = meta.drop(
        "collection_stop",
        "channel",
        "bandwidth",
        "antenna_idxs",
        "stream_idxs",
        "campaign_name",
        "schedule_name",
        "collection_name",
    )

    csi = csi.drop(
        "stream_capture_num",
        "rx_antenna_capture_num",
        "timestamp",
        "sequence_number",
        "rssi",
        "antenna_rssi",
        "collection_name",
        "antenna_idxs",
        "stream_idxs",
    )

    return csi, meta


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def multi_day_stability(df: pl.DataFrame, modality: str = "csi_abs_scaled"):
    """Compute multi-day stability score using cosine similarity for each receiver."""

    profiles = df.group_by(
        ["receiver_name", "collection_start", "subcarrier_idxs"], maintain_order=True
    ).agg(pl.col(modality).mean().alias("mean_values"))

    results = []

    for (receiver_name,), receiver_group in profiles.group_by(
        "receiver_name", maintain_order=True
    ):
        day_profiles = {
            day[0]: group["mean_values"].to_numpy()
            for day, group in receiver_group.group_by(
                "collection_start", maintain_order=True
            )
        }

        if len(day_profiles) < 2:
            continue

        similarities = [
            cosine_similarity(day_profiles[d1], day_profiles[d2])
            for d1, d2 in itertools.combinations(day_profiles, 2)
        ]
        if similarities:
            stability_index = np.prod(similarities) ** (1 / len(similarities))
            results.append(
                {"receiver_name": receiver_name, "stability_index": stability_index}
            )

    result = pl.DataFrame(results)
    print(f"Multi day similarities: {result}")


def profile_similarity(
    df: pl.DataFrame, modality: str = "csi_abs_scaled"
) -> pl.DataFrame:
    # Step 1: Extract unique collection_start values (days)
    days = df.get_column("collection_start").unique().sort().to_list()

    # Step 2: Group by receiver_name, collection_start, and subcarrier_idxs to calculate median profiles
    profiles = df.group_by(
        ["receiver_name", "collection_start", "subcarrier_idxs"], maintain_order=True
    ).agg(pl.col(modality).median().alias("median_values"))

    # Step 3: Group by receiver_name to calculate similarity for each receiver
    results = []
    for receiver, receiver_group in profiles.group_by(
        "receiver_name", maintain_order=True
    ):
        # print(f"{receiver}: {receiver_group.sort('subcarrier_idxs')}")
        # Extract profiles for the two days
        day_1 = (
            receiver_group.filter(pl.col("collection_start") == days[0])
            .sort("subcarrier_idxs")
            .get_column("median_values")
            .to_numpy()
        )
        day_2 = (
            receiver_group.filter(pl.col("collection_start") == days[1])
            .sort("subcarrier_idxs")
            .get_column("median_values")
            .to_numpy()
        )

        # Compute cosine similarity
        similarity = cosine_similarity(day_1, day_2)

        # Append results
        results.append({"receiver_name": receiver, "cosine_similarity": similarity})

    # Step 4: Create and return results DataFrame
    return pl.DataFrame(results)


def calculate_similarities(df: pl.DataFrame):
    sim1 = profile_similarity(df, "csi_abs_scaled")
    sim2 = profile_similarity(df, "csi_phase_detrended")

    print("Magnitude profile similarity:")
    print(sim1)
    print("Phase profile similarity:")
    print(sim2)


def plot_profile(
    df: pl.DataFrame,
    modality: str,
    ylabel: str,
    output_file: Path,
):
    """Generic function to plot CSI profiles for absolute values or phase."""

    plt.figure(figsize=(14, 4))

    ax = sns.boxplot(
        data=df,
        x="subcarrier_idxs",
        y=modality,
        hue="collection_start",
        palette=tgo_palette,
        showfliers=False,
        saturation=0.95,
    )

    ax.tick_params(axis="x", labelrotation=60)
    for label in ax.get_xticklabels()[::2]:
        label.set_visible(False)

    plt.ylabel(ylabel, fontsize=22)
    plt.xlabel("Subcarrier", fontsize=22)
    plt.legend(
        fontsize=18,
        loc="upper left",
        bbox_to_anchor=(0, 1),
        frameon=True,
    )
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.tight_layout()
    plt.savefig(
        data_dir / "img" / output_file,
        format="pdf",
        bbox_inches="tight",
        dpi=300,
    )


def plot_profiles(df: pl.DataFrame):
    for receiver_name, group_df in df.group_by("receiver_name", maintain_order=True):
        receiver_name = receiver_name[0]
        plot_profile(
            group_df,
            "csi_abs_scaled",
            ylabel=r"$|\overline{H}|$",
            output_file=data_dir / "img" / f"abs-profile-{receiver_name}.pdf",
        )
        plot_profile(
            group_df,
            "csi_phase_detrended",
            ylabel=r"$\arg \overline{H}$",
            output_file=data_dir / "img" / f"phs-profile-{receiver_name}.pdf",
        )


def plot_two_profiles(
    df: pl.DataFrame,
):
    """For each receiver_name in df, plot abs & phase profiles side by side with one legend."""
    # define which columns & y-labels to plot
    modalities = [
        ("csi_abs_scaled", r"$|\overline{H}|$"),
        ("csi_phase_detrended", r"$\arg \overline{H}$"),
    ]

    # iterate exactly as you had it
    for receiver_name, group_df in df.group_by("receiver_name", maintain_order=True):
        # if receiver_name is itself a list/tuple, grab the scalar
        rx = (
            receiver_name[0]
            if isinstance(receiver_name, (list, tuple))
            else receiver_name
        )

        # new Figure with two subplots
        fig, axes = plt.subplots(2, 1, figsize=(16, 13), sharey=False, sharex=True)

        for ax, (modality, ylabel) in zip(axes, modalities):
            sns.boxplot(
                data=group_df,
                x="subcarrier_idxs",
                y=modality,
                hue="collection_start",
                palette=tgo_palette,
                showfliers=False,
                saturation=0.95,
                ax=ax,
            )
            ax.tick_params(axis="x", labelrotation=60)
            for lbl in ax.get_xticklabels()[::4]:
                lbl.set_visible(False)

            ax.set_xlabel("Subcarrier", fontsize=26)
            ax.set_ylabel(ylabel, fontsize=26)
            ax.tick_params(axis="both", labelsize=25)
            # Remove per‐subplot legend if seaborn added one
            if ax.legend_:
                ax.legend_.remove()

        # Grab the handles & labels from the *first* axes:
        handles, labels = axes[0].get_legend_handles_labels()

        # Place them *inside* axes[0] at e.g. the upper right:
        axes[1].legend(
            handles,
            labels,
            title="Collection Start",
            loc="upper left",
            frameon=True,
            fontsize=22,
            title_fontsize=24,
            borderpad=1,
            labelspacing=0.5,
        )

        # tighten & adjust for the legend
        fig.tight_layout()
        fig.subplots_adjust(top=0.88)

        # save one PDF per receiver
        out_file = data_dir / "img" / f"profile-comb-{rx}.pdf"
        fig.savefig(out_file, format="pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)


def compute_correlation(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute the correlation matrix for "csi_abs_eq" values across different "subcarrier_idxs".
    """
    # Pivot the dataframe to have subcarriers as columns
    pivot_df = df.pivot(
        values="csi_abs_eq",
        index="capture_num",
        on="subcarrier_idxs",
        maintain_order=True,
    ).drop("capture_num")

    # Compute correlation using Polars
    corr_matrix = pivot_df.corr()

    return corr_matrix


def plot_correlations(df: pl.DataFrame):
    """
    Plot the correlation of "csi_abs_eq" values across different "subcarrier_idxs" for each receiver.
    """
    # Styling constants
    TICK_LABEL_SIZE = 17
    TITLE_FONT_SIZE = 19
    CBAR_TICK_SIZE = 17

    # Create subplots
    unique_receivers = RECEIVER_ORDER
    fig, axes = plt.subplots(
        nrows=2,
        ncols=4,
        figsize=(20, 10),
        sharex="col",
        sharey="row",
        constrained_layout=True,
    )
    axes = axes.flatten()

    # Compute all correlation matrices first to find global vmin and vmax
    corr_matrices, ticks_dict = {}, {}
    vmin, vmax = float("inf"), float("-inf")

    for receiver_name in unique_receivers:
        receiver_df = df.filter(pl.col("receiver_name") == receiver_name)
        corr_data = compute_correlation(receiver_df)
        corr_matrices[receiver_name] = corr_data

        # Get unique subcarrier indexes for ticks
        ticks_dict[receiver_name] = (
            receiver_df.unique("subcarrier_idxs", maintain_order=True)
            .get_column("subcarrier_idxs")
            .to_list()
        )

        # Update global vmin/vmax
        cmin = corr_data.min().min_horizontal().item()
        cmax = corr_data.max().max_horizontal().item()
        vmin, vmax = min(vmin, cmin), max(vmax, cmax)

    # Shared colorbar setup
    sm = plt.cm.ScalarMappable(cmap=fmlp_cmap, norm=Normalize(vmin=vmin, vmax=vmax))

    # Plot each receiver
    for i, receiver_name in enumerate(unique_receivers):
        corr_matrix_pd = corr_matrices[receiver_name].to_pandas()
        ax = axes[i]

        sns.heatmap(
            corr_matrix_pd,
            linewidths=0.5,
            ax=ax,
            xticklabels=ticks_dict[receiver_name],
            yticklabels=ticks_dict[receiver_name],
            cbar=False,
            vmin=vmin,
            vmax=vmax,
            cmap=fmlp_cmap,
        )

        # Title only the receiver name, with larger font
        ax.set_title(receiver_name, fontsize=TITLE_FONT_SIZE)

        # Increase tick label size and rotate
        ax.tick_params(axis="x", labelrotation=60, labelsize=TICK_LABEL_SIZE)
        ax.tick_params(axis="y", labelrotation=30, labelsize=TICK_LABEL_SIZE)

        # Reduce number of visible tick labels
        for num_label, label in enumerate(ax.get_xticklabels()):
            if num_label % 7 != 0:
                label.set_visible(False)
        for num_label, label in enumerate(ax.get_yticklabels()):
            if num_label % 7 != 0:
                label.set_visible(False)

    # Shared axis labels
    fig.supxlabel("Subcarrier Index", fontsize=23)
    fig.supylabel("Subcarrier Index", fontsize=23)

    # Add and style a single colorbar
    cbar = fig.colorbar(
        sm, ax=axes.ravel().tolist(), orientation="vertical", fraction=0.04, pad=0.04
    )
    cbar.ax.tick_params(labelsize=CBAR_TICK_SIZE)

    # Save and close
    plt.savefig(
        data_dir / "img" / "noise-correlation.pdf",
        format="pdf",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)


def plot_profiles_combined(df: pl.DataFrame, output_file: Path):
    """
    Plot a combined figure with two columns (amplitude on the left, phase on the right)
    and one row per receiver, in the order specified by receiver_order.
    """
    # Use receiver_order directly. Optionally, filter receivers that are not in the data.
    available_receivers = set(df.get_column("receiver_name").unique().to_list())
    ordered_receivers = [r for r in RECEIVER_ORDER if r in available_receivers]
    n_receivers = len(ordered_receivers)

    # Create a figure with 2 columns and one row per receiver
    fig, axes = plt.subplots(
        nrows=n_receivers, ncols=2, figsize=(16, 3 * n_receivers), sharey="col"
    )
    # In case there is only one receiver, ensure axes is 2D
    if n_receivers == 1:
        axes = np.array([axes])

    global_handles, global_labels = None, None

    for i, receiver in enumerate(ordered_receivers):
        # Filter the dataframe for the current receiver
        group_df = df.filter(pl.col("receiver_name") == receiver)
        # Convert to Pandas for seaborn plotting
        group_pd = group_df.to_pandas()

        # ---------------------------
        # Amplitude (left column)
        # ---------------------------
        ax_amp = axes[i, 0]
        sns.boxplot(
            data=group_pd,
            x="subcarrier_idxs",
            y="csi_abs_scaled",
            hue="collection_start",
            palette=tgo_palette,
            showfliers=False,
            saturation=0.95,
            ax=ax_amp,
        )
        ax_amp.set_title(f"Receiver: {receiver}", fontsize=16)
        ax_amp.set_ylabel(r"$\overline{H}$", fontsize=14)
        if i == n_receivers - 1:
            ax_amp.set_xlabel("Subcarrier", fontsize=14)
        else:
            ax_amp.set_xlabel("")
        ax_amp.tick_params(axis="x", labelrotation=60)
        for label in ax_amp.get_xticklabels()[::2]:
            label.set_visible(False)

        # Capture the legend handles from the first amplitude plot and then remove it.
        if i == 0:
            handles, labels = ax_amp.get_legend_handles_labels()
            global_handles, global_labels = handles, labels
            ax_amp.legend_.remove()
        else:
            leg = ax_amp.get_legend()
            if leg is not None:
                leg.remove()

        # ---------------------------
        # Phase (right column)
        # ---------------------------
        ax_phase = axes[i, 1]
        sns.boxplot(
            data=group_pd,
            x="subcarrier_idxs",
            y="csi_phase_detrended",
            hue="collection_start",
            palette=tgo_palette,
            showfliers=False,
            saturation=0.95,
            ax=ax_phase,
        )
        ax_phase.set_ylabel(r"$\arg \overline{H}$", fontsize=14)
        if i == n_receivers - 1:
            ax_phase.set_xlabel("Subcarrier", fontsize=14)
        else:
            ax_phase.set_xlabel("")
        ax_phase.tick_params(axis="x", labelrotation=60)
        for label in ax_phase.get_xticklabels()[::2]:
            label.set_visible(False)
        leg = ax_phase.get_legend()
        if leg is not None:
            leg.remove()

    # Add one global legend at the top of the figure
    if global_handles and global_labels:
        fig.legend(
            global_handles,
            global_labels,
            loc="upper center",
            ncol=len(global_labels),
            fontsize=14,
        )

    # Adjust layout so the global legend does not overlap the subplots
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig(output_file, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    valid_iwl_indices = list(range(-28, -1, 2)) + list(range(-1, 28, 2)) + [28]
    data_dir = Path.cwd() / "data" / "base_campaign_ch11"

    # --- Plot rsquared as measure of linearity
    df = pl.read_parquet(data_dir / "csi.parquet")
    meta = pl.read_parquet(data_dir / "meta.parquet")

    df, meta = preprocess(df, meta)

    plot_two_profiles(df)
    exit(0)
    # calculate_similarities(df)
    multi_day_stability(df)
    plot_profiles_combined(df, output_file=data_dir / "img" / "profiles-combined.pdf")

    first_session = df.unique("collection_start", maintain_order=True).item(
        1, "collection_start"
    )
    df = df.filter(pl.col("collection_start") == first_session)
    plot_correlations(df)
