"""
Single subcarrier precoding sweep experiment evaluation plots.

Expects you have run preprocessing before to get sensitivity stats!
> python -m evaluation.single_scs.preprocess

Or downloaded the results (single_phases_results and single_scs_results)
"""

import sys
from enum import Enum
from pathlib import Path

import polars as pl
from loguru import logger
from evaluation.common import (
    subcarrier_barplot,
    subcarrier_dual_barplot,
)


class Mode(Enum):
    AMP = 0
    PHASE = 1


def plot_sensitivity(df: pl.DataFrame, img_dir: Path):
    """
    Args:
        df: Dataframe including columns
            `correlation`, `spearman_corr`, and `mutual_info`
    """
    df = df.with_columns(
        mi_sensitivity=df["mutual_info"],
    )

    # Plot our sensitivity measure
    subcarrier_barplot(
        df,
        y="mi_sensitivity",
        ylabel="MI",
        file=img_dir / "mi-sensitivity.pdf",
    )


def plot_deviations(df: pl.DataFrame, mode: Mode, img_dir: Path):
    """
    Args:
        df: Dataframe including correlation statistics (named `correlation` and `spearman_corr`)
    """
    label = r"$\bar{D}^{phs}_k(H)$" if mode == Mode.PHASE else r"$\bar{D}^{amp}_k(H)$"
    subcarrier_dual_barplot(
        df,
        y="residual",
        ylabel=label,
        file=img_dir / "mean-prd.pdf",
    )


def main(mode: Mode, exp_t: str):
    # See 802.11n-2009 Table 7-25f
    if "80mhz" in exp_t:
        valid_iwl_indices = []  # iwl cant capture 80.
        edge = 122
    elif "40mhz" in exp_t:
        valid_iwl_indices = list(range(-58, 59, 4))
        edge = 58
    else:
        valid_iwl_indices = list(range(-28, -1, 2)) + list(range(-1, 28, 2)) + [28]
        edge = 28

    name = "single_phases_results" if mode == Mode.PHASE else "single_scs_results"
    data_dir = Path.cwd() / "data" / name
    img_dir = data_dir / "img" / f"{exp_t}"
    img_dir.mkdir(exist_ok=True, parents=True)

    # --- Plot rsquared as measure of linearity
    df = pl.read_parquet(data_dir / f"sensitivity_{exp_t}.parquet")
    df = df.filter(pl.col("antenna_idx") == 0)

    df = df.filter(
        (df["receiver_name"] != "iwl5300")
        | (df["modified_idx"].is_in(valid_iwl_indices))
    )
    if mode == Mode.PHASE:
        df = df.filter(~pl.col("modified_idx").is_in([-edge, edge]))

    logger.trace("Read data.")

    plot_sensitivity(df, img_dir)
    logger.trace("Finished sensitivity plots")

    plot_deviations(df, mode, img_dir)
    logger.trace("Finished plotting deviations")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")

    for exp_t in ["ch01", "ch06", "ch11", "ch36", "ch40", "ch44", "ch157"]:
        main(Mode.AMP, exp_t)
        main(Mode.PHASE, exp_t)
