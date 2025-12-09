"""
Module to perform preprocessing of the data on disk, e.g. subsampling and selection
of relevant features to save some memory later.
"""

import sys
from typing import List
from pathlib import Path
from datetime import timedelta

import polars as pl
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="TRACE")


def subsample(
    df: pl.DataFrame,
    target_num: int,
    safe_period_ms: int = 1000,
    label_col_name: str = "activity_idx",
):
    """
    Subsample each of the 700 packet CSI sequences to a target of target_num
    packets. This helps in fighting data loss.
    """
    logger.trace("Starting subsampling; This can take a while...")

    # Period in which there is at least one value. If this isn't the case,
    # subsampling will crash. This is fine because in such a case subsampling
    # doesn't make much sense. Missing values are essentially substituted by
    # the closest available ones. If that "closest" is actually far away, that
    # would suck.
    safe_period = timedelta(milliseconds=safe_period_ms)

    # NOTE: The dynamic groupby itself does not perform subsampling, it simply
    # performs windowed grouping. That means that every "subsample_timedelta",
    # it will group all values within a "500ms" period from then.
    # To subsample, we take the very first appearing values in all the windows.
    aggs = [
        pl.col("meta_id").first(),
        pl.col("csi_abs").first(),
        pl.col("csi_phase").first(),
        pl.col("sequence_number").first(),
        pl.col("antenna_rssi").first().alias("rssi"),
    ]

    # Drop null values, then perform the subsampling and truncate to the target
    # subsampling number
    def grouper(group: pl.DataFrame) -> pl.DataFrame:
        min_time = group.select(pl.min("timestamp")).item()
        max_time = group.select(pl.max("timestamp")).item()
        total_time = max_time - min_time
        if total_time > timedelta(days=1):
            logger.error(
                f"Problematic metaid: {group.select('meta_id').unique()} (time: {total_time})"
            )
            with pl.Config(tbl_cols=-1):
                print(group)

        subsample_timedelta = total_time / target_num

        a = (
            group.sort("timestamp")
            .group_by_dynamic(
                "timestamp", every=subsample_timedelta, period=safe_period
            )
            .agg(*aggs)
            .head(target_num)
        )
        return a

    return df.drop_nulls().group_by("meta_id", maintain_order=True).map_groups(grouper)


def prepare_data(
    data_dir: Path,
    num_csi_per_sample_window: int = 800,
    target_dir: Path = None,
    position: List[int] = [0, 1, 2, 3],
):
    """
    Prepare data from the raw captured data:

        - Drop non-needed data and columns
        - Subsample to have homogenous data dimensions (each window being the same length)
    """
    if data_dir == target_dir:
        logger.error("Please dont write back to original file!")
        exit(1)

    if target_dir == None:
        logger.error("Specify a path to store results to")
        exit(1)

    logger.trace(f"Reading data from {data_dir}")
    target_dir.mkdir(exist_ok=True, parents=True)

    csi = pl.read_parquet(data_dir / "csi.parquet")
    meta = pl.read_parquet(data_dir / "meta.parquet")

    # filtering to one position
    # logger.trace(f"Selecting and filtering relevant data ...")
    # meta = meta.filter(pl.col("setting").is_in(["E1n"]))
    # meta = meta.filter(pl.col("position").is_in(position))
    # csi = csi.filter(pl.col("meta_id").is_in(meta.select("meta_id")))

    # Select only relevant columns for our experiments to minimize memory usage
    csi = csi.select(
        "meta_id",
        "timestamp",
        "sequence_number",
        "csi_abs",
        "csi_phase",
        "antenna_rssi",
    )
    meta = meta.select(
        "meta_id",
        # "setting",
        "activity_idx",
        "human_label",
        "position",
        # "curve_label",
        "receiver_name",
        "rep_nr",
        "subcarrier_idxs",
    )

    csi = csi.join(meta, on="meta_id", how="semi")

    # Perform subsampling for every of the captures to ensure matching window sizes
    logger.trace(f"Performing subsampling ...")
    csi = subsample(csi, num_csi_per_sample_window)
    offenders = (
        csi.group_by("meta_id")
        .len()
        .filter(pl.col("len") != num_csi_per_sample_window)
        .get_column("meta_id")
        .to_list()
    )

    if offenders:
        logger.warning(f"Found buggy captures: {offenders}. Removing.")
        csi = csi.filter(~pl.col("meta_id").is_in(offenders))
        meta = meta.filter(~pl.col("meta_id").is_in(offenders))

    logger.trace("Done with subsampling")

    logger.trace(f"Writing data back to preprocessed file ...")
    logger.info(f"Final shapes: csi: {csi.shape}, meta: {meta.shape}")
    csi.write_parquet(target_dir / f"csi.parquet")
    meta.write_parquet(target_dir / f"meta.parquet")


if __name__ == "__main__":
    # Subsample
    prepare_data(
        Path.cwd() / "data" / "har_fabian", target_dir=Path.cwd() / "data" / "_test"
    )
