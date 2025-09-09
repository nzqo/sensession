"""
A few commonly used preprocessing steps for sessions
"""

# see sensession.processing.processor
# mypy: disable-error-code="arg-type, union-attr"
import gc
from typing import Self, Type
from pathlib import Path

import polars as pl
from loguru import logger

from sensession.database import DataFrameType, get_csi_schema
from sensession.processing import CsiProcessor


class CampaignProcessor(CsiProcessor):
    """
    Preprocessor class to bundle preprocessing methods
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        csi: DataFrameType | None = None,
        meta: DataFrameType | None = None,
        meta_attach_cols: set[str] | None = None,
        checkpoint_dir: Path = Path.cwd() / ".cache" / "checkpoints",
        lazy: bool = False,
    ):
        _meta_attach_cols = {
            "meta_id",
            "antenna_idxs",
            "stream_idxs",
            "subcarrier_idxs",
            "collection_name",
            "receiver_name",
        }

        if meta_attach_cols:
            _meta_attach_cols.update(meta_attach_cols)

        super().__init__(csi, meta, _meta_attach_cols, checkpoint_dir, lazy)

    def equalize_magnitude(
        self, normed_column: str = "csi_abs", column_alias: str = "csi_abs"
    ) -> Self:
        """
        Profile correction to flatten CSI magnitudes.

        This method first calculates a "CSI Magnitude Profile" by averaging across all
        captures per session with the warmup/training runs. Averaging is per subcarrier.
        Then, we shape-normalize by dividing the remaining values in every session by
        those profiles, which should then yield a flat CSI with respect to the unchanged
        channel.
        """
        self._ensure_unwrapped()

        # Get schedule name and relevant columns
        df = (
            self.csi.select(
                "meta_id",
                "receiver_name",
                "antenna_idxs",
                "subcarrier_idxs",
                "collection_name",
                normed_column,
            )
            .join(
                self.meta.select("meta_id", "schedule_name"),
                on="meta_id",
                how="left",
                maintain_order="left",
            )
            .drop("meta_id")
        )

        logger.info(
            "Equalizing CSI magnitude: calculating group profiles on warmup rows."
        )

        # Instead of a window sum on the full data, filter for warmup rows first.
        profiles = (
            df.filter(pl.col("collection_name").str.contains("warmup"))
            .drop("collection_name")
            .group_by(
                ["schedule_name", "receiver_name", "antenna_idxs", "subcarrier_idxs"],
                maintain_order=True,
            )
            .agg(pl.col(normed_column).mean().alias("session_abs_profile"))
        )

        # Join the small 'profiles' DataFrame back to the full dataset.
        df = df.drop("collection_name").join(
            profiles,
            on=["schedule_name", "receiver_name", "antenna_idxs", "subcarrier_idxs"],
            how="left",
            maintain_order="left",
        )

        # Drop columns no longer needed to keep memory usage low.
        df = df.select(normed_column, "session_abs_profile")
        gc.collect()

        # Compute the normalized column.
        df = df.with_columns(
            (pl.col(normed_column) / pl.col("session_abs_profile")).alias(column_alias)
        ).select(column_alias)

        # Reattach the normalized column back to the full DataFrame.
        # (Assumes that the row order remains unchanged by the lazy computation.)
        self.csi = self.csi.with_columns(df.get_column(column_alias)).filter(
            pl.col(column_alias).is_not_nan()
        )
        return self

    def equalize_phase(
        self, normed_column: str = "csi_phase", column_alias: str = "csi_phase"
    ) -> Self:
        """
        Profile correction for phases. Same methodology as amplitude correction.
        """
        self._ensure_unwrapped()

        df = self.csi
        if "collection_name" not in self.csi.columns:
            df = df.join(self.meta.select("meta_id", "collection_name"), on="meta_id")

        df = (
            df.select(
                "meta_id",
                "receiver_name",
                "antenna_idxs",
                "subcarrier_idxs",
                "collection_name",
                normed_column,
            )
            .join(
                self.meta.select("meta_id", "schedule_name"),
                on="meta_id",
                how="left",
                maintain_order="left",
            )
            .drop("meta_id")
        )

        logger.info("Equalizing CSI phase: calculating group profiles on warmup rows.")

        profiles = (
            df.filter(pl.col("collection_name").str.contains("warmup"))
            .drop("collection_name")
            .group_by(
                ["schedule_name", "receiver_name", "antenna_idxs", "subcarrier_idxs"],
                maintain_order=True,
            )
            .agg(pl.col(normed_column).mean().alias("session_phase_profile"))
        )

        df = df.drop("collection_name").join(
            profiles,
            on=["schedule_name", "receiver_name", "antenna_idxs", "subcarrier_idxs"],
            how="left",
            maintain_order="left",
        )

        df = df.select(normed_column, "session_phase_profile")
        gc.collect()

        df = df.with_columns(
            (pl.col(normed_column) - pl.col("session_phase_profile")).alias(
                column_alias
            )
        ).select(column_alias)

        self.csi = self.csi.with_columns(df.get_column(column_alias))
        return self

    def remove_phase_outliers(self, n: int = 3, phase_col: str = "csi_phase") -> Self:
        """
        Remove data points with highly irregular phases as measured by high
        deviation from the mean.
        """
        self._ensure_unwrapped()
        df = self.csi

        logger.info(
            "Removing phase outliers whenever a value is more than {n} stddevs from the mean... "
        )

        df = (
            df.filter(pl.col("collection_name").str.contains("warmup"))
            .group_by(
                "meta_id",
                "subcarrier_idxs",
                "antenna_idxs",
                maintain_order=True,
            )
            .agg(
                pl.col(phase_col).mean().alias("session_phase_profile"),
                pl.col(phase_col).std().alias("session_phase_std"),
            )
            .join(
                df,
                on=[
                    "meta_id",
                    "subcarrier_idxs",
                    "antenna_idxs",
                ],
            )
        )

        # Get normalized deviations and remove all data points with
        outliers = (
            df.with_columns(
                devs=(
                    (pl.col(phase_col) - pl.col("session_phase_profile"))
                    / pl.col("session_phase_std")
                ).abs()
            )
            .filter(pl.col("devs").is_not_nan())
            .group_by(self.stream_index, maintain_order=True)
            .agg(pl.col("devs").mean())
            .filter(pl.col("devs") > n)
        )

        self.csi = df.join(outliers, on=self.stream_index, how="anti").drop(
            "session_phase_std", "session_phase_profile"
        )

        return self

    def _align_sequence_nums(self, orig_df: DataFrameType) -> DataFrameType:
        # To align, we find pairs of (collection, sequence numb) that we want to keep.
        # To save on some RAM, start by subselecting only required columns for that.
        df = orig_df.select("collection_name", "receiver_name", "sequence_number")

        # We keep track of the total number of receivers participating in each collection
        num_receivers = df.group_by("collection_name", maintain_order=True).agg(
            pl.col("receiver_name").n_unique().alias("n_receivers")
        )

        # Next, we figure out if there are any receivers that have seen the same sequence
        # number twice in the same collection. If that happened, we can not align anymore,
        # since aligning requires unique sequence numbers.
        #
        # We check for all pairs of (collection, receiver, sequence number, stream, antenna)
        # whether there is a duplicate.
        #
        # NOTE: This could likely be solved with timestamps. But I don't want to implement that
        # since I don't need it right now.
        duplicate_sequence_nums = (
            df.with_row_index("count_idx")
            .group_by(
                "collection_name",
                "receiver_name",
                "sequence_number",
            )
            .agg(pl.col("count_idx").len())
            .filter(pl.col("count_idx") > 1)
        )

        # Discard all duplicates. Afterwards, all pairs of (collection, receiver, sequence)
        # will be unique.
        df = df.join(
            duplicate_sequence_nums,
            on=["collection_name", "receiver_name", "sequence_number"],
            how="anti",
        )

        # We can proceed the aligning by ensuring that only captures with sequence numbers
        # seen by all receivers are kept, while the rest is discarded.
        # First figure out how many receivers were taking part within each collection
        df = (
            df.group_by("collection_name", "sequence_number")
            .agg(pl.col("receiver_name").n_unique().alias("seen_by"))
            .join(num_receivers, on="collection_name")
            .filter(pl.col("seen_by") == pl.col("n_receivers"))
        )

        # Semi-joining with the original df ensures only desired rows are kept
        return orig_df.join(df, on=["sequence_number", "collection_name"], how="semi")

    def align_sequence_nums(self) -> Self:
        """
        Sequence number sequentially enumerates the packets sent to invoke CSI collection.

        To align sequence numbers, we discard all data points that were missed by at least
        one of the receivers participating in a session.
        """
        self._ensure_wrapped()
        logger.info("Aligning sequence numbers ...")
        self.csi = self._align_sequence_nums(self.csi)
        return self

    def drop_contains(
        self, column: str = "collection_name", value: str = "warmup"
    ) -> Self:
        """
        Drop rows based on columns containing a specified value
        """
        logger.info(f"Dropping all rows in which column {column} contains '{value}'. ")
        self.csi = self.csi.filter(~pl.col(column).str.contains(value))
        self.meta = self.meta.filter(~pl.col(column).str.contains(value))

        return self

    def discard_low_counts(
        self,
        min_count: int = 600,
    ) -> Self:
        """
        Discard all sessions in which the number of CSI captured is below threshold.

        Args:
            min_count  : The threshold minimum count of CSI values
        """
        self._ensure_unwrapped()
        df = self.csi
        logger.info(
            f"Discarding sessions with low (<{min_count}) CSI capture counts ... "
        )

        capture_count = df.group_by("meta_id").len("n_captures")
        keep_metas = capture_count.filter(pl.col("n_captures") >= min_count)
        self.csi = self.csi.filter(pl.col("meta_id").is_in(keep_metas))
        self.meta = self.meta.filter(pl.col("meta_id").is_in(keep_metas))

        return self

    def fill_missing_sequence_numbers(self, sequence_len: int) -> Self:
        """
        Fill up sequence with nulls where there are missing items, where being missing is
        recognized by sequence numbers.

        NOTE: If sequence numbers are not increased by one between successive frames,
        this method can't work. If you still use it, that's on you.
        """
        self._ensure_wrapped()
        logger.info("Inserting Nulls for missing sequence number values...")

        # We construct the complete index by creating all combinations of meta_id
        # and sequence number.
        meta_ids = self.meta.select("meta_id").unique()

        df_type: Type[pl.LazyFrame | pl.DataFrame] = (
            pl.LazyFrame if self.lazy else pl.DataFrame
        )
        complete_index = df_type(
            {"sequence_number": range(0, sequence_len)},
            schema={"sequence_number": pl.UInt16},
        )
        complete_index = meta_ids.join(complete_index, how="cross")

        # Left-join allows us to add rows of missing sequence number with values being Null
        self.csi = complete_index.join(
            self.csi, on=["meta_id", "sequence_number"], how="left"
        )

        # Caveat: Not only values such as CSI but also meta columns such as labels are Null
        # in the newly added rows. We use a simple filling strategy to alleviate that.
        # We don't know which rows are metadata statically, but we know which aren't. We
        # can form the set of rows to fill by exclusion.
        csi_schema = get_csi_schema()
        non_predict_columns = set(list(csi_schema)) - {"meta_id"}

        self.csi = self.csi.with_columns(
            pl.exclude(non_predict_columns)
            .fill_null(strategy="forward")
            .fill_null(strategy="backward")
            .over("meta_id")
        )

        return self

    def lin_interpolate_nulls(self) -> Self:
        """
        Interpolate nulls linearly.

        """
        self._ensure_unwrapped()
        logger.info("Interpolating missing (Null) values.")

        csi_schema = get_csi_schema()
        value_cols = set(list(csi_schema)) - {"meta_id"}

        self.csi = self.csi.with_columns(
            pl.col(value_cols)
            .interpolate()
            .forward_fill()
            .backward_fill()
            .over(["meta_id", "antenna_idxs", "stream_idxs", "subcarrier_idxs"])
        )
        return self
