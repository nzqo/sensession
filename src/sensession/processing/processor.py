"""
A few commonly used preprocessing steps
"""
# This file is a pain to type. We use either both LazyFrames or both DataFrames
# for CSI and metadata below. However, mypy doesn't understand this. It also
# doesn't understand that this is linked to the `lazy` variable, which we use
# to dispatch to the respective calls.

# mypy: disable-error-code="arg-type, union-attr"
import inspect
from typing import Self
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger

from sensession.database import (
    DataFrameType,
    frame_is_empty,
    get_csi_schema,
    get_meta_schema,
)


def safe_explode(df: pl.DataFrame, *cols, row_idx: str | None = None) -> pl.DataFrame:
    """
    Safely explode, i.e. allow optionally non-existent column names
    """
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return df
    if row_idx:
        df = df.with_row_index(row_idx)
    return df.explode(*existing)


class CsiProcessor:
    """
    Preprocessor class to bundle preprocessing methods
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        csi: DataFrameType | None = None,
        meta: DataFrameType | None = None,
        meta_attach_cols: set[str] | None = None,
        checkpoint_dir=Path.cwd() / ".cache" / "postprocessing_checkpoints",
        lazy: bool = False,
    ):
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.checkpoint_dir = checkpoint_dir
        self.lazy = lazy

        if lazy:
            assert not isinstance(csi, pl.DataFrame), "Lazy mode must use LazyFrame"
            assert not isinstance(meta, pl.DataFrame), "Lazy mode must use LazyFrame"
        else:
            assert not isinstance(csi, pl.LazyFrame), "Use lazy mode for LazyFrame"
            assert not isinstance(csi, pl.LazyFrame), "Use lazy mode for LazyFrame"

        self.meta_attach_cols = {
            "meta_id",
            "antenna_idxs",
            "stream_idxs",
            "subcarrier_idxs",
        }
        if meta_attach_cols is not None:
            self.meta_attach_cols.update(meta_attach_cols)

        self.wrapped: bool = True
        if meta is not None:
            assert csi is not None, "Meta provided -> Must also give CSI"
            mod = "csi_abs" if "csi_abs" in csi.columns else "csi_phase"
            self.wrapped = csi.collect_schema()[mod] == pl.List
            self.meta: DataFrameType = meta
            self.csi: DataFrameType = csi.join(
                meta.select(self.meta_attach_cols),
                on="meta_id",
                maintain_order="left",
            )
        elif lazy:
            self.csi = pl.LazyFrame({}, schema=get_csi_schema())
            self.meta = pl.LazyFrame({}, schema=get_meta_schema())
        else:
            self.csi = pl.DataFrame({}, schema=get_csi_schema())
            self.meta = pl.DataFrame({}, schema=get_meta_schema())

        # Column names for different indices. These are created during unwrapping such that
        # rows can still be assigned to their origin. This way, even after unwrapping we can
        # distinguish which antenna/stream combination a CSI pair belongs to.
        self.capture_index = "capture_num"
        self.antenna_index = "rx_antenna_capture_num"
        self.stream_index = "stream_capture_num"

    def get(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Get CSI and meta after processing
        """
        if self.lazy:
            # NOTE: This function coerces us to manifest the dataframes anyway.
            # To not redo this computation all the time, why not use this...
            self.csi = self.csi.collect().lazy()
            self.meta = self.meta.collect().lazy()

            # Since we just collected, collect is now NOOP just converting to DataFrame
            return self.csi.collect(), self.meta.collect()
        assert isinstance(self.csi, pl.DataFrame)
        assert isinstance(self.meta, pl.DataFrame)
        return self.csi, self.meta

    def load_checkpoint(self, name: str) -> Self | None:
        """
        Load data from a named checkpoint if it exists.
        """
        subdir = self.checkpoint_dir / name

        csi_file = subdir / "csi.parquet"
        meta_file = subdir / "meta.parquet"
        if not (csi_file.is_file() and meta_file.is_file()):
            return None

        if self.lazy:
            self.csi = pl.scan_parquet(csi_file)
            self.meta = pl.scan_parquet(meta_file)
        else:
            self.csi = pl.read_parquet(csi_file)
            self.meta = pl.read_parquet(meta_file)

        mod = "csi_abs" if "csi_abs" in self.csi.columns else "csi_phase"
        self.wrapped = self.csi.collect_schema()[mod] == pl.List
        return self

    def save_checkpoint(self, name: str) -> Self:
        """
        Store a checkpoint so it can be retrieved later.
        """
        if frame_is_empty(self.csi) and frame_is_empty(self.meta):
            logger.info("No data in database; nothing to write.")
            return self

        subdir = self.checkpoint_dir / name
        subdir.mkdir(exist_ok=True, parents=True)
        csi_file = subdir / "csi.parquet"
        meta_file = subdir / "meta.parquet"

        if self.lazy:
            self.csi.collect(engine="streaming").write_parquet(csi_file)
            self.meta.collect(engine="streaming").write_parquet(meta_file)
        else:
            self.csi.write_parquet(csi_file)
            self.meta.write_parquet(meta_file)

        # We load this checkpoint to overwrite the CSI, effectively truncating
        # the LazyFrame Plan. Otherwise, upon every manifestation, everything
        # already computed for this checkpoint (required for streaming) is
        # recomputed.
        loaded = self.load_checkpoint(name)
        assert loaded is not None, "Just saved the checkpoint; Must be loadable"
        return loaded

    def meta_attach(self, *col_names: str) -> Self:
        """
        Attach column(s) from metadata to CSI dataframe.
        Accepts a variable number of column names as arguments.
        """
        self.csi = self.csi.join(
            self.meta.select("meta_id", *col_names), on="meta_id", maintain_order="left"
        )
        return self

    def _ensure_wrapped(self):
        if not self.wrapped:
            raise RuntimeError(
                f"Function {inspect.stack()[1][3]} must be called on wrapped data; Don't call `unwrap` before!"
            )

    def _ensure_unwrapped(self):
        if self.wrapped:
            raise RuntimeError(
                f"Function {inspect.stack()[1][3]} must be called on unwrapped data; first call `unwrap`!"
            )

    def _ensure_outfile_exists(self, out_file: Path | str | None):
        if out_file:
            if isinstance(out_file, str):
                out_file = Path(out_file)
            out_file.parent.mkdir(exist_ok=True, parents=True)

    def unwrap(self, unwrap_phase: bool = True) -> Self:
        """
        Unwrap nested lists, phases, etc.
        """
        self._ensure_wrapped()

        logger.info("Unwrapping Dataframe")

        # First, unwrap the antenna-nesting
        self.csi = safe_explode(
            self.csi,
            "antenna_idxs",
            "antenna_rssi",
            "csi_abs",
            "csi_phase",
            row_idx=self.capture_index,
        )

        # stream level
        self.csi = safe_explode(
            self.csi, "stream_idxs", "csi_abs", "csi_phase", row_idx=self.antenna_index
        )

        # If desired, unwrap phases (to avoid [0,2pi) wraparound)
        if unwrap_phase and "csi_phase" in self.csi.columns:
            self.csi = self.csi.with_columns(
                csi_phase=pl.col("csi_phase").map_elements(
                    lambda x: np.unwrap(np.array(x)).tolist(),
                    return_dtype=pl.List(inner=pl.Float64),
                ),
            )

        # Then, unwrap subcarrier data into separate rows
        self.csi = safe_explode(
            self.csi,
            "subcarrier_idxs",
            "csi_abs",
            "csi_phase",
            row_idx=self.stream_index,
        )

        self.wrapped = False
        return self

    def drop(self, *columns: str) -> Self:
        """
        Drop one or more columns to save on RAM.
        """
        logger.info(f"Dropping columns: {columns}")
        self.csi = self.csi.drop(columns)
        return self

    def filter(self, column: str, value) -> Self:
        """
        Filter
        """
        logger.info(f"Filtering column {column} by value {value}")

        self.csi = self.csi.filter(pl.col(column) == value)
        return self

    def fill_null_antenna_rssi(self, use_rssi: bool = False) -> Self:
        """
        In case antenna rssi was not reported, fill it just to ensure shape matches.
        """
        self._ensure_wrapped()

        fill_val = pl.col("rssi") if use_rssi else pl.lit(None)
        self.csi = self.csi.with_columns(
            pl.col("antenna_rssi").fill_null(
                pl.concat_list([fill_val.repeat_by(pl.col("antenna_idxs").list.len())])
            )
        )

        return self

    def correct_rssi_by_agc(self, drop_agc: bool = True) -> Self:
        """
        Apply correction to RSSI based on AGC values. This is only done where
        such values are available, other rows are not changed.

        NOTE: In the current tools, this affects the iwl5300. Other tools' RSSIs
        seem to be adjusted for that already, as comparison experiments show.

        Return:
            Dataframe with AGC corrected RSSI, where AGC values available
        """
        self._ensure_wrapped()

        logger.info("Correcting RSSI using AGC values ... ")
        df = self.csi.with_row_index("temp_idx")

        # Sum up per-antenna RSSI values. NOTE: They are summed up in linear scale, hence
        # we first convert to that, sum, then convert back to dB again. See also:
        # https://github.com/dhalperi/linux-80211n-csitool-supplementary/blob/master/matlab/get_total_rss.m
        df = df.join(
            df.select("temp_idx", "antenna_rssi")
            .explode("antenna_rssi")
            .with_columns(
                combined_rssi=10 ** (pl.col("antenna_rssi") / 10),
            )
            .group_by("temp_idx", maintain_order=True)
            .agg(pl.col("combined_rssi").sum())
            .with_columns(combined_rssi=10 * pl.col("combined_rssi").log(base=10)),
            on="temp_idx",
            maintain_order="left",
        ).drop("temp_idx")

        # Where applicable, we then correct the combined rssi value by the AGC power
        self.csi = df.with_columns(
            rssi=pl.when(pl.col("receiver_name").str.contains("iwl5300"))
            .then(pl.col("combined_rssi") - 44 - pl.col("agc"))
            .otherwise(pl.col("rssi")),
        ).drop("combined_rssi")

        if drop_agc:
            self.csi = self.csi.drop("agc")

        return self

    def scale_magnitude(
        self,
        column_alias: str = "csi_abs",
        exclude_expr: pl.Expr | None = None,
    ) -> Self:
        """
        Normalize CSI magnitudes for each data point separately.

        It is well known that power and voltage are related by:
                        P = V^2 / R
        Assuming the resistance in our circuit is constant and that signals are
        represented by voltage, we can either power-normalize by using the sum of
        squares or voltage-normalize by using just the sum of CSI values.

        Voltage normalization seems to be more stable, hence this is what is done
        in this scaling method.

        Args:
            column_alias : A possible alias to give to the rescaled magnitude CSI.
                Defaults to overwriting "csi_abs"
            exclude_range : A possible range of subcarriers to exclude in voltage

        """
        self._ensure_unwrapped()
        df = self.csi

        logger.info("Scaling per-datapoint magnitude by dividing with CSI-voltage ... ")

        # Range exclusion is important to get a solid "baseline" to normalize to.
        # If we expect spikes from precoding in a subcarrier range, they would also
        # affect the normalization.
        if exclude_expr is not None:
            df = df.filter(exclude_expr)

        # Keep only necessary columns for the next parts
        df = df.select(self.stream_index, "csi_abs")

        # calculate voltage
        volt = df.group_by(self.stream_index, maintain_order=True).agg(
            pl.col("csi_abs").mean().alias("voltage")
        )

        # Join voltage back to full CSI and perform normalization
        df = self.csi.select(self.stream_index, "csi_abs").join(
            volt, on=self.stream_index, maintain_order="left"
        )
        df = df.with_columns(pl.col("csi_abs") / pl.col("voltage")).drop("voltage")

        # Put volage corrected CSI in base dataframe
        self.csi = self.csi.with_columns(df.get_column("csi_abs").alias(column_alias))

        return self

    def rescale_csi_by_rssi(
        self, column_alias: str = "csi_abs", exclude_expr: pl.Expr = pl.lit(False)
    ) -> Self:
        """
        Rescale CSI with RSSI values to adjust for AGC scaling.
        Normalization according to eq. 4.7 in:
        https://rgu-repository.worktribe.com/output/2071646
        """
        self._ensure_unwrapped()
        logger.info("Rescaling CSI to RSSI amplitudes ... ")

        self.csi = (
            self.csi.join(
                self.csi.group_by(self.stream_index, maintain_order=True).agg(
                    (pl.col("csi_abs") ** 2).sum().alias("rxchain_power"),
                ),
                on=self.stream_index,
                maintain_order="left",
            )
            .with_columns(
                pl.when(exclude_expr)
                .then(pl.col("csi_abs"))
                .otherwise(
                    pl.col("csi_abs")
                    * ((10 ** (pl.col("rssi") / 10)) / pl.col("rxchain_power")) ** 0.5
                )
                .alias(column_alias)
            )
            .drop("rxchain_power")
        )

        return self

    def detrend_phase(
        self, column_alias="csi_phase", is_sorted: bool = True, pin_edges: bool = True
    ) -> Self:
        """
        Detrend phases by linear correction.

        Phases are subject to a few offsets from different receiver inaccuracies. For example,
        CFO results in a constant offset, SFO in a subcarrier-linear offset, etc.

        The first mention of this correction method I could find was from the 2014 paper:
            PADS: Passive Detection of Moving Targets with Dynamic Speed using PHY Layer
            Information

        We simply fix the edge subcarriers to have zero phase and apply a linear correction
        across subcarriers for normalization.

        Args:
            column_alias: Potential alias to give for the resulting phases
            is_sorted: Whether CSI is sorted in subcarrier dimension
            pin_edges: Whether to use edge pinning or ensure phase average is zero.
        """
        self._ensure_unwrapped()

        # NOTE: This part would be cleaner if done earlier in the preprocessing.
        # However, we are filtering out invalid subcarriers only at the end of magnitude
        # processing, and this influences phase normalization.
        logger.info(
            "Detrending phase -- fixing outermost phases to zero and subtracting linear interpolation "
        )

        # This is quite expensive, we only do it if we must.
        if not is_sorted:
            self.csi = self.csi.select(
                pl.all().sort_by("subcarrier_idxs").over(self.stream_index)
            )

        # Columns to use for linear phase normalization
        phase_helper_cols = {
            "phase_slope": (
                pl.col("csi_phase").list.last() - pl.col("csi_phase").list.first()
            )
            / (
                pl.col("subcarrier_idxs").list.last()
                - pl.col("subcarrier_idxs").list.first()
            ),
            "reference_sc": pl.col("subcarrier_idxs").list.first()
            if pin_edges
            else pl.col("subcarrier_idxs").list.mean(),
            "sc_offset": pl.col("csi_phase").list.first()
            if pin_edges
            else pl.col("csi_phase").list.mean(),
        }

        # Aggregate phases and calculate helper columns
        phase_helpers = (
            self.csi.group_by(self.stream_index, maintain_order=True)
            .agg("csi_phase", "subcarrier_idxs")
            .select(self.stream_index, **phase_helper_cols)
        )

        # To correct linear trend: Shift down by first value, then remove linear slope
        # in dependence of subcarrier number
        lin_corr_expr = (
            pl.col("csi_phase")
            - pl.col("sc_offset")
            - (
                (pl.col("subcarrier_idxs") - pl.col("reference_sc"))
                * pl.col("phase_slope")
            )
        )

        # Get sub-section of df for memory usage and correct phase
        df = self.csi.select(self.stream_index, "csi_phase", "subcarrier_idxs")
        df = df.join(phase_helpers, on=self.stream_index, maintain_order="left").drop(
            self.stream_index
        )
        df = df.with_columns(lin_corr_expr)

        # Write fixed phase back to original df
        self.csi = self.csi.with_columns(df.get_column("csi_phase").alias(column_alias))

        return self

    def remove_edge_subcarriers(self, num: int = 1) -> Self:
        """
        Remove edge subcarriers.

        This is useful because of two reasons:
        - Mainly because Nexmon currently yields a fixed, garbage value for one edge carrier.
          With this, we can create a balanced dataset.
        - Filters cause more extreme shapes at the edges, possibly one would like to focus on
          more central subcarriers

        Args:
            num : Number of subcarriers to remove PER EDGE
        """
        self._ensure_unwrapped()
        logger.info(f"Removing outermost {num} edge subcarriers (per edge). ")

        # First get the outermost subcarriers
        subcs = (
            self.csi.select("subcarrier_idxs")
            .unique("subcarrier_idxs")
            .select(pl.col("subcarrier_idxs").sort().slice(num, pl.len() - 2 * num))
        )

        # Join to filter the remaining ones
        self.csi = self.csi.join(
            subcs, on="subcarrier_idxs", how="semi", maintain_order="left"
        )

        return self

    def remove_guard_subcarriers(self, abs_edge: int = 28) -> Self:
        """
        Remove guard subcarriers, i.e. indices greater than abs_edge or smaller than -abs_edge

        abs_edge : Greatest subcarrier value (absolute) to include
        """
        self._ensure_unwrapped()
        logger.info(
            f"Removing guard subcarriers; i.e. truncating to [-{abs_edge}, {abs_edge}] "
        )

        # Truncate to edge
        self.csi = self.csi.filter(pl.col("subcarrier_idxs").abs() <= abs_edge)

        return self

    def remove_dc_subcarrier(self) -> Self:
        """
        Remove DC subcarrier (0)
        """
        self._ensure_unwrapped()
        logger.info("Removing DC subcarrier (idx 0)")

        # Remove DC subcarrier
        self.csi = self.csi.filter(pl.col("subcarrier_idxs").abs() != 0)

        return self

    def drop_interpolated_iwl_subcarriers(self) -> Self:
        """
        Remove the interpolated subcarrier data reported by PicoScenes.
        """
        self._ensure_unwrapped()
        logger.info("Removing interpolated iwl indices")

        # Dont use interpolated values from picoscenes
        valid_iwl_indices = list(range(-28, -1, 2)) + list(range(-1, 28, 2)) + [28]

        # If receiver column was attached before, we dont do anything.
        # Otherwise we attach it and remove it after to leave the dataframe in the same state.
        recv_present = True
        if "receiver_name" not in self.csi.columns:
            self.csi = self.csi.join(
                self.meta.select("meta_id", "receiver_name"),
                on="meta_id",
                maintain_order="left",
            )
            recv_present = False

        self.csi = self.csi.filter(
            (pl.col("receiver_name") != "iwl5300")
            | (pl.col("subcarrier_idxs").is_in(valid_iwl_indices))
        )

        if not recv_present:
            self.csi = self.csi.drop("receiver_name")

        return self
