"""
Database implementation

We use polars LazyFrames as "database", persisting them into parquet files
"""

# see sensession.processing.processor
# mypy: disable-error-code="arg-type, union-attr, type-var"
from __future__ import annotations

import json
import traceback
from typing import Type
from pathlib import Path
from datetime import datetime

import numpy as np
import polars as pl
from loguru import logger
from polars._typing import TimeUnit

from sensession.config import APP_CONFIG
from sensession.util.hash import get_timed_hash
from sensession.tools.tool import (
    DEFAULT_TIMESTAMP_UNIT,
    CsiMeta,
    CsiGroup,
    CaptureResult,
)

DataFrameType = pl.DataFrame | pl.LazyFrame

# fmt: off
# -------------------------------------------------------------------------------------
# -- Database Schema Definition
# -----------------------------
# We keep the data in separate files:
# - One main parquet file for just the CSI data
# - Another file for metadata that changes infrequently
#
# The files are linked by the `meta_id`. Whenever context or any of the metaparameters
# change, so does the `meta_id`.
# -------------------------------------------------------------------------------------
CSI_LIST = (
    pl.List(                         # List 1: One element per receive antenna
        inner=pl.List(               # List 2: One element per spatial stream
            inner=pl.List(           # List 3: One element per subcarrier
                inner=pl.Float64     # Inner level : The actual data
            )
        )
    )
)
# fmt: on
def get_csi_schema(timestamp_unit: TimeUnit = DEFAULT_TIMESTAMP_UNIT):
    """
    Captured Data Schema.

    CSI from multiple antennas is laid out flat in the CSI lists.
    The `used_antenna_idxs` may be used to infer to which antenna CSI belongs

    Args:
        timestamp_unit : Timestamp (integer) unit
    """
    # fmt: off
    return {
        "meta_id"          : pl.String,                                # Pointer to metadata in meta database
        "timestamp"        : pl.Datetime(timestamp_unit),              # Timestamp (may be local to capture device)
        "sequence_number"  : pl.UInt16,                                # Sequence number of frame from which CSI was extracted
        "csi_abs"          : CSI_LIST,                                 # Absolute values of CSI symbols
        "csi_phase"        : CSI_LIST,                                 # Phase values of CSI symbols
        "rssi"             : pl.Int16,                                 # Reported Received Signal Strength
        "antenna_rssi"     : pl.List(inner=pl.Int16),                  # Per-antenna Signal strength (for antennas specified in idxs)
        "agc"              : pl.Int8,                                  # Automatic Gain Control Gain
    }
    # fmt: on


def get_meta_schema():
    """
    Get DataFrame metadata schema. Labels map collected data onto a specific sensing
    session performed within an experiment.
    """
    # fmt: off
    return {
        "meta_id"             : pl.String,                             # Identifier so that data can point to this entry
        "collection_start"    : pl.Datetime(DEFAULT_TIMESTAMP_UNIT),   # When the collection was started
        "collection_stop"     : pl.Datetime(DEFAULT_TIMESTAMP_UNIT),   # When the collection was stopped
        "channel"             : pl.UInt16,                             # Channel that was sent on
        "bandwidth"           : pl.UInt16,                             # Channel bandwidth in MHz
        "antenna_idxs"        : pl.List(inner=pl.UInt8),               # idxs of antennas used to capture
        "stream_idxs"         : pl.List(inner=pl.UInt8),               # idxs of used spatial streams
        "subcarrier_idxs"     : pl.List(inner=pl.Int16),               # Subcarrier idx to which CSI list index i corresponds
        "receiver_name"       : pl.String,                             # Name of the receiver
        "filter_frame"        : pl.String,                             # Digest of the frame that was used
    }
    # fmt: on


def get_df_type(lazy: bool) -> Type[DataFrameType]:
    """
    Get either of the DataFrame types dependent on lazy variable
    """
    return pl.LazyFrame if lazy else pl.DataFrame


# -------------------------------------------------------------------------------------
# -- Database Conversion Helpers
# ------------------------------
# Some helper functions to create suitable DataFrames for the database.
# -------------------------------------------------------------------------------------
def csigroup_to_dataframe(
    group: CsiGroup,
    meta_id: str,
    lazy: bool,
    timestamp_unit: TimeUnit = DEFAULT_TIMESTAMP_UNIT,
) -> DataFrameType:
    """
    Create a LazyFrame from a group of captured CSI values
    """

    # fmt: off
    data = [
        {
            "meta_id"         : meta_id,
            "timestamp"       : dp.timestamp,
            "sequence_number" : dp.sequence_num,
            "csi_abs"         : np.abs(dp.csi).tolist(),
            "csi_phase"       : np.angle(dp.csi).tolist(),
            "rssi"            : dp.rssi,
            "antenna_rssi"    : dp.antenna_rssi,
            "agc"             : dp.agc,
        } for dp in group
    ]
    # fmt: on

    # Cast to common time unit of nanoseconds
    df_type = get_df_type(lazy)
    return df_type(data, schema=get_csi_schema(timestamp_unit)).with_columns(
        pl.col("timestamp").dt.cast_time_unit(DEFAULT_TIMESTAMP_UNIT)
    )


def meta_to_dataframe(
    metadata: CsiMeta | None, meta_id: str, lazy: bool
) -> DataFrameType:
    """
    Convert a CSI metadata struct to a DataFrame
    """
    df_type = get_df_type(lazy)
    if not metadata:
        return df_type({}, schema=get_meta_schema())

    assert metadata.channel, "Metadata must have channel specified"

    # fmt: off
    return df_type(
        {
            "meta_id"          : [meta_id],
            "collection_start" : [metadata.collection_start],
            "collection_stop"  : [metadata.collection_stop],
            "receiver_name"    : [metadata.receiver_name],
            "filter_frame"     : [metadata.filter_frame] or [""],
            "antenna_idxs"     : [metadata.antenna_idxs],
            "subcarrier_idxs"  : [metadata.subcarrier_idxs],
            "stream_idxs"      : [metadata.stream_idxs],
            "channel"          : [metadata.channel.number],
            "bandwidth"        : [metadata.channel.bandwidth.in_mhz()],
        },
        schema=get_meta_schema(),
    )
    # fmt: on


# -------------------------------------------------------------------------------------
# -- Database Wrapper Class
# -------------------------
# The following section contains a database wrapper that handles persistence and
# manages data over time by appending to present data.
# NOTE: If databases become too large, this part could handle a database backend
# or at least work with LazyFrames.
# -------------------------------------------------------------------------------------
def _read_from_disk(filepath: Path, lazy: bool = False) -> DataFrameType:
    """
    Read a parquet file from disk

    Args:
        filepath : Path to the file to read.
    """
    df: DataFrameType = pl.scan_parquet(filepath) if lazy else pl.read_parquet(filepath)

    logger.debug(
        f"Database/Dataframe loaded -- Recovered previous data from {filepath}."
    )
    logger.trace(
        f"Dataframe schema: \n{json.dumps(df.collect_schema(), default=str, indent=4)}"
    )
    return df


def frame_is_empty(df: DataFrameType) -> bool:
    """
    Check if LazyFrame is empty or not
    """
    return (
        df.limit(1).collect().is_empty()
        if isinstance(df, pl.LazyFrame)
        else df.is_empty()
    )


class Database:
    """
    Database wrapper around LazyFrame to automatically append and store data
    """

    def __init__(
        self,
        db_path: Path | str,
        append: bool = True,
        lazy: bool = APP_CONFIG.lazy_database,
    ):
        if isinstance(db_path, str):
            db_path = Path(db_path)

        logger.trace(f"Opening database at: {db_path}")
        if not (db_path / "csi.parquet").is_file():
            logger.warning("Database doesnt exist yet. Creating empty one.")

        # Keep
        self.tmp_data: list[pl.DataFrame | pl.LazyFrame] = []
        self.tmp_meta: list[pl.DataFrame | pl.LazyFrame] = []

        self.changed = False
        self.collected = True
        self.lazy = lazy

        df_type = get_df_type(lazy)
        self.csi = df_type(schema=get_csi_schema())
        self.meta = df_type(schema=get_meta_schema())
        self.errors = df_type()

        if not db_path:
            raise RuntimeError("Please provide a valid Path")

        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True, parents=True)

        self.csi_path = self.db_path / "csi.parquet"
        self.meta_path = self.db_path / "meta.parquet"
        self.error_path = self.db_path / "errors.csv"

        if append:
            self._read()

    def __enter__(self) -> Database:
        logger.info(f"Opening database. Mode: {'lazy' if self.lazy else 'eager'}")
        return self

    def __exit__(self, exc_type, exc_value, tb) -> bool:
        logger.info("Wrapping up database usage (writing to disk) ...")
        self.write()

        if exc_type is not None:
            trace = "".join(traceback.format_exception(exc_type, exc_value, tb))
            logger.error(f"Encountered exception in Database context: {trace}")
            return False

        return True

    def _read(self):
        """
        Read database from folder

        NOTE: Per convention, both database files (csi.parquet and meta.parquet) must be
        present in the directory
        """

        if self.csi_path.is_file() and self.meta_path.is_file():
            self.tmp_data.append(_read_from_disk(self.csi_path, self.lazy))
            self.tmp_meta.append(_read_from_disk(self.meta_path, self.lazy))
            self.collected = False

        if self.error_path.is_file() and self.error_path.stat().st_size > 10:
            if self.lazy:
                self.errors = pl.scan_csv(self.error_path)
            else:
                self.errors = pl.read_csv(self.error_path)

    def collect(self):
        """
        Manifest temporarily stored CSI into a dataframe
        """
        # If there isnt more than the "current" values cached, we dont need to collect.
        if self.collected:
            return

        # Collect temp list into a single dataframe.
        self.csi = pl.concat(self.tmp_data)
        self.meta = pl.concat(self.tmp_meta)

        # Reset to not collect again
        self.tmp_data = [self.csi]
        self.tmp_meta = [self.meta]
        self.collected = True

    def write(self):
        """
        Persist values to database files
        """
        self.collect()

        if not self.changed:
            logger.debug("Database hasn't changed; nothing to write.")
            return

        if len(self.tmp_data) == 0 and len(self.tmp_meta) == 0:
            logger.debug("No data in database; nothing to write.")
            return

        if self.lazy:
            # NOTE: Streaming into the same file is quite oof. Instead, we stream into
            # a temp file, which we then move to be the old file. Afterwards, reload to
            # be safe.
            tmp_csi_path = self.csi_path.with_suffix(".tmp")
            tmp_meta_path = self.meta_path.with_suffix(".tmp")
            tmp_error_path = self.error_path.with_suffix(".tmp")

            self.csi.sink_parquet(tmp_csi_path)
            self.meta.sink_parquet(tmp_meta_path)
            self.errors.sink_csv(tmp_error_path)

            tmp_csi_path.rename(self.csi_path)
            tmp_meta_path.rename(self.meta_path)
            tmp_error_path.rename(self.error_path)

            # Since we are in lazy mode, this just reloads the scan handle and
            # should not cause much overhead therefore.
            self._read()
        else:
            self.csi.write_parquet(self.csi_path)
            self.meta.write_parquet(self.meta_path)
            self.errors.write_csv(self.error_path)

        # If someone decides to call write twice... well, it was already written.
        # Unless it changed again, we won't do anything.
        self.changed = False

    def add_data(self, data: CaptureResult, meta_id: str | None = None, **extra_meta):
        """
        Add data to the database

        Args:
            data : Result from a data capture, including data and metadata
            meta_id : An id for the metadata to allow linking between csi and meta.
                Will be auto-generated by default.
            extra_meta : Keyword args of additional metadata; Each arg must be of
                the form (value, polars datatype).
        """
        if meta_id is None:
            logger.debug("No `meta_id` provided, generating random one.")
            meta_id = get_timed_hash(data.receiver_id)

        if not data:
            logger.debug("No data to add. Ignoring ...")
            return

        assert data.csi, "Valid data must contain CSI"
        assert data.meta, "Valid data must have metainfo"

        # Put CSI into dataframe
        csi_df = csigroup_to_dataframe(
            group=data.csi,
            meta_id=meta_id,
            lazy=self.lazy,
            timestamp_unit=data.meta.timestamp_unit,
        )

        # Put metadata with additional columns
        meta_df = meta_to_dataframe(data.meta, meta_id=meta_id, lazy=self.lazy)

        if extra_meta:
            meta_df = meta_df.with_columns(
                **{
                    column: pl.lit(value, dtype=dtype)
                    for column, (value, dtype) in extra_meta.items()
                }
            )

        n_data = csi_df.select(pl.len())
        if self.lazy:
            n_data = n_data.collect()

        num_data = n_data.item()  # pylint: disable=no-member # (we ensured its a dataframe)
        logger.debug(
            "Adding CSI data to database ...\n"
            + f" -- receiver name.: {data.meta.receiver_name} (id: {data.receiver_id})\n"
            + f" -- meta id.......: {meta_id}\n"
            + f" -- num data......: {num_data}\n"
        )

        if frame_is_empty(csi_df) or frame_is_empty(meta_df):
            logger.warning("Found empty frames; Not adding to database")
            return

        self.tmp_data.append(csi_df)
        self.tmp_meta.append(meta_df)
        self.changed = True
        self.collected = False

    def add_errors(self, error_list: list):
        """
        Add a list of errors. Errors may be of a nested type (e.g. dict)
        with accompanying information. They will be stored in human-readable
        format (csv).
        """
        if len(error_list) == 0:
            return

        df_type = get_df_type(self.lazy)
        error_df = df_type(error_list)
        if frame_is_empty(self.errors):
            self.errors = error_df
        else:
            self.errors = pl.concat([self.errors, df_type(error_list)])
        self.changed = True

    def remove_after(self, time: datetime):
        """
        Remove all entries in the database after given time.

        Args:
            time : Timestamp after which to remove all data
        """
        # Only keep what was before
        self.collect()
        self.meta = self.meta.filter(pl.col("experiment_start_time") < time)
        self.csi = self.csi.join(self.meta, on="meta_id", how="semi")
        self.changed = True

    def remove_before(self, time: datetime):
        """
        Remove all entries in the database before given time.

        Args:
            time : Timestamp before which to remove all data
        """
        # Only keep what was after
        self.collect()
        self.meta = self.meta.filter(pl.col("experiment_start_time") > time)
        self.csi = self.csi.join(self.meta, on="meta_id", how="semi")
        self.changed = True

    def get_csi(self) -> pl.DataFrame:
        """
        Get the internal CSI DataFrame
        """
        self.collect()
        if self.lazy:
            return self.csi.collect()
        assert isinstance(self.csi, pl.DataFrame)
        return self.csi

    def get_meta(self) -> pl.DataFrame:
        """
        Get the internal Metadata DataFrame
        """
        self.collect()
        if self.lazy:
            return self.meta.collect()
        assert isinstance(self.meta, pl.DataFrame)
        return self.meta
