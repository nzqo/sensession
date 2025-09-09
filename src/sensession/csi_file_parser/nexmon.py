"""
Nexmon file parser

Parses tcpdump files received from Nexmon. For now, relies on the provided
Matlab script to do the actual extraction, float conversion, etc.
"""

from typing import cast
from pathlib import Path
from operator import itemgetter

import numpy as np
import scipy.io as sio
from loguru import logger

from sensession.config import Bandwidth
from sensession.tools.tool import CsiGroup, CsiDataPoint
from sensession.util.temp_file import TempFile
from sensession.util.matlab_parallelizer import EngineWrapper


def get_subcarrier_idxs(bandwidth: Bandwidth) -> np.ndarray:
    """
    Get subcarrier indices for data reported by Nexmon
    """
    match bandwidth.in_mhz():
        case 20:
            subc_idxs = np.concatenate((np.arange(-28, 0), np.arange(1, 29)))
        case 40:
            subc_idxs = np.concatenate((np.arange(-58, -1), np.arange(2, 59)))
        case 80:
            subc_idxs = np.concatenate((np.arange(-122, -1), np.arange(2, 123)))
        case _:
            raise ValueError(
                f"Bandwidth {bandwidth.in_mhz()} not supported in nexmon file parsing"
            )
    return subc_idxs


def read_nexmon_csi(  # pylint: disable=too-many-locals
    engine,
    csi_file: Path,
    bandwidth: Bandwidth,
    antenna_idxs: list[int],
    stream_idxs: list[int],
) -> tuple[CsiGroup, list[int]] | None:
    """
    Read nexmon data from file and return as dict

    Args:
        engine        : matlab engine to use for call to file parser
        csi_file      : Path to capture file
        bandwidth     : Channel bandwidth used during capture
        antenna_idxs  : Indices of antennas used in capture
        stream_idxs   : List of streams to extract CSI from
    """
    num_antennas = len(antenna_idxs)
    num_streams = len(stream_idxs)

    # Run matlab CSI extractor and store in file. Do not use the returns from
    # the matlab function here because of memory leaks. Open matlab instances
    # of engines would cause a memory overflow after a while.
    with TempFile(
        csi_file.name + ".postprocessed.mat", csi_file.parent
    ) as postproc_file:
        engine.read_csi(
            str(csi_file.resolve()),
            bandwidth.in_mhz(),
            num_antennas,
            num_streams,
            str(postproc_file.path.resolve()),
        )

        # Read the temp file and extract the variables
        postproc = sio.loadmat(postproc_file.path)
    subc_idxs = get_subcarrier_idxs(bandwidth)

    fft_size = int(bandwidth.in_mhz() * 3.2)
    subc_shift = fft_size // 2

    timestamps, csi, antenna_rssi, sequence_nums = itemgetter(
        "timestamps",
        "csi",
        "rssi",
        "sequence_nums",
    )(postproc)

    # Properly format the retrieved values
    # NOTE: Need to filter out the proper subcarrier number ourselves.
    subcarrier_idxs = cast(list[int], subc_idxs.tolist())

    timestamps = timestamps.flatten().astype(int).tolist()
    sequence_nums = sequence_nums.flatten().tolist()
    csi = csi[:, :, :, subc_idxs + subc_shift]

    # Sanity check CSI shape before formatting to list
    num_captures = len(sequence_nums)
    num_subcarrs = len(subcarrier_idxs)

    if num_captures == 0:
        return None

    assert csi.shape == (
        num_captures,
        num_antennas,
        num_streams,
        num_subcarrs,
    ), f"Wrong CSI shape, got: {csi.shape}"

    assert antenna_rssi.shape == (
        num_captures,
        num_antennas,
    ), f"Wrong antenna RSSI shape, got: {antenna_rssi.shape}"
    rssi = (
        (10 * np.log10(np.mean(np.power(10, antenna_rssi / 10), 1)))
        .astype(int)
        .tolist()
    )

    # Assemble into dict of named argument values for database/dataframe creation
    logger.trace(f"Nexmon file parser: Loaded {num_captures} data points!")

    # Assemble data and convert to dataframe to finish
    return (
        [
            CsiDataPoint(
                timestamp=ts,
                sequence_num=sn,
                csi=csi,
                rssi=rssi,
                antenna_rssi=arssi.tolist(),
            )
            for ts, sn, csi, rssi, arssi in zip(
                timestamps, sequence_nums, csi, rssi, antenna_rssi
            )
        ],
        subcarrier_idxs,
    )


# -------------------------------------------------------------------------------------
# Module-wide matlab instance solely used for nexmon postprocessing.
# Theoretically, we could probably write the pool used in the frame generation so
# that it would allow for a reuse here.
# For now, doing at least some caching of such an instance is already good enough.
engine_singleton: EngineWrapper | None = None


def lazy_init_singleton():
    """
    Initialize the matlab engine singleton if it wasn't initialized yet
    """
    matlab_nexmon_path = Path.cwd() / "matlab" / "nexmon_csi"
    global engine_singleton  # pylint: disable=global-statement
    if not engine_singleton:
        logger.trace(
            "Starting matlab subprocess to extract nexmon data from tmp file ..."
        )
        engine_singleton = EngineWrapper(
            processing_callback=read_nexmon_csi, start_path=matlab_nexmon_path
        )


def load_nexmon_data(
    file: Path,
    bandwidth: Bandwidth,
    antenna_idxs: list[int],
    stream_idxs: list[int],
) -> tuple[CsiGroup, list[int]] | None:
    """
    Load nexmon data from file and parse into dataframe

    Args:
        file          : Path fo pcap capture file
        antenna_idxs  : List of used antenna indices
        bandwidth     : Bandwidth used in capture, in MHz
        stream_idxs   : List of streams to extract CSI from
    """
    # Lazy init the engine singleton only if we use this function!
    lazy_init_singleton()
    assert engine_singleton, (
        "Matlab engine required for nexmon processing not initialized"
    )

    # If file is empty, failsafe return empty DatFrame
    if not file.exists() or not file.is_file() or file.stat().st_size == 0:
        return None

    logger.trace("Processing nexmon data in Matlab singleton instance ...")
    return engine_singleton.process(
        kwargs={
            "csi_file": file,
            "bandwidth": bandwidth,
            "antenna_idxs": antenna_idxs,
            "stream_idxs": stream_idxs,
        }
    )
