"""
Picoscenes file parser

File parser for log files from PicoScenes. Relies on the picoscenes python
toolbox for the initial parsing, but offers some further functionality for
further extraction and conversion to our internal data format (dataframes)
"""

from typing import cast
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from loguru import logger

try:
    from picoscenes import Picoscenes  # pylint: disable=no-name-in-module

    PICOSCENES_AVAILABLE = True
except ImportError:
    PICOSCENES_AVAILABLE = False


from sensession.tools.tool import CsiGroup, CsiDataPoint


@dataclass
class CsiExtractionConfig:
    """
    Collection of data required to extract CSI from picoscenes frames
    """

    num_antennas: int
    num_streams: int
    num_tones: int
    antenna_idxs: list[int]
    stream_idxs: list[int]
    subcarrier_idxs: np.ndarray


def _get_csi(frame: dict, config: CsiExtractionConfig) -> np.ndarray | None:
    """
    Extract CSI

    Args:
        frame  : Picoscenes Frame to extract data from
        config : Parameters required for proper extraction and reshaping
    """

    # According to the maintainers, layout is identical to MATLAB Toolbox, where we have an
    # [n_tones, n_streams, n_antennas] flattened array: https://ps.zpj.io/matlab.html
    # Because of MATLAB using Fortran column order, this is transposed in python to:
    # [num_rx_antennas, num_streams, num_subcarriers]
    n_tones = frame.get("CSI", {}).get("numTones", 0)
    if n_tones != config.num_tones:
        logger.warning("Frame with wrong number of tones encountered: {n_tones}")
        return None

    csi = np.array(frame.get("CSI", {}).get("CSI", {})).reshape(
        (config.num_antennas, config.num_streams, config.num_tones)
    )

    # Extract the antennas used for capture
    # NOTE: This would be cleaner if done via tool invocation, using the --rxcm chainmask command.
    # However, this command breaks reception with the ax210. Therefore, we dont do it.
    # NOTE: We need to reshape because slicing may drop dimensions
    csi = csi[config.antenna_idxs, config.stream_idxs, :].reshape(
        (len(config.antenna_idxs), len(config.stream_idxs), config.num_tones)
    )

    return csi[:, :, np.flatnonzero(config.subcarrier_idxs)]


def _extract_antenna_rssi(frame: dict, antenna_idxs: list[int]) -> list[np.int8]:
    """
    Extract per-antenna RSSI for all captured frames

    Args:
        frames       : Picoscenes frame collection
        antenna_idxs : List of antennas used for capture
    """
    # Picoscenes stores antenna rssi in rssi1, rssi2, rssi3
    rssi_strs = [f"rssi{idx + 1}" for idx in antenna_idxs]
    return [frame.get("RxSBasic", {}).get(key) for key in rssi_strs]


def get_data(
    file: Path,
    antenna_idxs: list[int],
    stream_idxs: list[int],
) -> tuple[CsiGroup, list[int]] | None:
    """
    Parse data from file
    """
    logger.trace(f"Starting to extract data from Picoscenes capture file {file}")

    frames = Picoscenes(str(file))
    subcarrier_idxs = np.array(
        frames.raw[0].get("CSI").get("SubcarrierIndex"), dtype=np.int16
    )

    # Extract some needed variables
    extraction_cfg = CsiExtractionConfig(
        num_antennas=frames.raw[0].get("RxSBasic").get("numRx"),
        num_streams=frames.raw[0].get("RxSBasic").get("numSTS"),
        num_tones=frames.raw[0].get("CSI").get("numTones"),
        antenna_idxs=antenna_idxs,
        stream_idxs=stream_idxs,
        subcarrier_idxs=subcarrier_idxs,
    )

    data = [
        CsiDataPoint(
            timestamp=frame.get("RxSBasic").get("systemns"),
            sequence_num=frame.get("StandardHeader").get("Sequence"),
            csi=csi,
            rssi=frame.get("RxSBasic").get("rssi"),
            antenna_rssi=_extract_antenna_rssi(frame, antenna_idxs),
            agc=frame.get("RxExtraInfo").get("agc"),
        )
        for frame in frames.raw
        if (csi := _get_csi(frame, extraction_cfg)) is not None
    ]

    logger.trace(f"Parsed {len(data)} data points from PicoScenes file.")

    if len(data) == 0:
        return None

    subc_indices = extraction_cfg.subcarrier_idxs
    subc_list = cast(list[int], subc_indices[np.flatnonzero(subc_indices)].tolist())
    return data, subc_list


def load_picoscenes_data(
    file: Path,
    antenna_idxs: list[int],
    stream_idxs: list[int],
) -> tuple[CsiGroup, list[int]] | None:
    """
    Parse picoscene log files to extract CSI data

    Args:
        file         : Path to picoscenes output file
        antenna_idxs : Indices of used antennas in the capture
        stream_idxs  : List of streams to extract CSI from
    """
    if not PICOSCENES_AVAILABLE:
        raise ModuleNotFoundError("Picoscenes package not installed; can't parse data")

    # If file is empty, failsafe return empty DatFrame
    if not file.exists() or file.stat().st_size == 0:
        logger.warning(f"File {file} is empty, no data to load!")
        return None

    return get_data(file, antenna_idxs, stream_idxs)
