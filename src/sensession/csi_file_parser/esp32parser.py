"""
ESP32 csi file parser
"""

from pathlib import Path

import numpy as np

from sensession.tools.tool import CsiGroup, CsiDataPoint


def parse_line(
    line: str,
    subcarrier_idxs: list[int],
    subcarrier_mask: list[int],
) -> tuple[np.uint64, np.int8, np.uint8, np.uint8, np.uint16, np.ndarray]:
    """
    Parse a single line to extract timestamp and csi_vals
    Maybe switch to binary format in the future
    """
    # MACs parsed but not currently used for anything
    # pylint: disable=W0612
    (
        timestamp_str,
        smac,
        dmac,
        rssi,
        agc,
        fft_gain,
        rx_seq,
        n_str,
        array_str,
    ) = line.split(";")

    csi_cleaned = array_str.strip("[] \n")
    csi_vals = list(map(float, csi_cleaned.split(",")))
    # imag part first, then real
    # (see https://docs.espressif.com/projects/esp-idf/en/stable/esp32s3/api-guides/wifi.html#wi-fi-channel-state-information)
    # NOTE: Esp has only one antenna, and can thus also receive from only one stream.
    # We have a doubly-nested one element list because of that

    # Need to reshape to (n_rx = 1, n_sts = 1, n_subcarrier)
    csi = np.array(csi_vals[1::2]) + 1j * np.array(csi_vals[::2])
    csi = csi[subcarrier_mask]  # filter unwanted subcarriers
    csi = csi[np.argsort(subcarrier_idxs)]  # sort by subcarrier idx
    csi = csi.reshape(1, 1, -1)

    return (
        np.uint64(timestamp_str),
        np.int8(rssi),
        np.uint8(agc),
        np.uint8(fft_gain),
        np.uint16(rx_seq),
        csi,
    )


def load_esp32_data(
    file: Path, subcarrier_idxs: list[int], subcarrier_mask: list[int]
) -> CsiGroup | None:
    """
    Load esp32 data from file and parse into dataframe
    """

    # If file is empty, failsafe return empty DatFrame
    if not file.exists() or not file.is_file() or file.stat().st_size == 0:
        return None

    data: CsiGroup = []

    with open(file.resolve(), "r", encoding="UTF-8") as h_file:
        for line in h_file:
            timestamp, rssi, agc, fft_gain, seq_num, csi = parse_line(
                line, subcarrier_idxs, subcarrier_mask
            )

            data.append(
                CsiDataPoint(
                    timestamp=timestamp,
                    sequence_num=seq_num,
                    csi=csi,
                    rssi=rssi,
                    antenna_rssi=[rssi],
                    agc=agc,
                    fft_gain_esp=fft_gain,
                )
            )

    return data
