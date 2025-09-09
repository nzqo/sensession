"""
Single subcarriers experiment

Check how amplitude modifications on single subcarriers are detected.
"""
# pylint: disable=duplicate-code

import itertools

import numpy as np
import polars as pl
from common import ExperimentFixture


def singlesc_mask(sc_idx: int, num_packets: int) -> np.ndarray:
    """
    Get single subcarrier scaling mask with values mapped in the range [0, 2] over num_packets steps,
    and repeated 64 times for each packet.
    """
    # Create a mask of shape (64, num_packets)
    mask = np.ones((64, num_packets), dtype=np.complex64)

    # Set each column in the mask to the corresponding value
    mask[sc_idx, :] = np.linspace(0, 2, num_packets, endpoint=True)
    return mask


def get_fixture() -> ExperimentFixture:
    """
    Configure experiment fixture
    """
    idxs = list(range(64))
    # factors = [0.0, 0.05, 0.5, 0.75, 1.25, 2, 3, 4.5]

    fixture = ExperimentFixture("single_scs")

    for sc_idx, rep in itertools.product(idxs, range(10)):
        fixture.add_schedule_for_mask(
            mask=singlesc_mask(sc_idx, num_packets=1000),
            schedule_name=f"idx_{sc_idx}_rep_{rep}",
            group_reps=1,
            modified_idx=(sc_idx - 32, pl.Int8),
            scale_range=(2, pl.UInt8),
            rep_nr=(rep, pl.UInt32),
        )

    return fixture


def run():
    """run experiment"""
    fixture = get_fixture()
    fixture.run()


if __name__ == "__main__":
    run()
