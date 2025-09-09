"""
Single subcarriers experiment

Check how amplitude modifications on single subcarriers are detected.
"""

# pylint: disable=duplicate-code
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
    mask[sc_idx] *= 2.5 * np.exp(1j * np.pi * 0.5)
    return mask


def get_fixture() -> ExperimentFixture:
    """
    Configure experiment fixture
    """
    idx = 16  # random normal subcarrier

    fixture = ExperimentFixture("methodology", gain=0)

    fixture.add_schedule_for_mask(
        mask=singlesc_mask(idx, num_packets=1000),
        schedule_name=f"method_idx_{idx}",
        group_reps=1,
        modified_idx=(idx - 32, pl.Int8),
        scale=(2, pl.UInt8),
    )

    return fixture


def run():
    """run experiment"""
    fixture = get_fixture()
    fixture.run()


if __name__ == "__main__":
    run()
