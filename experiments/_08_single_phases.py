"""
Phase jump experiments

Phases can be detected relative to another, which means that we need to
introduce a step function at the place where we want to see a difference
between neighbors
"""

import itertools

import numpy as np
import polars as pl
from common import ExperimentFixture


def phasejump_mask(sc_idx: int, num_packets: int) -> np.ndarray:
    """
    Phase jump at sc_idx, i.e. step function from there
    """
    base_mask = np.ones((64, num_packets), dtype=np.complex64)

    # Introduce phase rotation on just the respective subcarrier in a sweeping fashion
    # starting from a rotation of 0 up to 0.5 * pi.
    base_mask[sc_idx, :] *= np.exp(
        1j * np.linspace(0, 0.5, num_packets, endpoint=True) * np.pi
    )
    return base_mask


def get_fixture() -> ExperimentFixture:
    """
    Configure experiment fixture
    """
    idxs = list(range(1, 63))

    # Masking happens after rescaling to ensure masked and unmasked frames are comparable.
    # To ensure that masking doesn't cause out-of-scale problems, choose a lower rescale
    # factor.
    fixture = ExperimentFixture("single_phases")

    for sc_idx, rep in itertools.product(idxs, range(10)):
        fixture.add_schedule_for_mask(
            mask=phasejump_mask(sc_idx, num_packets=1000),
            schedule_name=f"idx_{sc_idx}_rep_{rep}",
            group_reps=1,
            modified_idx=(sc_idx - 32, pl.Int8),
            scale_range=(0.5 * np.pi, pl.Float32),
            rep_nr=(rep, pl.UInt32),
        )

    return fixture


def run():
    """run experiment"""
    fixture = get_fixture()
    fixture.run()


if __name__ == "__main__":
    run()
