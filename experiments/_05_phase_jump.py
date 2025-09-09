"""
Phase jump experiments

Phases can be detected relative to another, which means that we need to
introduce a step function at the place where we want to see a difference
between neighbors
"""
# pylint: disable=duplicate-code

import itertools

import numpy as np
import polars as pl
from common import ExperimentFixture


def phasejump_mask(sc_idx: int, scale: float) -> np.ndarray:
    """
    Phase jump at sc_idx, i.e. step function from there
    """
    base_mask = np.ones((64, 1), dtype=np.complex64)
    base_mask[sc_idx:] *= np.exp(1j * scale * np.pi)
    return base_mask


def get_fixture() -> ExperimentFixture:
    """
    Configure experiment fixture
    """
    idxs = list(range(64))
    factors = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]

    # Masking happens after rescaling to ensure masked and unmasked frames are comparable.
    # To ensure that masking doesn't cause out-of-scale problems, choose a lower rescale
    # factor.
    fixture = ExperimentFixture("phase_jumps")

    for i, (sc_idx, scale) in enumerate(itertools.product(idxs, factors)):
        fixture.add_schedule_for_mask(
            mask=phasejump_mask(sc_idx, scale),
            schedule_name=f"run_{i}",
            group_reps=1000,
            modified_idx=(sc_idx - 32, pl.Int8),
            scale_factor=(scale, pl.Float32),
            session_nr=(i, pl.UInt32),
        )

    return fixture


def run():
    """run experiment"""
    fixture = get_fixture()
    fixture.run()


if __name__ == "__main__":
    run()
