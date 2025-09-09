"""
Rectangles experiments

Test how cards detect rectangles, i.e. choosing a set of neighboring
subcarriers and rescaling their CSI amplitudes by a range of factors.
"""
# pylint: disable=duplicate-code

import itertools

import numpy as np
import polars as pl
from common import ExperimentFixture, interesting_subcarriers


def rectangle_mask(sc_idx: int, scale: float) -> np.ndarray:
    """
    Get a rectangle mask; affecting the main subcarrier at sc_idx and its
    three left and right neighbors on each side.
    """
    base_mask = np.ones((64, 1), dtype=np.complex64)

    # Rectangular precoding for a total of 7 subcarriers
    for k in range(-3, 4):
        base_mask[sc_idx + k] *= scale

    return base_mask


def get_fixture() -> ExperimentFixture:
    """
    Configure experiment fixture
    """
    idxs = interesting_subcarriers()
    factors = [0.0, 0.5, 0.75, 1.25, 2, 3, 4.5]

    # Masking happens after rescaling to ensure masked and unmasked frames are comparable.
    # To ensure that masking doesn't cause out-of-scale problems, choose a lower rescale
    # factor.
    fixture = ExperimentFixture("rectangles", rescale_factor=12000)

    for i, (sc_idx, scale) in enumerate(itertools.product(idxs, factors)):
        fixture.add_schedule_for_mask(
            mask=rectangle_mask(sc_idx, scale),
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
