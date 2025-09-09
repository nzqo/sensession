"""
Triangles experiments

Test how cards detect triangles, i.e. choosing a set of neighboring
subcarriers and rescaling their CSI amplitudes by a range of factors
in triangular shape
"""
# pylint: disable=duplicate-code

import itertools

import numpy as np
import polars as pl
from common import ExperimentFixture, interesting_subcarriers


def triangle_mask(sc_idx: int, scale: float) -> np.ndarray:
    """
    Get a triangle shaped mask affecting the subcarrier at sc_idx
    and its left-/right neighbors.
    """
    base_mask = np.ones((64, 1), dtype=np.complex64)
    base_mask[sc_idx] *= scale
    base_mask[sc_idx - 1] *= (scale + 1) * 0.5
    base_mask[sc_idx + 1] *= (scale + 1) * 0.5
    return base_mask


def get_fixture() -> ExperimentFixture:
    """
    Configure experiment fixture
    """
    # Define parameters to construct precoding masks and construct sessions
    idxs = interesting_subcarriers()
    idxs = list(range(1, 63))
    factors = [0.0, 0.5, 0.75, 1.25, 2, 3, 4.5]

    fixture = ExperimentFixture("triangles")

    for i, (sc_idx, scale) in enumerate(itertools.product(idxs, factors)):
        fixture.add_schedule_for_mask(
            mask=triangle_mask(sc_idx, scale),
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
