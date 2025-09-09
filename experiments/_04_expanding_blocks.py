"""
Expanding scale blocks

See how the cards detect CSI when a block of subcarriers around DC
has their magnitude scale by a range of factors
"""
# pylint: disable=duplicate-code

import itertools

import numpy as np
import polars as pl
from common import ExperimentFixture


def scaleblock_mask(blocksize: int, scale: float) -> np.ndarray:
    """
    Create mask in the shape of a growing rectangle block around
    the center subcarrier.
    """
    base_mask = np.ones((64, 1), dtype=np.complex64)
    # NOTE: Assumes 20 MHz channels, fft size 64
    base_mask[32 - blocksize // 2 : 32 + blocksize // 2 + 1] *= scale
    return base_mask


def get_fixture() -> ExperimentFixture:
    """
    Configure experiment fixture
    """
    block_sizes = list(range(1, 50, 2))
    factors = [0.0, 0.5, 0.75, 1.25, 2, 3, 4.5]

    # Masking happens after rescaling to ensure masked and unmasked frames are comparable.
    # To ensure that masking doesn't cause out-of-scale problems, choose a lower rescale
    # factor.
    fixture = ExperimentFixture("expanding_blocks", rescale_factor=5000)

    for i, (block_size, scale) in enumerate(itertools.product(block_sizes, factors)):
        fixture.add_schedule_for_mask(
            mask=scaleblock_mask(block_size, scale),
            schedule_name=f"run_{i}",
            group_reps=1000,
            blocksize=(block_size, pl.UInt8),
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
