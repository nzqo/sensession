"""
Expanding scale blocks

See how the cards detect CSI when a block of subcarriers around DC
has their magnitude scale by a range of factors
"""
# pylint: disable=duplicate-code

import numpy as np
from common import ExperimentFixture


def alternating_mag_mask() -> np.ndarray:
    """
    Masking alternating amplitudes with 0.5 and 1.5 factor
    """
    base_mask = np.ones((64, 1), dtype=np.complex64)
    base_mask[::2] = 0.5 + 0j
    base_mask[1::2] = 1.5 + 0j
    return base_mask


def alternating_phs_mask() -> np.ndarray:
    """
    Masking every second phase with a phase shift of 0.2 pi
    """
    base_mask = np.ones((64, 1), dtype=np.complex64)
    base_mask[1::2] *= np.exp(1j * 0.2 * np.pi)
    return base_mask


def mag_block_mask() -> np.ndarray:
    """
    Masking a block with doubled amplitude
    """
    base_mask = np.ones((64, 1), dtype=np.complex64)
    base_mask[24:40] = 2 + 0j
    return base_mask


def phase_block_mask() -> np.ndarray:
    """
    Masking a block with 0.4pi phaseshift
    """
    base_mask = np.ones((64, 1), dtype=np.complex64)
    base_mask[24:40] *= np.exp(1j * 0.4 * np.pi)
    return base_mask


def get_fixture() -> ExperimentFixture:
    """
    Configure experiment fixture
    """
    schedules = {
        "alternating_magnitude": alternating_mag_mask(),
        "alternating_phase": alternating_phs_mask(),
        "magnitude_block": mag_block_mask(),
        "phase_block": phase_block_mask(),
    }

    # Masking happens after rescaling to ensure masked and unmasked frames are comparable.
    # To ensure that masking doesn't cause out-of-scale problems, choose a lower rescale
    # factor.
    fixture = ExperimentFixture("ax210_averaging", rescale_factor=5000)

    for schedule_name, mask in schedules.items():
        fixture.add_schedule_for_mask(
            mask=mask,
            schedule_name=schedule_name,
            training_reps=500,
            group_reps=1000,
        )

    return fixture


def run():
    """run experiment"""
    fixture = get_fixture()
    fixture.run()


if __name__ == "__main__":
    run()
