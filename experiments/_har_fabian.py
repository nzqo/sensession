"""
Base experiment

Simply performing warmups - i.e. repetitions of the same unmasked frame
"""
# pylint: disable=duplicate-code

from datetime import timedelta

import polars as pl
from common import ExperimentFixture

if __name__ == "__main__":
    fixture = ExperimentFixture(
        "har_fabian",
        gain=25,
        if_delay=timedelta(milliseconds=1),
    )

    # Look in table
    ACTIVITY_IDX = 20
    POSITION_IDX = 0

    # repeat 10 times
    for i in range(0, 10):
        fixture.add_schedule_for_mask(
            mask=None,
            schedule_name="activity_recognition",
            training_reps=1500,
            activity_idx=(ACTIVITY_IDX, pl.UInt8),
            human_label=("Fabian", pl.String),
            position=(POSITION_IDX, pl.UInt8),
            rep_nr=(i, pl.UInt8),
        )

    fixture.run()
