"""
Base experiment

Simply performing warmups - i.e. repetitions of the same unmasked frame
"""
# pylint: disable=duplicate-code

from datetime import datetime

from common import ExperimentFixture

if __name__ == "__main__":
    fixture = ExperimentFixture("base")

    date = datetime.now().strftime("%d-%m-%Y")
    fixture.add_schedule_for_mask(
        mask=None, schedule_name=f"{date}_base_schedule", training_reps=10000
    )
    fixture.run()
