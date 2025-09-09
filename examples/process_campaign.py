"""
Example of processing CSI data
"""

import polars as pl

from sensession.database import Database
from sensession.campaign.processor import CampaignProcessor

if __name__ == "__main__":
    # Data was created with the campaign.py example
    with Database("data/test_campaign") as db:
        with pl.Config(tbl_cols=-1):
            print(db.csi)

        proc = (
            CampaignProcessor(db.csi, db.meta)
            .correct_rssi_by_agc()
            .unwrap()
            .filter("antenna_idxs", 0)
            .remove_edge_subcarriers()
            .scale_magnitude()
            .rescale_csi_by_rssi()
            .detrend_phase()
            .equalize_magnitude()
            .equalize_phase()
            .remove_phase_outliers()
            .align_sequence_nums()
            .drop_contains()
            .discard_low_counts(min_count=100)
            .fill_missing_sequence_numbers(100)
            .lin_interpolate_nulls()
        )
        with pl.Config(tbl_cols=-1):
            print(proc.csi)
