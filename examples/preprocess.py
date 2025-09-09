"""
Example of processing CSI data
"""

import polars as pl

from sensession.database import Database
from sensession.processing import CsiProcessor

if __name__ == "__main__":
    with Database("data/base_campaign") as db:
        proc = CsiProcessor(db.csi, db.meta)
        proc = (
            proc.load_checkpoint("test_check")
            or proc.correct_rssi_by_agc()
            .unwrap()
            .scale_magnitude()
            .rescale_csi_by_rssi()
            .detrend_phase()
            .save_checkpoint("test_check")
        )
        with pl.Config(tbl_cols=-1):
            print(proc.csi)
