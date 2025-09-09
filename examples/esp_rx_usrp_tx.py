"""
ESP32 sensing proof of concept
"""

from pathlib import Path
from datetime import timedelta
from ipaddress import IPv4Address

import polars as pl

from sensession import (
    Channel,
    MacAddr,
    Database,
    Bandwidth,
    DataRateMode,
    FrameGenerator,
)
from sensession.config import BaseTransmissionConfig
from sensession.devices import (
    UsrpulseUsrp,
    IQFrameConfig,
    UsrpulseConfig,
    IQFrameGroupConfig,
)
from sensession.tools.espion import ESP32Tool
from sensession.devices.esp32 import ESP32, ESP32Config

if __name__ == "__main__":
    # Create devices
    esp1_cfg = ESP32Config(name="ESP32S3", short_name="ESP", comport="/dev/ttyACM2")
    esp1 = ESP32(esp1_cfg)

    usrp_config = UsrpulseConfig(
        "USRP",
        addr=IPv4Address("192.168.10.3"),
        port=8080,
        ssh_name="sdr",
    )
    usrp = UsrpulseUsrp(usrp_config)

    # Define channel and frame to send
    channel = Channel(
        number=1,
        bandwidth=Bandwidth(20),
    )

    # note: this frame is created only for filtering purposes.
    # everything but the transmitter_address is ignored.
    iq_frame = IQFrameConfig(
        receiver_address=MacAddr("ff:ff:ff:ff:ff:ff"),
        bssid_address=MacAddr("ff:ff:ff:ff:ff:ff"),
        transmitter_address=MacAddr("3c:37:12:97:20:e1"),
        data_rate_mode=DataRateMode.HIGH_THROUGHPUT,
    )
    iq_group = IQFrameGroupConfig(
        base_frame=iq_frame,
        interframe_delay=timedelta(milliseconds=5),
        group_repetitions=1000,
    )

    # --------------------------------------------------------------------------------
    # Set up the tools
    esp_tool = ESP32Tool()
    esp1_id = esp_tool.add_device(esp1)

    # Generate corresponding frame to transmit
    manager = FrameGenerator()
    digest = manager.add_frame(iq_group)
    frame = manager.retrieve_frame(digest)

    # Start capturing
    esp_tool.setup_capture(
        [esp1_id],
        channel,
        cache_dir=Path.cwd() / ".cache" / "ESP32cache",
        filter_frame=iq_frame,
    )

    tx_config = BaseTransmissionConfig(n_reps=1, pause_ms=0, gain=5)

    esp_tool.run()

    # bk: is usrp.transmit blocking?
    usrp.transmit(frame, channel, tx_config)
    esp_tool.stop()

    # Collect
    res = esp_tool.reap()

    with Database("data/ESP32test", append=False) as db:
        for r in res:
            db.add_data(r, session_id=("session_1", pl.String))
