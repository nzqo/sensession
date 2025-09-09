"""
Example of running a manual sensing session.

Receiver: Asus router, operated with Nexmon
Transmitter: USRP (raw IQ frame transmission)
"""

import sys
from pathlib import Path
from datetime import timedelta
from ipaddress import IPv4Address

import polars as pl
import sensei
from loguru import logger

from sensession import Channel, MacAddr, Database, Bandwidth, FrameGenerator
from sensession.tools import Sensei
from sensession.config import DataRateMode
from sensession.devices import (
    SenseiNic,
    SenseiDevice,
    SenseiNexmon,
    UsrpulseUsrp,
    IQFrameConfig,
    UsrpulseConfig,
    IQFrameGroupConfig,
    SenseiRemoteConfig,
    ServerConnectionType,
    BaseTransmissionConfig,
)

if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")

    nic_config = SenseiRemoteConfig(
        short_name="iwl",
        name="Intel Wireless Link 5300",
        remote_resource_id="iwl",
        addr="192.168.20.124",
        port=8080,
        connection_type=ServerConnectionType.WEB,
        antenna_idxs=[0],
        stream_idxs=[0],
        source_cfg=SenseiNic(interface="wlp1s0", scale_csi=True),
    )

    asus1_config = SenseiRemoteConfig(
        short_name="asus1",
        name="Asus RT-AC86U",
        remote_resource_id="nexmon",
        addr="192.168.10.31",
        port=5501,
        connection_type=ServerConnectionType.TCP,
        antenna_idxs=[0, 1, 2, 3],
        stream_idxs=[0],
        source_cfg=SenseiNexmon(),
    )

    usrp_config = UsrpulseConfig(
        "USRP x310",
        addr=IPv4Address("192.168.10.3"),
        port=8080,
        ssh_name="sdr",
    )

    # Channel used for tranmissions
    channel = Channel(
        number=157,
        bandwidth=Bandwidth(20),
    )

    # Some random arbitrary frame (asus can capture everything)
    iq_frame = IQFrameConfig(
        receiver_address=MacAddr("00:16:ea:12:34:56"),
        transmitter_address=MacAddr("00:16:ea:12:34:56"),
        bssid_address=MacAddr("11:11:11:11:11:11"),
        bandwidth=Bandwidth(20),
        send_rate_hz=20_000_000,
        data_rate_mode=DataRateMode.VERY_HIGH_THROUGHPUT,
    )

    # Basic iq frame config
    iq_group = IQFrameGroupConfig(
        base_frame=iq_frame,
        interframe_delay=timedelta(milliseconds=1),
        group_repetitions=1000,
    )

    tx_config = BaseTransmissionConfig(n_reps=1, pause_ms=10, gain=5)

    # --------------------------------------------------------------------------------
    # Create frame manager to build frames
    manager = FrameGenerator()
    digest = manager.add_frame(iq_group)
    frame = manager.retrieve_frame(digest)

    # Okaerinasai, master
    sensei = Sensei()

    # Create devices
    router = SenseiDevice(asus1_config)
    router_id = sensei.add_device(router)

    nic = SenseiDevice(nic_config)
    nic_id = sensei.add_device(nic)

    usrp = UsrpulseUsrp(usrp_config)
    misses = []
    res = []
    # Start capturing with Nexmon
    sensei.setup_capture(
        [router_id, nic_id],
        channel,
        cache_dir=Path.cwd() / ".cache" / "sensei_test",
        filter_frame=iq_frame,
    )

    sensei.run()

    # Start transmitting
    usrp.prepare_frame(frame)
    usrp.transmit(frame, channel, tx_config)

    # Reap what you sow!
    sensei.stop()
    new_res = sensei.reap()
    res.extend(new_res)

    print(f"Num successful devices: {len(new_res)}")
    for r in new_res:
        if len(r.csi) < 800:
            misses.append(r.meta.receiver_name)
        else:
            print(f"Num captured CSI: {len(r.csi)}")

    if len(res) > 20:
        with Database("data/sensei_rx_test", append=True) as db:
            for r in res:
                db.add_data(r, session_id=("session_1", pl.String))
        res.clear()

    print(f"Misses: {misses}")
