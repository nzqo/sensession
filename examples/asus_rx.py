"""
Example of running a manual sensing session.

Receiver: Asus router, operated with Nexmon
Transmitter: USRP (raw IQ frame transmission)
"""

from pathlib import Path
from datetime import timedelta
from ipaddress import IPv4Address

import numpy as np
import polars as pl

from sensession import (
    Channel,
    MacAddr,
    Database,
    Bandwidth,
    DataRateMode,
    FrameGenerator,
    SshPasswordless,
)
from sensession.tools import Nexmon
from sensession.devices import (
    Mask,
    NexmonRouter,
    UsrpulseUsrp,
    IQFrameConfig,
    UsrpulseConfig,
    IQFrameGroupConfig,
    NexmonRouterConfig,
    BaseTransmissionConfig,
)

if __name__ == "__main__":
    # Asus router used for receival
    router_cfg = NexmonRouterConfig(
        access_cfg=SshPasswordless("asus1"),
        name="Asus RT-AC86U BM3000",
        short_name="asus1",
        interface="eno1",
        mac_address=MacAddr("24:4b:fe:be:e6:28"),
        antenna_idxs=[0],
        stream_idxs=[0],
        host_ip=IPv4Address("192.168.20.30"),
        netcat_port=5502,
    )

    # USRP used for transmission
    usrp_config = UsrpulseConfig(
        "USRP-x310",
        addr=IPv4Address("192.168.20.125"),
        port=8080,
        ssh_name="sdr",
    )

    channel = Channel(
        center_freq_hz=5_795_000_000,  # 2_452_000_000,  # 2_437_000_000,
        number=157,  # 7,
        bandwidth=Bandwidth(20),
    )

    # Some random arbitrary frame (asus can capture everything)
    iq_frame = IQFrameConfig(
        receiver_address=MacAddr("aa:bb:cc:dd:ee:ff"),
        transmitter_address=MacAddr("aa:aa:aa:aa:aa:aa"),
        bssid_address=MacAddr("11:11:11:11:11:11"),
        data_rate_mode=DataRateMode.HIGH_THROUGHPUT,
        bandwidth=Bandwidth(20),  # Channel bandwidth
        send_rate_hz=20_000_000,
    )
    iq_group = IQFrameGroupConfig(
        base_frame=iq_frame,
        interframe_delay=timedelta(milliseconds=5),
        group_repetitions=1000,
        mask=Mask(np.ones((64, 1), dtype=np.complex64)),
    )

    tx_config = BaseTransmissionConfig(n_reps=10, pause_ms=10, gain=25)

    # --------------------------------------------------------------------------------
    # Create frame manager to build frames
    manager = FrameGenerator()
    digest = manager.add_frame(iq_group)
    frame = manager.retrieve_frame(digest)

    # Create devices
    router = NexmonRouter(router_cfg)
    nexmon = Nexmon()
    router_id = nexmon.add_device(router)
    usrp = UsrpulseUsrp(usrp_config)

    # Start capturing with Nexmon
    nexmon.setup_capture(
        router_id,
        channel,
        cache_dir=Path.cwd() / ".cache" / "asus_rx_test",
        filter_frame=iq_frame,
    )

    nexmon.run()

    # Start transmitting
    usrp.prepare_frame(frame)
    usrp.transmit(frame, channel, tx_config)

    # Reap what you sow!
    nexmon.stop()
    res = nexmon.reap()

    with Database("data/asus_rx_test", append=False) as db:
        for r in res:
            db.add_data(r, session_id=("session_1", pl.String))
