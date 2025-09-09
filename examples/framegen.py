"""
Example of running a manual sensing session.

Receiver: Asus router, operated with Nexmon
Transmitter: USRP (raw IQ frame transmission)
"""

from datetime import timedelta

from sensession import MacAddr, Bandwidth, FrameGenerator
from sensession.config import DataRateMode
from sensession.devices import IQFrameConfig, IQFrameGroupConfig

if __name__ == "__main__":
    # Some random arbitrary frame (asus can capture everything)
    iq_frame = IQFrameConfig(
        receiver_address=MacAddr("00:16:ea:12:34:56"),
        transmitter_address=MacAddr("00:16:ea:12:34:56"),
        bssid_address=MacAddr("00:16:ea:12:34:56"),
        bandwidth=Bandwidth(80),
        data_rate_mode=DataRateMode.VERY_HIGH_THROUGHPUT,
        send_rate_hz=100_000_000,
    )

    iq_group = IQFrameGroupConfig(
        base_frame=iq_frame,
        interframe_delay=timedelta(milliseconds=5),
        group_repetitions=100,
    )

    # --------------------------------------------------------------------------------
    # Create frame manager to build frames
    manager = FrameGenerator()
    digest = manager.add_frame(iq_group)
    frame = manager.retrieve_frame(digest)
    print(f"digest: {digest}")
