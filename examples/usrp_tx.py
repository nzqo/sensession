"""
Example of running a manual sensing session.

Receiver: Asus router, operated with Nexmon
Transmitter: USRP (raw IQ frame transmission)
"""

from sensession import Channel, MacAddr, Bandwidth, FrameGenerator, SshPasswordless
from sensession.devices import (
    UhdUsrp,
    IQFrameConfig,
    UhdUsrpConfig,
    BaseTransmissionConfig,
)

if __name__ == "__main__":
    usrp_config = UhdUsrpConfig(
        "N2954R",
        serial_num="30F3CEE",
        access=SshPasswordless("sdr"),
    )

    channel = Channel(
        center_freq_hz=2_452_000_000,  # 2_437_000_000,
        number=7,
        bandwidth=Bandwidth(40),
    )

    iq_frame = IQFrameConfig(
        receiver_address=MacAddr("a1:b2:c3:d4:e5:f6"),
        transmitter_address=MacAddr("aa:aa:aa:aa:aa:aa"),
        bssid_address=MacAddr("12:34:56:78:9a:bc"),
        bandwidth=Bandwidth(40),
        send_rate_hz=50_000_000,
    )

    tx_config = BaseTransmissionConfig(n_reps=10, pause_ms=10, gain=25)

    # --------------------------------------------------------------------------------
    # Create frame manager to build frames
    manager = FrameGenerator()
    digest = manager.add_frame(iq_frame)
    frame = manager.retrieve_frame(digest)

    # Create devices
    usrp = UhdUsrp(usrp_config)

    # Start transmitting
    usrp.transmit(frame, channel, tx_config)
