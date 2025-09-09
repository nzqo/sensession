"""
Capture with picoscenes devices example

In this example, we capture ambient WiFi frames from whatever communication
is happening in the environment.

We do so using two devices:
    Intel Wi-Fi 6 ax200 NIC
    USRP X310 SDR

The target is Channel 1 (center frequency 2412 MHz and bandwidth of 20 MHz).

- USRP is configured to capture on TX/RX antenna port
- both receivers are detecting with only their first RF channel (antenna)
- With one antenna we are also limited with capturing on one stream
"""

import time
from pathlib import Path

from sensession import Channel, Database, Bandwidth
from sensession.tools import PicoScenes
from sensession.devices import PSNIC, PSUSRP, PSNICConfig, PSUSRPConfig


def run():
    """
    RX example run
    """
    # e4:0d:36:1f:19:f9 is a NIC-specific MAC address
    # We get the MAC address of a NIC by using the command `array_status`
    nic_cfg = PSNICConfig(
        name="Intel ax200",
        short_name="ax200",
        interface="wlp11s0",
        mac_address="e4:0d:36:1f:19:f9",
        antenna_idxs=[0],
        stream_idxs=[0],
        phy_path=4,
    )

    # See `uhd_find_devices` to find e.g. the serial number
    ps_usrp_cfg = PSUSRPConfig(
        name="Usrp X310 #1",
        short_name="x310",
        serial_num="30F1626",
        rf_port_num=0,
        antenna_idxs=[0],
        stream_idxs=[0],
        rx_gain=0.45,
    )

    channel = Channel(
        number=1,
        bandwidth=Bandwidth(20),
    )

    # --------------------------------------------------------------------------------
    # Create NIC, receiver USRP and transmitter USRP
    nic = PSNIC(nic_cfg)
    ps_usrp = PSUSRP(ps_usrp_cfg)

    # Set up capture tools
    ps_tool = PicoScenes()
    nic_id = ps_tool.add_device(nic)
    ps_usrp_id = ps_tool.add_device(ps_usrp)

    # Start capturing with PicoScenes
    # NOTE: filter_frame argument could be used to filter for frames with specific
    # MAC addresses. Here, we capture all traffic on the channel.
    ps_tool.setup_capture(
        [ps_usrp_id, nic_id],
        channel,
        cache_dir=Path.cwd() / ".test",
    )

    # Run!
    ps_tool.run()

    # Capture for 10 seconds
    time.sleep(10)

    # Reap what you sow!
    ps_tool.stop()
    result = ps_tool.reap()

    with Database("data/picoscenes_test", append=False) as db:
        for r in result:
            print(f"Receiver {r.meta.receiver_name} captured: {len(r.data.csi_vals)}")
            db.add_data(r)


if __name__ == "__main__":
    run()
