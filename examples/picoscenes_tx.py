"""
Transmit and capture with picoscenes devices example
In this example, we operate Picoscenes in order to transmit frames with an
Intel NIC and receive corresponding CSI on a USRP SDR.

The devices' models are:

- Intel Wi-Fi 6 ax200 NIC
- USRP X310 SDR

We operate on Channel 1 (center frequency 2412 MHz and bandwidth of 20 MHz).

- The transmitter (NIC) sends packets at fixed time intervals for fixed repetitions
- USRP is configured to capture on TX/RX antenna port
- USRP uses both RF channels to capture (i.e. antennas 1 and 2)
- Data is collected from two streams (using the two antennas on the NIC)
"""

import time
from pathlib import Path

from sensession import Channel, MacAddr, Database, Bandwidth
from sensession.tools import PicoScenes
from sensession.config import DataRateMode, BaseFrameConfig
from sensession.devices import PSNIC, PSUSRP, PSNICConfig, PSUSRPConfig
from sensession.tools.picoscenes import PicoscenesTransmissionConfig


def run():
    """
    Transmission example
    """
    # e4:0d:36:1f:19:f9 is a NIC-specific MAC address
    # We get the MAC address of a NIC by using the command `array_status`
    nic_cfg = PSNICConfig(
        name="Intel ax200",
        short_name="ax200",
        interface="wlp5s0",
        mac_address="e4:0d:36:1f:19:f9",
        antenna_idxs=[0, 1],
        stream_idxs=[0, 1],
        phy_path=3,
    )

    usrp_cfg = PSUSRPConfig(
        name="Usrp X310 #1",
        short_name="x310",
        serial_num="30F3CF1",
        rf_port_num=0,
        antenna_idxs=[0, 1],
        stream_idxs=[0, 1],
    )

    channel = Channel(
        number=1,
        bandwidth=Bandwidth(20),
    )

    # MAC information:
    # RX: PicoScenes uses a magic default 00:16:ea:12:34:56 for the injection target.
    #     This can be modified however, and we do so here.
    # TX: We get the transmitter address (NIC MAC) by using `array_status`
    frame = BaseFrameConfig(
        receiver_address=MacAddr("00:16:ea:99:77:88"),
        transmitter_address=MacAddr("e4:0d:36:1f:19:f9"),
        bandwidth=Bandwidth(20),
        data_rate_mode=DataRateMode("HT"),
    )

    # NOTE: The total time this transmission takes is a bit longer, since frames
    # themselves have a duration and we must consider setup/shutdown times.
    tx_config = PicoscenesTransmissionConfig(
        n_reps=1000,
        pause_ms=10,  # 1000*10 millisecond => total time : 10 seconds of breaks
        tx_startdelay_s=3,
        gain=25,  # not adjustable for NIC
    )

    # --------------------------------------------------------------------------------
    # Create a nic
    # Create a usrp
    nic = PSNIC(nic_cfg)
    usrp = PSUSRP(usrp_cfg)
    ps_tool = PicoScenes(frames=[frame])
    nic_id = ps_tool.add_device(nic)
    usrp_id = ps_tool.add_device(usrp)

    # Construct PicoScenes Command
    ps_tool.setup_capture(
        usrp_id,
        channel,
        cache_dir=Path.cwd() / ".test",
        filter_frame=frame,
    )
    ps_tool.setup_transmit(
        nic_id, frame_id=frame.frame_id(), channel=channel, tx_config=tx_config
    )

    # Start transmitting and receiving with PicoScenes
    ps_tool.run()

    print("Running .")
    while ps_tool.is_running():
        print(".")
        time.sleep(1)

    # Stop and reap.
    ps_tool.stop()
    result = ps_tool.reap()

    with Database("data/picoscenes_test", append=False) as db:
        for res in result:
            print(f"Device {res.meta.receiver_name} captured: {len(res.data.csi_vals)}")
            db.add_data(res)


if __name__ == "__main__":
    run()
