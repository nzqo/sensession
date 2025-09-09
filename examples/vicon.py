"""
To avoid code duplication, some parts can be shared between experiments,
since we generally use the same devices etc. across them
"""

import time

# pylint: disable=duplicate-code
from pathlib import Path
from datetime import timedelta

import numpy as np
import polars as pl
from rich import traceback
from loguru import logger

from sensession import Channel, MacAddr, Database, Bandwidth, SshPasswordless
from sensession.devices import (
    Mask,
    Viconic,
    SenseiNic,
    ESP32Config,
    PSNICConfig,
    PSUSRPConfig,
    SenseiNexmon,
    IQFrameConfig,
    ViconicConfig,
    Ath9kNICConfig,
    UsrpulseConfig,
    SenseiRemoteConfig,
    ServerConnectionType,
    BaseTransmissionConfig,
    InterleavedIQFrameGroupConfig,
)
from sensession.campaign import (
    Campaign,
    CampaignRunner,
    ScheduleBuilder,
    extract_failed_campaign,
)
from sensession.util.frame_generation import FrameGenerator

traceback.install(show_locals=False)


def get_usrp() -> UsrpulseConfig:
    """
    Get config for the USRP transmitter
    """
    return UsrpulseConfig(
        name="USRP x310",
        addr="192.168.20.125",
        port=8080,
        ssh_name="sdr",
    )


def get_ps_usrp() -> PSUSRPConfig:
    """
    Get config for USRP operated with PicoScenes
    """
    return PSUSRPConfig(
        name="Usrp X310",
        short_name="x310",
        serial_num="30F3CEE",
        rf_port_num=0,
        antenna_idxs=[0],
        stream_idxs=[0],
        rx_gain=0.55,
    )


def get_iwl5300() -> SenseiRemoteConfig:
    """
    Get device config of iwl5300 nic
    """
    return SenseiRemoteConfig(
        short_name="iwl",
        name="Intel wireless link 5300",
        remote_resource_id="iwl",
        addr="192.168.20.124",
        port=8080,
        connection_type=ServerConnectionType.WEB,
        antenna_idxs=[0],
        stream_idxs=[0],
        source_cfg=SenseiNic(
            interface="wlp1s0",
            scale_csi=True,
        ),
    )


def get_asus_1() -> SenseiRemoteConfig:
    """
    Get config for first nexmon router
    """

    return SenseiRemoteConfig(
        short_name="asus1",
        name="Asus RT-ac86u #1",
        remote_resource_id="nexmon",
        addr="192.168.20.31",
        port=5501,
        connection_type=ServerConnectionType.TCP,
        antenna_idxs=[0],
        stream_idxs=[0],
        source_cfg=SenseiNexmon(),
    )


def get_asus_2() -> SenseiRemoteConfig:
    """
    Get config for second nexmon router
    """
    return SenseiRemoteConfig(
        short_name="asus2",
        name="Asus RT-ac86u #2",
        remote_resource_id="nexmon",
        addr="192.168.20.32",
        port=5501,
        connection_type=ServerConnectionType.TCP,
        antenna_idxs=[0],
        stream_idxs=[0],
        source_cfg=SenseiNexmon(),
    )


def get_ax210() -> PSNICConfig:
    """
    Get device config of ax210 nic
    """
    return PSNICConfig(
        name="Intel ax210",
        short_name="ax210",
        interface="wlp2s0",
        mac_address="b0:dc:ef:b4:cb:6b",
        antenna_idxs=[0],
        stream_idxs=[0],
        phy_path=2,
    )


def get_qca1() -> Ath9kNICConfig:
    """
    Get device config of ath9k nic
    """
    return Ath9kNICConfig(
        name="Qualcomm Atheros AR9462",
        short_name="qca",
        interface="wlp3s0",
        mac_address="70:77:81:69:51:bf",
        antenna_idxs=[0],
        stream_idxs=[0],
        repo_path="~/Development/csi-modules",
        access=SshPasswordless(remote_ssh_hostname="shuttle"),
    )


def get_esp32_1() -> ESP32Config:
    """
    Get device config of ESP32
    """
    return ESP32Config(
        name="ESP32_S3 DevKitC-1U", short_name="ESP1", comport="/dev/ttyACM0"
    )


def get_esp32_2() -> ESP32Config:
    """
    Get device config of ESP32
    """
    return ESP32Config(
        name="ESP32_S3 DevKitC-1U", short_name="ESP2", comport="/dev/ttyACM1"
    )


def get_2ghz_channel() -> Channel:
    """
    Get default 2.4 GHz Channel
    """
    return Channel(
        number=11,
        bandwidth=Bandwidth(20),
    )


def get_iwl_frame(rescale_factor: int = 25000) -> IQFrameConfig:
    """
    Get frame configuration for receival with iwl5300
    """
    return IQFrameConfig(
        receiver_address=MacAddr("00:16:ea:12:34:56"),
        transmitter_address=MacAddr("00:16:ea:12:34:56"),
        bssid_address=MacAddr("ff:ff:ff:ff:ff:ff"),
        bandwidth=Bandwidth(20),
        send_rate_hz=20_000_000,
        rescale_factor=rescale_factor,
    )


def get_qca_frame(rescale_factor: int = 25000) -> IQFrameConfig:
    """
    Get frame configuration for receival with qca
    """
    return IQFrameConfig(
        receiver_address=MacAddr("70:77:81:69:51:bf"),
        transmitter_address=MacAddr("24:4b:fe:bc:a6:fc"),
        bssid_address=MacAddr("24:4b:fe:bc:a6:fc"),
        enable_sounding=True,
        bandwidth=Bandwidth(20),
        send_rate_hz=20_000_000,
        rescale_factor=rescale_factor,
    )


def get_interleaved_frame_group(
    base_frames: list[IQFrameConfig],
    mask: np.ndarray,
    group_reps: int = 1000,
    if_delay: timedelta = timedelta(milliseconds=5),
) -> InterleavedIQFrameGroupConfig:
    """
    Create a masked frame group
    """
    return InterleavedIQFrameGroupConfig(
        base_frames=base_frames,
        interframe_delay=if_delay,
        group_repetitions=group_reps,
        mask=Mask(mask),
    )


def get_exp_tx_config(gain: int, start: int, n_reps=1) -> BaseTransmissionConfig:
    """
    Create an example experiment config.
    """
    return BaseTransmissionConfig(gain=gain, start_at=start, n_reps=n_reps)


def generate_frame(frame: InterleavedIQFrameGroupConfig):
    """
    Generate frame
    """
    # ensure frame is generated
    manager = FrameGenerator()
    digest = manager.add_frame(frame)
    frame = manager.retrieve_frame(digest)


if __name__ == "__main__":
    receiver = [
        get_qca1(),
        get_asus_1(),
        get_asus_2(),
        get_ax210(),
        get_ps_usrp(),
        get_esp32_1(),
        get_esp32_2(),
        get_iwl5300(),
    ]

    vicon_cfg = ViconicConfig(short_name="vicon", addr="192.168.20.11", port=8000)
    vicon = Viconic(vicon_cfg)

    transmitter = get_usrp()
    transmitter_id = transmitter.device_id()

    iwl_group = [receiver.device_id() for receiver in receiver[1:]]
    qca_group = [receiver[0].device_id()]

    # Settings:
    # 0: length of room (across long side, lets say x)
    # 1: Approximately continuing (across long side, LOS)
    # 2. Continuing after 1

    # 10: width of room, starting lower point (entrance)
    # 11: continuation of 10
    # 12: continuation, approximately from middle of room
    # 13: continuation

    # 20: Starting in lower edge in cantor snake
    # 21: Continuation, ending approximately in middle of room
    # 22: Continuation
    # 23: Ending approximately upper right corner

    # 3x: Jogging lol

    # 40: Random slow movements.
    SETTING = 40
    RESCALE_FACTOR = 28_000
    IF_DISTANCE = 2  # twice the distance below because of interleaving
    SECONDS = 30
    FRAME_REPS = 3000
    FPS = 1000 // IF_DISTANCE  # should be 500
    REPS = FPS * SECONDS // FRAME_REPS  # Number of times to repeat the 3k frames
    CAPTURE_NAME = f"tracker_capture_{SETTING}"

    # Get channel
    channel = get_2ghz_channel()

    # Generate frame
    qca_base_frame = get_qca_frame(RESCALE_FACTOR)
    iwl_base_frame = get_iwl_frame(RESCALE_FACTOR)
    base_mask = np.ones((int(channel.bandwidth.in_mhz() * 3.2), 1), dtype=np.complex64)
    tx_frame = get_interleaved_frame_group(
        [iwl_base_frame, qca_base_frame],
        base_mask,
        FRAME_REPS,
        timedelta(milliseconds=IF_DISTANCE // 2),
    )
    generate_frame(tx_frame)
    FRAME_ID = tx_frame.frame_id()

    # Repeat for 40 seconds. Start in 20 seconds.
    start_at = int(time.time()) + 20
    logger.info(f"Starting at: {start_at}")
    tx_config = get_exp_tx_config(gain=30, n_reps=REPS, start=start_at)

    builder = ScheduleBuilder(
        CAPTURE_NAME,
        channel=channel,
        cache_dir=Path.cwd() / ".cache" / CAPTURE_NAME,
    )
    # Add base session
    builder = (
        builder.prepare_collect_csi(
            device_ids=iwl_group,
            collection_name=CAPTURE_NAME,
            filter_frame=iwl_base_frame,
        )
        .prepare_collect_csi(
            device_ids=qca_group,
            collection_name=CAPTURE_NAME,
            filter_frame=qca_base_frame,
        )
        .prepare_transmit(transmitter_id, FRAME_ID, tx_config)
        .run(min_capture_num=int(FRAME_REPS * REPS * 0.5))
    )

    schedule = builder.build()
    schedule.extra_labels = {
        "start_epoch": (start_at, pl.UInt64),
        "num_frames": (FRAME_REPS * REPS, pl.UInt32),
        "frame_spacing_ms": (IF_DISTANCE, pl.UInt8),
        "max_sequence": (FRAME_REPS - 1, pl.UInt16),
        "setting": (SETTING, pl.UInt16),
    }

    campaign = Campaign(
        schedules=[schedule],
        frames=[tx_frame],
        device_cfgs=receiver + [transmitter],
        name=CAPTURE_NAME,
    )

    # Campaign may have failing sessions, so we iterate until all are successful
    with Database(f"data/{CAPTURE_NAME}", append=True) as db:
        # Run the current campaign
        with CampaignRunner(campaign) as runner:
            vicon.start_at(seconds=start_at, capture_id=CAPTURE_NAME)
            vicon.stop_at(seconds=start_at + SECONDS)
            result = runner.run()

            # Store results and advance with new campaign containing only failed sessions
            runner.store_results(db)
            campaign = extract_failed_campaign(campaign, result)

    if not campaign.is_empty():
        logger.error("Campaign failed")
