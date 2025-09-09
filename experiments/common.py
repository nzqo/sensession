"""
To avoid code duplication, some parts can be shared between experiments,
since we generally use the same devices etc. across them
"""

from pathlib import Path
from datetime import timedelta

# pylint: disable=duplicate-code
from ipaddress import IPv4Address

import rich
import numpy as np
from rich import traceback
from loguru import logger

from sensession import Channel, MacAddr, Database, Bandwidth, SshPasswordless
from sensession.config import DataRateMode
from sensession.devices import (
    Mask,
    ESP32Config,
    PSNICConfig,
    PSUSRPConfig,
    SenseiNexmon,
    IQFrameConfig,
    Ath9kNICConfig,
    UsrpulseConfig,
    IQFrameGroupConfig,
    SenseiRemoteConfig,
    ServerConnectionType,
    BaseTransmissionConfig,
    InterleavedIQFrameGroupConfig,
)
from sensession.campaign import (
    Campaign,
    Schedule,
    CampaignRunner,
    ScheduleBuilder,
    SerializationMode,
    write_to_disk,
    extract_failed_campaign,
)

traceback.install(show_locals=False)


def get_usrp() -> UsrpulseConfig:
    """
    Get config for the USRP transmitter
    """
    return UsrpulseConfig(
        name="USRP x310",
        addr=IPv4Address("192.168.10.3"),
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
        serial_num="30F3CE3",
        rf_port_num=0,
        antenna_idxs=[0, 1],
        stream_idxs=[0],
        rx_gain=0.55,
    )


def get_iwl5300() -> PSNICConfig:
    """
    Get device config of iwl5300 nic
    """
    return PSNICConfig(
        name="Intel iwl5300",
        short_name="iwl5300",
        interface="wlp6s0",
        mac_address=MacAddr("00:21:6a:0b:58:9e"),
        antenna_idxs=[0, 1, 2],
        stream_idxs=[0],
        phy_path=23,
    )


def get_ax210() -> PSNICConfig:
    """
    Get device config of ax210 nic
    """
    return PSNICConfig(
        name="Intel ax210",
        short_name="ax210",
        interface="wlp4s0",
        mac_address=MacAddr("70:1a:b8:97:17:5b"),
        antenna_idxs=[0, 1],
        stream_idxs=[0],
        phy_path=21,
    )


def get_asus_1() -> SenseiRemoteConfig:
    """
    Get config for first nexmon router
    """

    return SenseiRemoteConfig(
        short_name="asus1",
        name="Asus RT-ac86u #1",
        remote_resource_id="nexmon",
        addr=IPv4Address("192.168.10.31"),
        port=5501,
        connection_type=ServerConnectionType.TCP,
        antenna_idxs=[0, 1, 2, 3],
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
        addr=IPv4Address("192.168.10.32"),
        port=5501,
        connection_type=ServerConnectionType.TCP,
        antenna_idxs=[0, 1, 2, 3],
        stream_idxs=[0],
        source_cfg=SenseiNexmon(),
    )


def get_qca1() -> Ath9kNICConfig:
    """
    Get device config of ath9k nic
    """
    return Ath9kNICConfig(
        name="Qualcomm Atheros AR9462",
        short_name="qca",
        interface="wlp1s0",
        mac_address=MacAddr("90:48:9a:b6:1f:39"),
        antenna_idxs=[0, 1],
        stream_idxs=[0],
        repo_path="~/Development/csi-modules",
        access=SshPasswordless(remote_ssh_hostname="qca"),
    )


def get_esp32_1() -> ESP32Config:
    """
    Get device config of ESP32
    """
    return ESP32Config(
        name="ESP32_S3 DevKitC-1U",
        short_name="ESP1",
        comport="/dev/ttyACM0",
    )


def get_esp32_2() -> ESP32Config:
    """
    Get device config of ESP32
    """
    return ESP32Config(
        name="ESP32_S3 DevKitC-1U",
        short_name="ESP2",
        comport="/dev/ttyACM1",
    )


def get_2ghz_channel() -> Channel:
    """
    Get default 2.4 GHz Channel
    """
    return Channel(
        number=1,
        bandwidth=Bandwidth(20),
    )


def get_5ghz_channel() -> Channel:
    """
    Get default 5 GHz Channel
    """
    return Channel(
        number=157,
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
        receiver_address=MacAddr("90:48:9a:b6:1f:39"),
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


def get_frame_group(
    base_frame: IQFrameConfig,
    mask: np.ndarray,
    group_reps: int = 1000,
    if_delay: timedelta = timedelta(milliseconds=5),
) -> IQFrameGroupConfig:
    """
    Get a frame group made by (masked) repetition of a single frame
    """
    return IQFrameGroupConfig(
        base_frame=base_frame,
        group_repetitions=group_reps,
        interframe_delay=if_delay,
        mask=Mask(mask),
    )


def get_exp_tx_config(gain: int) -> BaseTransmissionConfig:
    """
    Create an example experiment config.
    """
    return BaseTransmissionConfig(gain=gain)


class ExperimentFixture:
    """
    Experiment fixture to simplify experiment definition / DRY
    """

    def __init__(
        self,
        campaign_name: str,
        rescale_factor: int = 25000,
        gain: int = 11,
        if_delay: timedelta = timedelta(milliseconds=1),
    ):
        self.num_retries = 10
        self.campaign_name = campaign_name
        self.receiver = [
            get_qca1(),
            get_asus_1(),
            get_asus_2(),
            get_ax210(),
            get_ps_usrp(),
            get_esp32_1(),
            get_esp32_2(),
            get_iwl5300(),
        ]

        self.iwl_group = [receiver.device_id() for receiver in self.receiver[1:]]
        self.qca_group = [self.receiver[0].device_id()]

        self.transmitter = get_usrp()
        self.transmitter_id = self.transmitter.device_id()

        self.channel = get_2ghz_channel()

        self.qca_base_frame = get_qca_frame(rescale_factor)
        self.iwl_base_frame = get_iwl_frame(rescale_factor)

        self.schedules: list[Schedule] = []
        self.frames = [self.qca_base_frame, self.iwl_base_frame]

        self.gain = gain
        self.if_delay = if_delay
        rich.traceback.install(show_locals=False)

    def add_schedule_for_mask(
        self,
        mask: np.ndarray | None,
        schedule_name: str,
        training_reps: int = 500,
        group_reps: int = 1000,
        **extra_labels,
    ):
        """
        Add schedules to capture with qca and iwl, using warmup (for equalization)
        and masked frames in the actual runs.
        """

        # -------------------------------------------------------------------
        # Create builder
        # -------------------------------------------------------------------
        tx_config = get_exp_tx_config(gain=self.gain)
        builder = ScheduleBuilder(
            schedule_name,
            channel=self.channel,
            cache_dir=Path.cwd() / ".cache" / self.campaign_name,
        )

        # Add base session
        if training_reps > 0:
            # NOTE: We use frame groups also for the warmup. The main reason is the USRP:
            # Transmitting small frames repeatedly does not work well, probably because of
            # some type of "ramp-up" happening in the USRP. A continuous stream (zero padded
            # between frames) works much better. Otherwise, receiving does not work at all.
            base_mask = np.ones(
                (int(self.channel.bandwidth.in_mhz() * 3.2), 1), dtype=np.complex64
            )

            warmup_frame = get_interleaved_frame_group(
                [self.iwl_base_frame, self.qca_base_frame],
                base_mask,
                training_reps,
                self.if_delay,
            )

            self.frames.append(warmup_frame)
            warmup_frame_id = warmup_frame.frame_id()

            builder = (
                builder.prepare_collect_csi(
                    device_ids=self.iwl_group,
                    collection_name=schedule_name + "_warmup",
                    filter_frame=self.iwl_base_frame,
                )
                .prepare_collect_csi(
                    device_ids=self.qca_group,
                    collection_name=schedule_name + "_warmup",
                    filter_frame=self.qca_base_frame,
                )
                .prepare_transmit(self.transmitter_id, warmup_frame_id, tx_config)
                .run(
                    min_capture_num=int(training_reps * 0.8),
                    max_capture_num=training_reps,
                )
            )

        if mask is not None:
            frame = get_interleaved_frame_group(
                [self.iwl_base_frame, self.qca_base_frame],
                mask,
                group_reps,
                self.if_delay,
            )
            frame_id = frame.frame_id()
            self.frames.append(frame)

            # Figure out number of frames in this transmission
            num_frames = group_reps * mask.shape[1]

            builder = (
                builder.prepare_collect_csi(
                    device_ids=self.iwl_group,
                    collection_name=schedule_name + "_run",
                    filter_frame=self.iwl_base_frame,
                )
                .prepare_collect_csi(
                    device_ids=self.qca_group,
                    collection_name=schedule_name + "_run",
                    filter_frame=self.qca_base_frame,
                )
                .prepare_transmit(self.transmitter_id, frame_id, tx_config)
                .run(
                    min_capture_num=int(num_frames * 0.8),
                    max_capture_num=num_frames,
                )
            )

        schedule = builder.build()
        schedule.extra_labels = extra_labels
        self.schedules.append(schedule)

    def run(self):
        """
        Run the configured schedules
        """
        seen = set()
        dedup_frames = []

        for x in self.frames:
            if x.frame_id() not in seen:
                seen.add(x.frame_id())
                dedup_frames.append(x)

        campaign = Campaign(
            schedules=self.schedules,
            frames=dedup_frames,
            device_cfgs=self.receiver + [self.transmitter],
            name=self.campaign_name,
        )

        # Persist for later inspection
        write_to_disk(
            campaign,
            Path.cwd() / "data" / f"{self.campaign_name}" / "campaign",
            mode=SerializationMode.PICKLE,
        )

        n_retry = self.num_retries

        # Campaign may have failing sessions, so we iterate until all are successful
        with Database(Path.cwd() / "data" / f"{self.campaign_name}", append=True) as db:
            while not campaign.is_empty() and n_retry > 0:
                # Run the current campaign
                with CampaignRunner(campaign) as runner:
                    try:
                        result = runner.run()
                    except ExceptionGroup as g:
                        print_exception_group(g)
                        n_retry -= 1
                        continue
                    except Exception as e:  # pylint: disable=broad-exception-caught
                        logger.exception(e)
                        # Note: campaign remains unchanged - it failed. We dont save data.
                        n_retry -= 1
                        continue

                    n_retry -= 1

                    # Store results and advance with new campaign containing only failed sessions
                    runner.store_results(db)
                    campaign = extract_failed_campaign(campaign, result)

        if not campaign.is_empty():
            logger.trace(
                f"Campaign contains failed sessions after {self.num_retries} retries; Saving for later..."
            )
            write_to_disk(
                campaign,
                Path.cwd() / ".cache" / "failed_campaigns" / f"{campaign.name}",
                append=False,
                mode=SerializationMode.PICKLE,
            )


class ExperimentFixture80MHz:
    """
    Experiment fixture to simplify experiment definition / DRY
    """

    def __init__(
        self,
        campaign_name: str,
        rescale_factor: int = 25000,
        gain: int = 11,
        if_delay: timedelta = timedelta(milliseconds=1),
    ):
        self.num_retries = 10
        self.campaign_name = campaign_name
        self.receiver = [
            get_asus_1(),
            get_asus_2(),
            get_ax210(),
            get_ps_usrp(),
        ]

        self.group = [receiver.device_id() for receiver in self.receiver]

        self.transmitter = get_usrp()
        self.transmitter_id = self.transmitter.device_id()

        self.channel = Channel(
            number=157, bandwidth=Bandwidth(80), center_freq_hz=5_775_000_000
        )

        self.base_frame = IQFrameConfig(
            receiver_address=MacAddr("00:16:ea:12:34:56"),
            transmitter_address=MacAddr("00:16:ea:12:34:56"),
            bssid_address=MacAddr("ff:ff:ff:ff:ff:ff"),
            bandwidth=Bandwidth(80),
            send_rate_hz=100_000_000,
            rescale_factor=rescale_factor,
            data_rate_mode=DataRateMode.VERY_HIGH_THROUGHPUT,
        )

        self.schedules: list[Schedule] = []
        self.frames = []

        self.gain = gain
        self.if_delay = if_delay
        rich.traceback.install(show_locals=False)

    def add_schedule_for_mask(
        self,
        mask: np.ndarray | None,
        schedule_name: str,
        training_reps: int = 500,
        group_reps: int = 1000,
        **extra_labels,
    ):
        """
        Add schedules to capture with qca and iwl, using warmup (for equalization)
        and masked frames in the actual runs.
        """

        # -------------------------------------------------------------------
        # Create builder
        # -------------------------------------------------------------------
        tx_config = get_exp_tx_config(gain=self.gain)
        builder = ScheduleBuilder(
            schedule_name,
            channel=self.channel,
            cache_dir=Path.cwd() / ".cache" / self.campaign_name,
        )

        # Add base session
        if training_reps > 0:
            # NOTE: We use frame groups also for the warmup. The main reason is the USRP:
            # Transmitting small frames repeatedly does not work well, probably because of
            # some type of "ramp-up" happening in the USRP. A continuous stream (zero padded
            # between frames) works much better. Otherwise, receiving does not work at all.
            base_mask = np.ones(
                (int(self.channel.bandwidth.in_mhz() * 3.2), 1), dtype=np.complex64
            )

            warmup_frame = get_frame_group(
                self.base_frame,
                base_mask,
                training_reps,
                self.if_delay,
            )

            self.frames.append(warmup_frame)
            warmup_frame_id = warmup_frame.frame_id()

            builder = (
                builder.prepare_collect_csi(
                    device_ids=self.group,
                    collection_name=schedule_name + "_warmup",
                    filter_frame=self.base_frame,
                )
                .prepare_transmit(self.transmitter_id, warmup_frame_id, tx_config)
                .run(
                    min_capture_num=int(training_reps * 0.8),
                    max_capture_num=training_reps,
                )
            )

        if mask is not None:
            frame = get_frame_group(
                self.base_frame,
                mask,
                group_reps,
                self.if_delay,
            )
            frame_id = frame.frame_id()
            self.frames.append(frame)

            # Figure out number of frames in this transmission
            num_frames = group_reps * mask.shape[1]

            builder = (
                builder.prepare_collect_csi(
                    device_ids=self.group,
                    collection_name=schedule_name + "_run",
                    filter_frame=self.base_frame,
                )
                .prepare_transmit(self.transmitter_id, frame_id, tx_config)
                .run(
                    min_capture_num=int(num_frames * 0.8),
                    max_capture_num=num_frames,
                )
            )

        schedule = builder.build()
        schedule.extra_labels = extra_labels
        self.schedules.append(schedule)

    def run(self):
        """
        Run the configured schedules
        """
        seen = set()
        dedup_frames = []

        for x in self.frames:
            if x.frame_id() not in seen:
                seen.add(x.frame_id())
                dedup_frames.append(x)

        campaign = Campaign(
            schedules=self.schedules,
            frames=dedup_frames,
            device_cfgs=self.receiver + [self.transmitter],
            name=self.campaign_name,
        )

        # Persist for later inspection
        write_to_disk(
            campaign,
            Path.cwd() / "data" / f"{self.campaign_name}" / "campaign",
            mode=SerializationMode.PICKLE,
        )

        n_retry = self.num_retries

        # Campaign may have failing sessions, so we iterate until all are successful
        with Database(Path.cwd() / "data" / f"{self.campaign_name}", append=True) as db:
            while not campaign.is_empty() and n_retry > 0:
                # Run the current campaign
                with CampaignRunner(campaign) as runner:
                    try:
                        result = runner.run()
                    except ExceptionGroup as g:
                        print_exception_group(g)
                        n_retry -= 1
                        continue
                    except Exception as e:  # pylint: disable=broad-exception-caught
                        logger.exception(e)
                        # Note: campaign remains unchanged - it failed. We dont save data.
                        n_retry -= 1
                        continue

                    n_retry -= 1

                    # Store results and advance with new campaign containing only failed sessions
                    runner.store_results(db)
                    campaign = extract_failed_campaign(campaign, result)

        if not campaign.is_empty():
            logger.trace(
                f"Campaign contains failed sessions after {self.num_retries} retries; Saving for later..."
            )
            write_to_disk(
                campaign,
                Path.cwd() / ".cache" / "failed_campaigns" / f"{campaign.name}",
                append=False,
                mode=SerializationMode.PICKLE,
            )


def print_exception_group(exception_group: ExceptionGroup):
    """
    Print the exceptions in an exception group..
    """
    logger.error(f"{exception_group.args[0]}:")  # args[0] contains the main message
    for exc in exception_group.exceptions:
        logger.exception(exc)


def interesting_subcarriers() -> list[int]:
    """
    Get a list of a few sort of representative subcarriers
    """
    return [
        4,  # 1: First  data subcarrier
        5,  # Mirror of 8
        6,  # 2: ax210 strange edge effect
        13,  # 3: ax210 skewed adjacency influence
        16,  # 4: just a "normal" subcarrier
        24,  # 5: qca weird improved detection performance
        25,  # 6: Pilot
        31,  # 7: Right next do DC zero subcarrier
        33,  # Mirror of 7
        39,  # Mirror of 6
        40,  # Mirror of 5
        48,  # Mirror of 4
        51,  # Mirror of 3
        58,  # Mirror of 2
        59,  # 8: Edge subcarrier for asus (because last edge sc is broken, nexmon issue?)
        60,  # Mirror of 1
    ]
