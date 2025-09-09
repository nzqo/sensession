"""
Example of running a manual sensing session.

Receiver: Asus router, operated with Nexmon
Transmitter: USRP (raw IQ frame transmission)
"""

from pathlib import Path
from datetime import timedelta
from ipaddress import IPv4Address

import polars as pl
from loguru import logger

from sensession import Channel, MacAddr, Database, Bandwidth
from sensession.config import DataRateMode
from sensession.devices import (
    SenseiNic,
    PSNICConfig,
    PSUSRPConfig,
    SenseiNexmon,
    IQFrameConfig,
    UsrpulseConfig,
    SenseiRemoteConfig,
    ServerConnectionType,
    BaseTransmissionConfig,
)
from sensession.campaign.runner import CampaignRunner, extract_failed_campaign
from sensession.campaign.campaign import Campaign, SerializationMode, write_to_disk
from sensession.campaign.schedule import Schedule, ScheduleBuilder
from sensession.devices.ath9k_nic import Ath9kNICConfig
from sensession.util.remote_access import SshPasswordless
from sensession.util.frame_generation import InterleavedIQFrameGroupConfig


def get_interleaved_frame_group(
    base_frames: list[IQFrameConfig],
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
        data_rate_mode=DataRateMode.HIGH_THROUGHPUT,
        bandwidth=Bandwidth(40),
        send_rate_hz=50_000_000,
        rescale_factor=rescale_factor,
    )


def get_iwl_frame(rescale_factor: int = 25000) -> IQFrameConfig:
    """
    IWL
    """
    return IQFrameConfig(
        receiver_address=MacAddr("00:16:ea:12:34:56"),
        transmitter_address=MacAddr("00:16:ea:12:34:56"),
        bssid_address=MacAddr("11:11:11:11:11:11"),
        data_rate_mode=DataRateMode.HIGH_THROUGHPUT,
        bandwidth=Bandwidth(40),
        send_rate_hz=50_000_000,
        rescale_factor=rescale_factor,
    )


class EightyMhzExperiment:
    """
    Eighty megahertz experiment fixture
    """

    def __init__(self, experiment_name="test_botong", tx_gain: int = 30):
        self.experiment_name = experiment_name
        self.frame_reps = 1500

        nic_cfg = PSNICConfig(
            name="Intel ax210",
            short_name="ax210",
            interface="wlp3s0",
            mac_address="7c:50:79:07:b8:e5",
            antenna_idxs=[0],
            stream_idxs=[0],
            phy_path=3,
        )

        # See `uhd_find_devices` to find e.g. the serial number
        ps_usrp_cfg = PSUSRPConfig(
            name="Usrp N2954-R",
            short_name="usrp",
            serial_num="30F3CEE",
            rf_port_num=0,
            antenna_idxs=[0],
            stream_idxs=[0],
            rx_gain=0.45,
        )

        iwl_config = SenseiRemoteConfig(
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

        qca_config = Ath9kNICConfig(
            name="Qualcomm Atheros AR9462",
            short_name="qca",
            interface="wlp3s0",
            mac_address="70:77:81:69:51:bf",
            antenna_idxs=[0],
            stream_idxs=[0],
            repo_path="~/Development/csi-modules",
            access=SshPasswordless(remote_ssh_hostname="shuttle"),
        )

        asus1_config = SenseiRemoteConfig(
            short_name="asus1",
            name="Asus RT-AC86U #1",
            remote_resource_id="nexmon",
            addr="192.168.20.31",
            port=5501,
            connection_type=ServerConnectionType.TCP,
            antenna_idxs=[0],
            stream_idxs=[0],
            source_cfg=SenseiNexmon(),
        )

        asus2_config = SenseiRemoteConfig(
            short_name="asus2",
            name="Asus RT-AC86U #2",
            remote_resource_id="nexmon",
            addr="192.168.20.32",
            port=5501,
            connection_type=ServerConnectionType.TCP,
            antenna_idxs=[0],
            stream_idxs=[0],
            source_cfg=SenseiNexmon(),
        )

        self.usrp_config = UsrpulseConfig(
            "USRP-x310",
            addr=IPv4Address("192.168.20.125"),
            port=8080,
            ssh_name="sdr",
        )

        self.receiver_cfgs = [
            nic_cfg,
            ps_usrp_cfg,
            asus1_config,
            asus2_config,
            iwl_config,
            qca_config,
        ]

        self.receiver_ids = [cfg.device_id() for cfg in self.receiver_cfgs]

        self.transmitter_id = self.usrp_config.device_id()

        self.channel = Channel(
            number=157,
            bandwidth=Bandwidth(40),
        )

        # Some random arbitrary frame (asus can capture everything)
        self.iwl_frame = get_iwl_frame()
        self.qca_frame = get_qca_frame()

        # Basic iq frame config
        self.frame = get_interleaved_frame_group(
            [self.iwl_frame, self.qca_frame],
            self.frame_reps,
            timedelta(milliseconds=1),
        )
        self.frame_id = self.frame.frame_id()

        self.tx_config = BaseTransmissionConfig(gain=tx_gain)

        self.schedules: list[Schedule] = []

    def add_session(
        self, rep_nr: int, activity_idx: int, position_idx: int, subject_name: str
    ):
        """
        Add session
        """
        builder = ScheduleBuilder(
            self.experiment_name,
            channel=self.channel,
            cache_dir=Path.cwd() / ".cache" / self.experiment_name,
        )

        # Add base session
        builder = (
            builder.prepare_collect_csi(
                device_ids=self.receiver_ids[:-1],
                collection_name=self.experiment_name,
                filter_frame=self.iwl_frame,
            )
            .prepare_collect_csi(
                device_ids=[self.receiver_ids[-1]],
                collection_name=self.experiment_name,
                filter_frame=self.qca_frame,
            )
            .prepare_transmit(self.transmitter_id, self.frame_id, self.tx_config)
            .run(min_capture_num=int(self.frame_reps * 0.7))
        )

        schedule = builder.build()
        schedule.extra_labels = {
            "activity_idx": (activity_idx, pl.UInt8),
            "position_idx": (position_idx, pl.UInt8),
            "human_label": (subject_name, pl.String),
            "rep_nr": (rep_nr, pl.UInt8),
        }

        self.schedules.append(schedule)

    def run(self, num_retries: int = 3):
        """
        Run the configured schedules
        """

        campaign = Campaign(
            schedules=self.schedules,
            frames=[self.frame],
            device_cfgs=self.receiver_cfgs + [self.usrp_config],
            name=self.experiment_name,
        )

        # Persist for later inspection
        write_to_disk(
            campaign,
            Path.cwd() / "data" / f"{self.experiment_name}" / "campaign",
            mode=SerializationMode.PICKLE,
        )

        n_retry = num_retries
        # Campaign may have failing sessions, so we iterate until all are successful
        with Database(f"data/{self.experiment_name}", append=True) as db:
            while not campaign.is_empty() and n_retry > 0:
                # Run the current campaign
                with CampaignRunner(campaign) as runner:
                    try:
                        result = runner.run()
                    except Exception as e:  # pylint: disable=broad-exception-caught
                        logger.error(f"Encountered unhandled exception: {e}")
                        # Note: campaign remains unchanged - it failed. We dont save data.
                        n_retry -= 1
                        continue

                    n_retry -= 1

                    # Store results and advance with new campaign containing only failed sessions
                    runner.store_results(db)
                    campaign = extract_failed_campaign(campaign, result)

        if not campaign.is_empty():
            logger.trace(
                f"Campaign contains failed sessions after {num_retries} retries; Saving for later..."
            )
            write_to_disk(
                campaign,
                Path.cwd() / ".cache" / "failed_campaigns" / f"{campaign.name}",
                append=False,
                mode=SerializationMode.PICKLE,
            )


if __name__ == "__main__":
    experiment = EightyMhzExperiment("test_botong")

    ACTIVITY_IDX: int = 4
    POSITION_IDX: int = 1
    REPS: int = 10
    SUBJECT_NAME: str = "Botong"

    for i in range(REPS):
        experiment.add_session(
            rep_nr=i,
            activity_idx=ACTIVITY_IDX,
            position_idx=POSITION_IDX,
            subject_name=SUBJECT_NAME,
        )

    experiment.run(num_retries=3)
