"""
Non-trivial Campaign execution example
"""

from pathlib import Path
from datetime import timedelta
from ipaddress import IPv4Address

from sensession import Channel, MacAddr, Database, Bandwidth, SshPasswordless
from sensession.devices import (
    IQFrameConfig,
    UhdUsrpConfig,
    IQFrameGroupConfig,
    NexmonRouterConfig,
    BaseTransmissionConfig,
)
from sensession.campaign import (
    Campaign,
    Schedule,
    CampaignRunner,
    ScheduleBuilder,
    write_to_disk,
    read_from_disk,
    extract_failed_campaign,
)


def get_router() -> NexmonRouterConfig:
    """
    Get config for an example nexmon router
    """
    return NexmonRouterConfig(
        access_cfg=SshPasswordless("asus"),
        name="Asus RT-AC86U BM3000",
        short_name="asus",
        interface="enp7s0",
        mac_address=MacAddr("24:4b:fe:be:e6:00"),
        antenna_idxs=[0, 1, 2, 3],
        stream_idxs=[0],
        host_ip=IPv4Address("192.168.10.1"),
        netcat_port=5501,
    )


def get_usrp() -> UhdUsrpConfig:
    """
    Get config for an exmaple USRP transmitter
    """
    return UhdUsrpConfig(
        "USRP N2954R",
        serial_num="30F3CEE",
        access=SshPasswordless("sdr"),
    )


def get_channel() -> Channel:
    """
    Get the channel to transmit/receive on
    """
    return Channel(
        # center_freq_hz=2_452_000_000,  # 2_437_000_000,
        number=157,
        bandwidth=Bandwidth(40),
    )


def get_frame() -> IQFrameConfig:
    """
    Create an example IQ frame
    """
    return IQFrameConfig(
        receiver_address=MacAddr("ff:ff:ff:ff:ff:ff"),
        transmitter_address=MacAddr("12:34:56:c0:ff:ee"),
        bssid_address=MacAddr("01:1b:11:22:31:55"),
        bandwidth=Bandwidth(20),
        send_rate_hz=50_000_000,  # Usrp doesnt like sending at 40 MHz
    )


def get_frame_group() -> IQFrameGroupConfig:
    """
    Create a masked frame group
    """
    return IQFrameGroupConfig(
        base_frame=get_frame(),
        interframe_delay=timedelta(milliseconds=2),
        group_repetitions=100,
        # mask=Mask(np.ones((128, 2), dtype=np.complex64)),
    )


def get_warmup_tx_config() -> BaseTransmissionConfig:
    """
    Create an exmaple warmup config; Simply repeating transmission of a simple
    frame a few times
    """
    return BaseTransmissionConfig(gain=5, n_reps=500, pause_ms=5)


def get_exp_tx_config() -> BaseTransmissionConfig:
    """
    Create an example experiment config. In this case, the experiment is a single
    transmission of the frame group, which is a masked, repeated version of the frame
    used during warmup
    """
    return BaseTransmissionConfig(gain=5)


def get_warmup_exp_schedule(
    builder: ScheduleBuilder,
    router_id: str,
    usrp_id: str,
    frame: IQFrameConfig,
    group_frame: IQFrameGroupConfig,
) -> Schedule:
    """
    Create an experiment schedule with a warmup collection phase
    """
    return (
        builder.prepare_collect_csi(router_id, "warmup", filter_frame=frame)
        .prepare_transmit(usrp_id, frame.frame_id(), get_warmup_tx_config())
        .run()
        .wait(timedelta(seconds=1))
        .prepare_collect_csi(
            router_id, "experiment_1", filter_frame=group_frame.base_frame
        )
        .prepare_transmit(usrp_id, group_frame.frame_id(), get_exp_tx_config())
        .run()
        .build()
    )


def get_basic_exp_schedule(
    builder: ScheduleBuilder,
    router_id: str,
    usrp_id: str,
    group_frame: IQFrameGroupConfig,
) -> Schedule:
    """
    Create an experiment schedule with a warmup collection phase
    """
    return (
        builder.wait(timedelta(seconds=1))
        .prepare_collect_csi(
            router_id, "experiment_2", filter_frame=group_frame.base_frame
        )
        .prepare_transmit(usrp_id, group_frame.frame_id(), get_exp_tx_config())
        .run()
        .build()
    )


def build_campaign() -> Campaign:
    """
    Create a campaign
    """
    router = get_router()
    usrp = get_usrp()
    frame = get_frame()
    frame_group = get_frame_group()

    builder = ScheduleBuilder(
        "example_schedule",
        channel=get_channel(),
        cache_dir=Path.cwd() / ".cache" / "test_campaign",
    )

    # NOTE: All the wait commands are to hope for buffers being emptied such
    # that all data is in fact collected at the host. Otherwise, we will often
    # miss data!
    warmup_schedule = get_warmup_exp_schedule(
        builder,
        router.device_id(),
        usrp.device_id(),
        frame,
        frame_group,
    )
    experiment_schedule = get_basic_exp_schedule(
        builder, router.device_id(), usrp.device_id(), frame_group.frame_id()
    )

    print(f"Warmup schedule: \n{warmup_schedule.to_json()}")
    print(f"Basic schedule: \n{experiment_schedule.to_json()}")

    return Campaign(
        schedules=[warmup_schedule, experiment_schedule],
        frames=[frame, frame_group],
        device_cfgs=[router, usrp],
        name="example_campaign",
    )


def execute_example():
    """
    Create and execute an example campaign
    """
    campaign = build_campaign()

    with CampaignRunner(campaign) as runner:
        result = runner.run()

        with Database("data/test_campaign", append=False) as db:
            runner.store_results(db)

    for sched_res in result.data:
        if sched_res.error:
            print(f"Error: {sched_res.error}")
            continue

        for collection_res in sched_res.data:
            for data in collection_res.data:
                coll_name = collection_res.name
                sched_name = sched_res.name
                recv_name = data.meta.receiver_name
                print(
                    f" -- {sched_name} - {coll_name} - {recv_name} - Meta: {data.meta} "
                )

    # Test persistence of failed campaign
    camp = extract_failed_campaign(campaign, result)
    if not camp.is_empty():
        write_to_disk(camp, Path.cwd() / ".cache" / "failed_campaigns" / f"{camp.name}")
        test = read_from_disk(
            Path.cwd() / ".cache" / "failed_campaigns" / f"{camp.name}"
        )

        for c in test:
            print(c.to_json())


if __name__ == "__main__":
    execute_example()
