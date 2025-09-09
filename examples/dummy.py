"""
Non-trivial Campaign execution example
"""

from pathlib import Path

from sensession import Channel, MacAddr, Database, Bandwidth
from sensession.devices import DummyConfig, IQFrameConfig, BaseTransmissionConfig
from sensession.campaign import Campaign, CampaignRunner, ScheduleBuilder


def get_dummy(idx: int = 0) -> DummyConfig:
    """
    Get dummy device config
    """
    return DummyConfig(
        name=f"Dummy {idx}",
        short_name=f"dummy_{idx}",
        interface="dummy",
        mac_address=MacAddr("ff:ff:ff:ff:ff:ff"),
        antenna_idxs=[0, 1],
        stream_idxs=[0, 1],
    )


def get_frame() -> IQFrameConfig:
    """
    Create an example IQ frame
    """
    return IQFrameConfig(
        receiver_address=MacAddr("ff:ff:ff:ff:ff:ff"),
        transmitter_address=MacAddr("ff:ff:ff:ff:ff:ff"),
        bssid_address=MacAddr("ff:ff:ff:ff:ff:ff"),
        bandwidth=Bandwidth(20),
    )


def build_campaign() -> Campaign:
    """
    Create a campaign
    """
    collecting_devices = [get_dummy(i) for i in range(10)]
    receiver_ids = [rx.device_id() for rx in collecting_devices]
    transmitter = get_dummy(11)
    transmitter_id = transmitter.device_id()
    frame = get_frame()

    builder = ScheduleBuilder(
        "dummy_schedule",
        channel=Channel(
            number=157,
            bandwidth=Bandwidth(20),
        ),
        cache_dir=Path.cwd() / ".cache" / "dummy_campaign",
    )

    schedule = (
        builder.prepare_collect_csi(receiver_ids, "dummy", filter_frame=frame)
        .prepare_transmit(
            transmitter_id, frame.frame_id(), BaseTransmissionConfig(gain=25)
        )
        .run()
        .build()
    )

    print(f"Schedule: \n{schedule.to_json()}")

    return Campaign(
        schedules=[schedule],
        frames=[frame],
        device_cfgs=collecting_devices + [transmitter],
        name="dummy_campaign",
    )


def execute_example():
    """
    Create and execute an example campaign
    """
    campaign = build_campaign()

    with CampaignRunner(campaign) as runner:
        result = runner.run()

        with Database("data/dummy_campaign", append=False) as db:
            runner.store_results(db)

    for sched_res in result.data:
        for collection_res in sched_res.data:
            for data in collection_res.data:
                receiver = data.meta.receiver_name
                coll_name = collection_res.name
                sched_name = sched_res.name
                print(
                    f" -- {sched_name} - {coll_name} - {receiver} - num captures: {len(data.csi)} "
                )


if __name__ == "__main__":
    execute_example()
