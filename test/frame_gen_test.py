from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pytest

from sensession.config import MacAddr, Bandwidth
from sensession.util.frame_generation import (
    Mask,
    IQFrameConfig,
    GeneratedFrameInfo,
    IQFrameGroupConfig,
    importance_sorted,
    select_trim_frames,
)


def get_dummy_iqframe() -> IQFrameConfig:
    """
    Get a dummy IQ Frame Config
    """

    return IQFrameConfig(
        receiver_address=MacAddr("aa:aa:aa:aa:aa:aa"),
        transmitter_address=MacAddr("bb:bb:bb:bb:bb:bb"),
        bssid_address=MacAddr("cc:cc:cc:cc:cc:cc"),
        bandwidth=Bandwidth(20),  # Channel bandwidth
    )


def get_dummy_iqframegroup() -> IQFrameGroupConfig:
    """
    Get a dummy IQ Frame Group Config
    """
    return IQFrameGroupConfig(base_frame=get_dummy_iqframe())


def get_dummy_gen_frame(size: int = 100):
    """
    Get a dummy Generated IQ Frame Config
    """

    return GeneratedFrameInfo(
        frame_config=get_dummy_iqframegroup(),
        frame_file=Path.cwd() / f"dummy_size{size}",
        created_at=datetime.now(),
        file_size=size,
    )


def test_iq_frame_config_generation():
    """
    Test generation of trivial IQ frame config
    """
    with pytest.raises(TypeError):
        _ = IQFrameConfig()


def test_frame_group_cfg_type_conversion():
    """
    Tests that frame group config enforces specified types to minimize initialization
    failure likelihoods.
    """
    addr = "aabbccddeeff"
    base_cfg = IQFrameConfig(addr, addr, addr, bandwidth=20)

    a = IQFrameGroupConfig(
        base_cfg,
        mask=np.ones((64, 1), dtype=np.complex64),
    )
    assert isinstance(a.base_frame.receiver_address, MacAddr), (
        "Expecting mac address conversion"
    )
    assert isinstance(a.base_frame.transmitter_address, MacAddr), (
        "Expecting mac address conversion"
    )
    assert isinstance(a.base_frame.bssid_address, MacAddr), (
        "Expecting mac address conversion"
    )
    assert isinstance(a.bandwidth, Bandwidth), "Expecting bandwidth conversion"
    assert isinstance(a.mask, Mask), "Expecting mask conversion"

    with pytest.raises(ValueError):
        _ = IQFrameGroupConfig(base_frame=base_cfg, group_repetitions=0)

    # Frame config must have Mac Addr as type
    with pytest.raises(TypeError):
        _ = IQFrameGroupConfig()


def test_invalid_framegroup_init():
    """
    Tests that invalid parameters in creation of framegroup config cause an exception
    """
    addr = MacAddr("aabbccddeeff")
    base_cfg = IQFrameConfig(addr, addr, addr)

    with pytest.raises(ValueError):
        _ = IQFrameGroupConfig(base_frame=base_cfg, group_repetitions=0)

    # Interframe delay can only be specified when there is a non trivial mask
    # or repetitions of the base frame are specified. Otherwise, there is no
    # "inter-frame" space anyway!
    with pytest.raises(ValueError):
        _ = IQFrameGroupConfig(base_frame=base_cfg, interframe_delay=10)


def test_frame_cfg_type_enforcement():
    """
    Test that frame config enforces specified types
    """
    with pytest.raises(ValueError):
        _ = IQFrameConfig(receiver_address="")

    with pytest.raises(TypeError):
        _ = IQFrameConfig(bandwidth=20)


def test_trivial_hashes():
    """
    Test frame and frame group config hashing, specifically under modifications that
    do not change the underlying frame ensure that the hashes don't change!
    """
    iq_config = get_dummy_iqframe()

    trivial_mask = Mask(np.ones((64, 1), dtype=np.complex64))
    iq_group1 = IQFrameGroupConfig(base_frame=iq_config)
    iq_group2 = IQFrameGroupConfig(base_frame=iq_config, mask=trivial_mask)
    iq_group3 = IQFrameGroupConfig(base_frame=iq_config, interframe_delay=timedelta(0))

    assert iq_config.frame_id() == "522d8652c40acdc1d6eb373b329ed64e", (
        "Hash must be reproducible and equal the hardcoded one!"
    )
    assert iq_config.frame_id() == iq_group1.frame_id(), (
        "Trivial group should have same hash as singular frame config"
    )
    assert iq_config.frame_id() == iq_group2.frame_id(), (
        "Mask should be considered trivial even if not None"
    )
    assert iq_config.frame_id() == iq_group3.frame_id(), (
        "Timedelta of zero must still be considered trivial."
    )


def test_importance_sort():
    """
    Test that importance sort logic works; This is required for proper functioning
    of the Frame Cache
    """

    frame_sizes = [500, 100, 200, 300]
    original_frames = {size: get_dummy_gen_frame(size=size) for size in frame_sizes}

    # First check that sizes are sorted in reverse
    frames = importance_sorted(frame_dict=original_frames)
    assert list(frames.keys()) == sorted(frame_sizes, reverse=True), (
        "Importance sort doesnt sort after size"
    )

    # If we schedule frames, this should cause a split
    frames = importance_sorted(frame_dict=original_frames, schedule=[500, 100])
    assert list(frames.keys()) == [
        300,
        200,
        500,
        100,
    ], "Schedule not taken into account properly"


def test_trim_frames():
    frame_sizes = [500, 100, 200, 300]
    frames = {size: get_dummy_gen_frame(size=size) for size in frame_sizes}

    frames1 = select_trim_frames(frame_dict=frames, remove_bytes=600)
    assert list(frames1.keys()) == [500, 100], "First two frames make up 600 bytes"

    frames2 = select_trim_frames(frame_dict=frames, remove_bytes=601)
    assert list(frames2.keys()) == [
        500,
        100,
        200,
    ], "To exceed 600 bytes, need to remove another"

    frames3 = select_trim_frames(frame_dict=frames, remove_bytes=0)
    assert not frames3, "Removing nothing should be trivial"

    frames3 = select_trim_frames(frame_dict=frames, remove_bytes=100000)
    assert list(frames3.keys()) == frame_sizes, "Overshooting should remove all"
