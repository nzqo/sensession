import pytest

from sensession.config import MacAddr, Bandwidth, BaseFrameConfig


def test_invalid_bandwidths():
    """
    Test that invalid bandwidths cause a type error in the creation of Bandwidth objects
    """

    invalid_bandwidths = [0, 5, 10, 21, 100, 180, 720]
    for bw in invalid_bandwidths:
        with pytest.raises(ValueError):
            _ = Bandwidth(bw)


def test_invalid_macs():
    """
    Assert that creation with invalid MAC addresses fails
    """

    with pytest.raises(ValueError):
        _ = MacAddr("aa:bb:cc:dd:ee:fff")

    with pytest.raises(ValueError):
        _ = MacAddr("a")

    with pytest.raises(ValueError):
        _ = MacAddr("")

    with pytest.raises(ValueError):
        _ = MacAddr("aa.bb.cc.dd.ee.ff")

    with pytest.raises(ValueError):
        _ = MacAddr("aabbccddeefff")

    with pytest.raises(ValueError):
        _ = MacAddr("gg:gg:gg:gg:gg:gg")


def test_valid_macs():
    """
    Assert that creation with invalid MAC addresses fails
    """
    _ = MacAddr("aa:bb:cc:dd:ee:ff")
    _ = MacAddr("aa-bb-cc-dd-ee-ff")
    _ = MacAddr("aabbccddeeff")


def test_base_frame_config_generation():
    """
    Test generation of trivial base frame config
    """
    triv_cfg = BaseFrameConfig()
    assert triv_cfg.bssid_address is None
    assert triv_cfg.receiver_address is None
    assert triv_cfg.transmitter_address is None
    assert triv_cfg.bandwidth.value == 20
