"""
A simply dummy device that can't do anything but provide a fixed CSI value
"""

from __future__ import annotations

from dataclasses import dataclass

from sensession.config import Bandwidth, DataRateMode
from sensession.devices.nic import NICConfig


@dataclass
class DummyConfig(NICConfig):
    """
    Config class for ath9k Network Interface Card
    """

    num_captures: int = 10

    def instantiate_device(self) -> DummyDevice:
        """
        Create corresponding device to this config
        """
        return DummyDevice(self)


def lrange(lower: int, upper: int) -> list[int]:
    """List range, edges included!"""
    return list(range(lower, upper + 1))


def get_subcarriers(bandwidth: Bandwidth, mode: DataRateMode):
    """
    Get number of total subcarrier (excluding DC and Guards)
    """
    # fmt: off
    subcarriers = {
        (Bandwidth.TWENTY,        DataRateMode.NON_HIGH_THROUGHPUT): lrange(-26, -1) + lrange(1, 26),
        (Bandwidth.TWENTY,        DataRateMode.HIGH_THROUGHPUT):     lrange(-28, -1) + lrange(1, 28),
        (Bandwidth.FOURTY,        DataRateMode.HIGH_THROUGHPUT):     lrange(-58, -2) + lrange(2, 58),
        (Bandwidth.EIGHTY,        DataRateMode.HIGH_THROUGHPUT):     lrange(-122, -2) + lrange(2, 122),
        (Bandwidth.HUNDRED_SIXTY, DataRateMode.HIGH_THROUGHPUT):     lrange(-250, -130) + lrange(-126, -6) + lrange(6,126) + lrange(130,250),
    }
    # fmt: on

    if (bandwidth, mode) not in subcarriers:
        raise ValueError(
            f"Bandwidth {bandwidth} and mode {mode} subcarrier list not found."
        )

    return subcarriers[(bandwidth, mode)]


### Device API
class DummyDevice:
    """
    Abstraction for an ESP32 device
    """

    def __init__(self, config: DummyConfig):
        """
        Constructor
        """
        self.config = config

    def reset(self):
        """
        Restore the device
        """

    def get_config(self) -> DummyConfig:
        """
        Get the internal config
        """
        return self.config
