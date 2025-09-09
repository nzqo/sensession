"""
Sensei sources
--------------

Sensei exposes a collection from different sources. Since sensei is the master
of the devices, the devices here are only shims for sensei's configuration.
"""

from __future__ import annotations

import time
from enum import Enum
from ipaddress import IPv4Address
from dataclasses import dataclass

try:
    from sensei import (  # pylint: disable=import-error
        SourceType,
        ConnectionType,
        PyDataCollector,
        PyNicControlParams,
        PyNexmonControlParams,
    )

    SENSEI_AVAILABLE = True
except ImportError:
    SENSEI_AVAILABLE = False


from sensession.config import Channel, BaseFrameConfig
from sensession.util.net import get_sideband_value
from sensession.util.hash import get_hash
from sensession.csi_file_parser import antenna_idxs_to_bitmask


@dataclass
class SenseiNic:
    """
    Network interface card in sensei
    """

    interface: str
    scale_csi: bool = True

    def to_sensei(self) -> SourceType:
        """
        Convert to sensei source type
        """
        return SourceType.Iwl(scale_csi=self.scale_csi)


class ServerConnectionType(Enum):
    """
    The type of the server connection.

    Wrapper type because the sensei one can't be pickled.
    """

    TCP = (0,)
    WEB = (1,)

    def to_sensei(self) -> ConnectionType:
        """
        Convert to sensei connection type
        """
        match self:
            case ServerConnectionType.TCP:
                return ConnectionType.Tcp
            case ServerConnectionType.WEB:
                return ConnectionType.Web
            case _:
                raise ValueError(f"Unknown connection type: {self}")


@dataclass
class SenseiNexmon:
    """
    Nexmon device in sensei
    """

    def to_sensei(self) -> SourceType:
        """
        Convert to sensei source type
        """
        return SourceType.Nexmon()


# fmt: off
@dataclass
class SenseiRemoteConfig:
    """
    Configuration of a remote sensei source
    """
    short_name         : str                  # Unique name for this source
    name               : str                  # Long descriptive device name
    remote_resource_id : str                  # The name given to the source at the remote
    addr               : IPv4Address          # Addr of the remote hosting the source
    port               : int                  # Port under which remote is reachable
    connection_type    : ServerConnectionType # Type of connection to the remote
    antenna_idxs       : list[int]            # Antennas to be used for captures
    stream_idxs        : list[int]            # Streams to be extracted
    source_cfg         : SenseiNic | SenseiNexmon

    def device_id(self) -> str:
        """
        Get deterministic device id based on config
        """
        return get_hash(f"{self.short_name}:{self.addr}:{self.port}")

    def __post_init__(self):
        if isinstance(self.addr, str):
            self.addr = IPv4Address(self.addr)

    def instantiate_device(self) -> SenseiDevice:
        """
        Create a sensei device from this config
        """
        return SenseiDevice(self)
# fmt: on


class SenseiDevice:  # pylint: disable=too-few-public-methods
    """
    Trivial device abstraction for sensei
    """

    def __init__(self, config: SenseiRemoteConfig):
        if not SENSEI_AVAILABLE:
            raise ModuleNotFoundError(
                "Sensei not installed; Can't create a Sensei device."
            )

        self.config = config

    def get_config(self) -> SenseiRemoteConfig:
        """
        Config getter
        """
        return self.config

    def tune(
        self,
        channel: Channel,
        frame: BaseFrameConfig | None,
        collector: PyDataCollector,
    ):
        """
        Tune that thing
        """

        if isinstance(self.config.source_cfg, SenseiNic):
            nic_params = get_nic_params(self.config, channel)
            collector.configure_nic(self.config.short_name, nic_params)
        elif isinstance(self.config.source_cfg, SenseiNexmon):
            nex_params = get_nexmon_params(self.config, channel, frame)
            collector.configure_nexmon(self.config.short_name, nex_params)
        else:
            raise NotImplementedError(
                f"Unknown source config type: {type(self.config.source_cfg)}"
            )

        # Some safety setup time for background processes :)
        time.sleep(0.1)

    def reset(self):
        """
        Reset the device
        """
        # Nothing to do.


def get_nexmon_params(
    config: SenseiRemoteConfig, channel: Channel, frame: BaseFrameConfig | None
) -> PyNexmonControlParams:
    """
    Create nexmon control params
    """
    assert isinstance(config.source_cfg, SenseiNexmon)

    params: dict[str, int | str] = {
        "channel_number": channel.number,
        "channel_bw_mhz": channel.bandwidth.in_mhz(),
        "recv_antenna_mask": antenna_idxs_to_bitmask(config.antenna_idxs),
        "spatial_stream_mask": antenna_idxs_to_bitmask(config.stream_idxs),
    }

    if sideband := get_sideband_value(channel):
        params["sideband"] = sideband

    if frame and frame.transmitter_address:
        params["source_addr"] = frame.transmitter_address.get()

    return PyNexmonControlParams(**params)  # type: ignore


def get_nic_params(
    config: SenseiRemoteConfig,
    channel: Channel,
) -> PyNicControlParams:
    """
    Create NIC control params
    """
    assert isinstance(config.source_cfg, SenseiNic)

    bandwidth = channel.bandwidth.in_mhz()
    params: dict[str, int | str] = {
        "interface": config.source_cfg.interface,
        "center_freq_mhz": int(channel.center_freq_hz / 1e6),
        "bandwidth_mhz": bandwidth,
    }

    if bandwidth != 20:
        params["control_freq_mhz"] = int(channel.control_freq_hz / 1e6)

    return PyNicControlParams(**params)  # type: ignore
