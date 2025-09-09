"""
Classes for handling Devices to be used by Nexmon
"""

from __future__ import annotations

from ipaddress import IPv4Address
from dataclasses import dataclass

from loguru import logger

from sensession.config import Channel, MacAddr
from sensession.util.net import get_sideband_value
from sensession.util.hash import get_hash
from sensession.util.shell import shell_run
from sensession.devices.nic import NICState, NetworkInterfaceMode
from sensession.csi_file_parser import antenna_idxs_to_bitmask
from sensession.util.remote_access import SshPasswordless


# fmt: off
@dataclass
class NexmonRouterConfig:
    """
    Device configuration of a router running the Nexmon-CSI patched firmware
    """
    access_cfg      : SshPasswordless   # Remote access to the router
    name            : str               # Full name of the card for information
    short_name      : str               # Short name for references
    interface       : str               # Network interface on host attached to router
    mac_address     : MacAddr           # Mac address of the router
    host_ip         : IPv4Address       # IP address of the host from the router
    netcat_port     : int               # Netcat port to stream data from router to host
    antenna_idxs    : list[int]         # Attached/to-be-used antennas
    stream_idxs     : list[int]         # Streams to be extracted

    def device_id(self) -> str:
        """
        Get deterministic device id based on config
        """
        return get_hash(f"{self.name}:{self.short_name}:{self.interface}:{self.access_cfg.remote_ssh_hostname}")

    def __post_init__(self):
        if not isinstance(self.host_ip, IPv4Address):
            self.host_ip = IPv4Address(self.host_ip)

        if isinstance(self.mac_address, dict):
            self.mac_address = MacAddr(**self.mac_address) # pylint: disable=not-a-mapping

        if self.access_cfg and isinstance(self.access_cfg, dict):
            self.access_cfg = SshPasswordless(**self.access_cfg) # pylint: disable=not-a-mapping

    def instantiate_device(self) -> NexmonRouter:
        """
        Create corresponding device to this config
        """
        return NexmonRouter(self)
# fmt: on


def get_interface(channel: Channel) -> str:
    """
    Nexmon devices all use the same interface for capturing, depending solely
    on the channel.
    """
    if 1 <= channel.number <= 14:
        return "eth5"
    if 32 <= channel.number <= 177:
        return "eth6"

    raise ValueError(
        f"Channel {channel.number} is outside of known valid channel range"
    )


class NexmonRouter:
    """
    Abstraction for a single router running Nexmon CSI
    """

    def __init__(self, config: NexmonRouterConfig):
        self.config = config
        self.state: NICState | None = None  # State of the card
        self.filter_addr: MacAddr | None = None  # Mac Addr to filter for

    def tune(
        self,
        mode: NetworkInterfaceMode,
        channel: Channel,
        transmitter_address: MacAddr | None,
    ):
        """
        Args:
            mode    : Target mode to tune into
            channel : Target channel to tune router to
        """
        target_state = NICState(mode, channel)
        if self.state == target_state and self.filter_addr == transmitter_address:
            return

        if mode == NetworkInterfaceMode.MONITOR:
            sideband = get_sideband_value(channel)
            interface = get_interface(channel)
            recv_antenna_bitmask = antenna_idxs_to_bitmask(self.config.antenna_idxs)
            stream_bitmask = antenna_idxs_to_bitmask(self.config.stream_idxs)

            shell_run(
                "./scripts/nexmon/setup_interface.sh "
                + f"{self.config.access_cfg.remote_ssh_hostname} "
                + f"{interface} "
                + f"{channel.number} "
                + f"{channel.bandwidth.in_mhz()} "
                + f"{transmitter_address.get() if transmitter_address else ''} "
                + f"{recv_antenna_bitmask} "
                + f"{stream_bitmask} "
                + f"{sideband}",
                try_count=3,  # hardcoded retry count. Rarely ssh connection hangs.
            )

            # Update state
            self.state = target_state
            self.filter_addr = transmitter_address
        else:
            raise NotImplementedError("Tuning is only implemented for monitor mode")

    def reset(self):
        """
        Restore the device
        """
        logger.trace(
            f"Resetting asus router state {self.config.access_cfg.remote_ssh_hostname}"
        )
        self.state = None
        self.filter_addr = None

    def get_config(self) -> NexmonRouterConfig:
        """
        Get the internal config
        """
        return self.config
