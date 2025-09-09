"""
A basic Network Interface Card (NIC) class to operate one.

The operation here is tool agnostic, but can be useful as a base class for
implementing tool-specific NIC abstractions.
"""

from enum import Enum
from dataclasses import dataclass

from sensession.config import Channel, MacAddr
from sensession.util.hash import get_hash
from sensession.util.shell import shell_run
from sensession.util.remote_access import Command, SshPasswordless


# fmt: off
@dataclass
class NICConfig:
    """
    Configuration information of a receiver
    """

    name         : str        # Full name of the card for information
    short_name   : str        # Short name for references
    interface    : str        # Corresponding network interface
    mac_address  : MacAddr    # Mac address of the card / interface
    antenna_idxs : list[int]  # Attached/to-be-used antennas
    stream_idxs  : list[int]  # Streams to be extracted

    def device_id(self) -> str:
        """
        Get deterministic device id based on config
        """
        return get_hash(f"{self.name}:{self.short_name}:{self.interface}")

    def __post_init__(self):
        if isinstance(self.mac_address, dict):
            self.mac_address = MacAddr(**self.mac_address) # pylint: disable=not-a-mapping
        if not isinstance(self.mac_address, MacAddr):
            self.mac_address = MacAddr(self.mac_address)

        if sorted(self.antenna_idxs) != self.antenna_idxs:
            raise ValueError("Please only use sorted antenna indices.")

class NetworkInterfaceMode(str, Enum):
    """
    Mode of network interface
    """
    MONITOR  = "monitor"      # Linux networking monitor mode
    MANAGED  = "managed"      # Managed mode

@dataclass
class NICState:
    """
    State of the NIC (e.g. mode and which channel it's tuned on).
    Used in state handling to avoid reperforming frequent operations.
    """
    mode : NetworkInterfaceMode
    channel : Channel | None
# fmt: on


class NetworkInterfaceCard:
    """
    Configuration information of a receiver
    """

    def __init__(
        self,
        config: NICConfig,
        remote_access_cfg: SshPasswordless | None = None,
    ):
        self.config = config
        self.state: NICState | None = None
        self.access = remote_access_cfg

    def tune(
        self,
        mode: NetworkInterfaceMode,
        channel: Channel,
    ):
        """
        Tune Card into specific mode and channel

        Args:
            mode    :  Target mode to tune card into
            channel : Target channel to tune card to listen on
        """
        target_state = NICState(mode, channel)
        if self.state == target_state:
            return

        if mode == NetworkInterfaceMode.MONITOR:
            monitor_mode_command = Command(
                "./scripts/monitor_mode.sh "
                + f"{self.config.interface} "
                + f"{int(channel.center_freq_hz / 1e6)} "
                + f"{channel.bandwidth.in_mhz()} "
                + f"{int(channel.control_freq_hz / 1e6)}"
            ).script_through_remote(self.access)

            shell_run(monitor_mode_command)
            self.state = target_state
        else:
            raise NotImplementedError("Tuning is only implemented for monitor mode")

    def reset(self):
        """
        Reset into base mode (managed)
        """
        cleanup_cmd = Command(
            f"sudo ifconfig {self.config.interface} down && "
            f"sudo iwconfig {self.config.interface} mode managed"
        ).on_remote(self.access)
        shell_run(cleanup_cmd)
        self.state = NICState(NetworkInterfaceMode.MANAGED, None)

    def get_config(self) -> NICConfig:
        """
        Get the internal config
        """
        return self.config
