"""
Abstraction for a USRP device to be operated by Picoscenes.
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from sensession.config import Channel
from sensession.csi_file_parser import antenna_idxs_to_bitmask
from sensession.util.remote_access import SshPasswordless


# fmt: off
@dataclass
class PSUSRPConfig:
    """
    Config for a PicoScenes-operated Network Interface Card
    """
    name              : str                            # Full name of the USRP for information
    short_name        : str                            # Short name
    serial_num        : str                            # Globally unique serial address!
    antenna_idxs      : list[int]                      # Attached/to-be-used antennas
    stream_idxs       : list[int]                        # Streams to be extracted
    rf_port_num       : int                    = 0     # Either 0 (left chain; TX/RX) or 1 (right; RX)
    rx_gain           : int | float            = 0.45  # Absolute gain in [0,66] dB or normalized hardware-supported in [0,1)
    remote_access_cfg : SshPasswordless | None = None  # Optionally USRP may be located on a remote system

    def device_id(self) -> str:
        """
        Get deterministic device id based on config
        """
        return self.serial_num

    def __post_init__(self):
        assert self.rf_port_num in [0, 1], "Port num can only be zero or one"
        assert (0 <= self.rx_gain <= 66), "RX Gain must be absolute integer in interval [0,66] or normalized float in interval [0,1]"

        if max(self.antenna_idxs) > 2 or min(self.antenna_idxs) < 0:
            raise AttributeError("Unknown antenna selected")

        if self.remote_access_cfg:
            raise NotImplementedError(
                "Picoscenes on a remote machine not yet implemented!"
            )

    def instantiate_device(self) -> PSUSRP:
        """
        Create corresponding device to this config
        """
        return PSUSRP(self)
# fmt: on


@dataclass
class PSUSRPState:
    """
    Struct to contain the modifiable internal USRP state
    """

    center_freq_mhz: int = 2412
    bandwidth_mhz: int = 20


class PSUSRP:
    """
    PicoScenes USRP SDR

    Configuration information of a receiver
    """

    def __init__(
        self,
        config: PSUSRPConfig,
    ):
        self.config = config
        self.state = PSUSRPState()

        # USRP has only 2 antennas (and two chains). The antenna indices are
        # given per chain, hence we only allow 0 and 1.
        if any(x not in [0, 1] for x in self.config.antenna_idxs):
            raise AttributeError("Unknown antenna selected")

    def tune(self, channel: Channel):
        """
        Tune USRP into specific channel

        Args:
            channel : Target channel to tune card to listen on
        """
        self.state = PSUSRPState(
            center_freq_mhz=int(channel.center_freq_hz / 1e6),
            bandwidth_mhz=channel.bandwidth.in_mhz(),
        )

        logger.trace("No need to change mode")

    def reset(self):
        """
        Restore the device
        Unlike NIC, no mode conversion is required to use USRP
        """
        return

    def get_config(self) -> PSUSRPConfig:
        """
        Get the internal config
        """
        return self.config

    def get_rxparams(self) -> str:
        """
        Construct RX parameter string according to current state
        """
        # recv_antenna_bitmask = antenna_idxs_to_bitmask(self.config.antenna_idxs)
        rf_port_name = "TX/RX" if self.config.rf_port_num == 0 else "RX2"
        antenna_bitmask = antenna_idxs_to_bitmask(self.config.antenna_idxs)

        # NOTE: Temporarily disabling rxcm chainmask setting because it leads to no-data
        # in the ax210. Because it affects postprocessing of the picoscenes file, we disable
        # it in total and instead post-filter the antennas
        return (
            f"--interface usrp{self.config.serial_num} "
            + f"--freq {self.state.center_freq_mhz} "
            + f"--preset RX_CBW_{self.state.bandwidth_mhz} "
            + f"--rx-ant {rf_port_name} "
            + f"--rx-gain {self.config.rx_gain} "
            + f"--rxcm {antenna_bitmask} "
        )

    def get_txparams(self, txpower: int | float) -> str:
        """
        Construct TX parameter string according to current state
        """
        antenna_bitmask = antenna_idxs_to_bitmask(self.config.antenna_idxs)

        return (
            f"--interface usrp{self.config.serial_num} "
            + f"--freq {self.state.center_freq_mhz} "
            + f"--txcm {antenna_bitmask} "
            + f"--txpower {txpower}"
        )
