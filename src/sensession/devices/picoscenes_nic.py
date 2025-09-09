"""
Abstraction for a single Network Interface Card to be operated by Picoscenes.
"""

from __future__ import annotations

from dataclasses import dataclass

from sensession.devices.nic import NICConfig, NetworkInterfaceCard
from sensession.csi_file_parser import antenna_idxs_to_bitmask
from sensession.util.remote_access import SshPasswordless


# fmt: off
@dataclass
class PSNICConfig(NICConfig):
    """
    Config for a PicoScenes-operated Network Interface Card
    """

    phy_path: int  # PHY Path, see PicoScenes 'array_status' command
    remote_access_cfg: SshPasswordless | None = None  # Optionally USRP may be located on a remote system

    def instantiate_device(self) -> PSNIC:
        """
        Create corresponding device to this config
        """
        return PSNIC(self)
# fmt: on


class PSNIC(NetworkInterfaceCard):
    """
    PicoScenes NetworkInterfaceCard

    For PicoScenes, we can reuse the same setup procedures as for the
    generic NIC. Hence, this class only needs to store the updated
    config.
    """

    def __init__(self, config: PSNICConfig):
        super().__init__(config, config.remote_access_cfg)
        self.config: PSNICConfig = config

    def get_rxparams(self) -> str:
        """
        Construct parameter string according to current state
        """
        if min(self.config.antenna_idxs) < 0:
            raise AttributeError("Unknown antenna selected")

        # NOTE: Temporarily disabling + f"--rxcm {recv_antenna_bitmask}"
        #         recv_antenna_bitmask = antenna_idxs_to_bitmask(self.config.antenna_idxs)
        # Reason: PicoScenes ax210 crashes and yields not data when rx chainmask is set!
        return f"--interface {self.config.phy_path} "

    def get_txparams(self) -> str:
        """
        Construct TX parameter string according to current state
        """
        if min(self.config.antenna_idxs) < 0:
            raise AttributeError("Unknown antenna selected")
        antenna_bitmask = antenna_idxs_to_bitmask(self.config.antenna_idxs)
        return f"--interface {self.config.phy_path} " + f"--txcm {antenna_bitmask}"
