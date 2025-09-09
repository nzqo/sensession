"""
Atheros9k CSI Tool Network Interface Card
"""

from __future__ import annotations

from dataclasses import dataclass

from sensession.devices.nic import NICConfig, NetworkInterfaceCard
from sensession.util.remote_access import SshPasswordless


# fmt: off
@dataclass
class Ath9kNICConfig(NICConfig):
    """
    Config class for ath9k Network Interface Card
    """

    repo_path: str                         # Path of the csi-modules repository, required to find the extractor
    access: SshPasswordless | None = None  # Optionally card may be located on a remote system

    def __post_init__(self):
        super().__post_init__()
        if self.access and isinstance(self.access, dict):
            self.access = SshPasswordless(**self.access)

    def instantiate_device(self) -> Ath9kNIC:
        """
        Create corresponding device to this config
        """
        return Ath9kNIC(self)
# fmt: on


class Ath9kNIC(NetworkInterfaceCard):
    """
    Ath9k NetworkInterfaceCard

    For Ath9k, we can reuse the same setup procedures as for the
    generic NIC. Hence, this class only needs to store the updated
    config.
    """

    def __init__(self, config: Ath9kNICConfig):
        super().__init__(config, remote_access_cfg=config.access)
        self.config = config

    def get_config(self) -> Ath9kNICConfig:
        """
        Access internal config
        """
        assert isinstance(self.config, Ath9kNICConfig), (
            "Ath9k device must be configured with dedicated Ath9kNICConfig."
        )

        return self.config
