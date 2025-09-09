"""
Remote triggering viconic
"""

from ipaddress import IPv4Address
from dataclasses import dataclass

import requests


@dataclass
class ViconicConfig:
    """
    Config for viconic device
    """

    short_name: str
    addr: IPv4Address
    port: int

    def __post_init__(self):
        """
        convert fromt str to IPv4Address
        """
        if isinstance(self.addr, str):
            self.addr = IPv4Address(self.addr)


class Viconic:
    """
    Viconic
    """

    def __init__(self, config: ViconicConfig):
        """
        Construct
        """
        self.config = config

    def start_at(self, seconds: int, capture_id: str):
        """
        Start at timecode (time since epoch)
        """
        url = f"http://{self.config.addr}:{self.config.port}/start_epoch"
        data = {"overwrite": True}
        params = {"capture_id": capture_id, "seconds_since_epoch": f"{seconds}"}

        requests.post(url, json=data, params=params, timeout=1).raise_for_status()

    def stop_at(self, seconds: int):
        """
        Stop at timecode (time since epoch)
        """

        url = f"http://{self.config.addr}:{self.config.port}/stop_epoch"

        data = {"overwrite": True}

        params = {"seconds_since_epoch": seconds}

        requests.post(url, json=data, params=params, timeout=1).raise_for_status()
