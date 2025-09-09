"""
The device registry is a singleton keeping track of the state of separate devices.
This allows to ensure that e.g. one does not accidentally try to have a piece of
hardware managed by two separate tools (e.g. an Atheros card by the Ath9k tool and
PicoScenes). It also allows for simpler references, where the managing tool can be
looked up based on the identifier of the card.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from dataclasses import dataclass

from loguru import logger

if TYPE_CHECKING:
    from sensession.tools import Tool
    from sensession.devices import DeviceId


@dataclass
class RegistryEntry:
    """
    Wrapper type for registry entry
    """

    config: Any
    tool: Tool


class DeviceRegistry:
    """
    Registry to keep track of devices
    """

    def __init__(self):
        self._registry: dict[DeviceId, RegistryEntry] = {}

    def add_device(self, device_id: DeviceId, device_config: Any, tool: Tool):
        """
        Add device to the registry

        Args:
            device_id     : Unique identifier of the device
            device_config : Device configuration
            tool          : Tool by which the device is managed
        """
        if device_id in self._registry:
            raise RuntimeError(
                f"Device {device_id} is already in the registry; Hash collision or conflicting use detected!"
            )

        self._registry[device_id] = RegistryEntry(device_config, tool)

    def remove(self, device_id: DeviceId):
        """
        Remove device from registry

        Args:
            device_id : Unique identifier of the device
        """
        logger.trace(
            f"Removing device {self.get_short_name(device_id)} ({device_id}) from registry ..."
        )
        del self._registry[device_id]

    def get(self, device_id: DeviceId) -> RegistryEntry:
        """
        Get device information from the registry

        Args:
            device_id : Unique identifier of the device
        """
        return self._registry[device_id]

    def get_short_name(self, device_id: DeviceId) -> str:
        """
        Get short name of a device from its ID

        Args:
            device_id : Unique identifier of the device
        """
        cfg = self.get(device_id).config
        return cfg.short_name if hasattr(cfg, "short_name") else ""

    def is_registered(self, device_id: DeviceId) -> bool:
        """
        Check whether a device with a given id is already registered
        """
        return device_id in self._registry


# Device registry singleton
DEVICE_REGISTRY = DeviceRegistry()
