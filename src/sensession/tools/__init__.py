"""
Re-export Tool classes to allow for more natural imports
"""

from sensession.devices import BaseEnum, DeviceConfigType
from sensession.tools.uhd import Uhd
from sensession.tools.tool import (
    Tool,
    CsiMeta,
    CsiGroup,
    CsiReceiver,
    CaptureResult,
    CsiTransmitter,
)
from sensession.tools.ath9k import Ath9k
from sensession.tools.dummy import Dummy
from sensession.tools.espion import ESP32Tool
from sensession.tools.nexmon import Nexmon
from sensession.tools.sensei import Sensei
from sensession.tools.usrpulse import Usrpulse
from sensession.tools.picoscenes import PicoScenes, PicoscenesTransmissionConfig
from sensession.tools.device_registry import DEVICE_REGISTRY


class ToolType(BaseEnum):
    """
    Enum of all the available Tool types
    """

    PICOSCENES = PicoScenes
    NEXMON = Nexmon
    ATH9K = Ath9k
    UHD = Uhd
    ESP_32 = ESP32Tool
    DUMMY = Dummy
    SENSEI = Sensei
    USRPULSE = Usrpulse


#######################################################################################
## Mapping between device config types and the corresponding tools to manage them
## Don't forget to add tool to DeviceConfigType
#######################################################################################
CONFIG_TOOL_MAP = {
    DeviceConfigType.ATH9K_NIC: ToolType.ATH9K,
    DeviceConfigType.NEXMON_ROUTER: ToolType.NEXMON,
    DeviceConfigType.PS_NIC: ToolType.PICOSCENES,
    DeviceConfigType.PS_USRP: ToolType.PICOSCENES,
    DeviceConfigType.UHD_USRP: ToolType.UHD,
    DeviceConfigType.ESP_32: ToolType.ESP_32,
    DeviceConfigType.DUMMY: ToolType.DUMMY,
    DeviceConfigType.USRPULSE_USRP: ToolType.USRPULSE,
    DeviceConfigType.SENSEI: ToolType.SENSEI,
}


def get_tool_type(config_type: DeviceConfigType) -> ToolType:
    """
    Find tool type corresponding to config type
    """
    if config_type not in CONFIG_TOOL_MAP:
        raise ValueError(
            f"Config type {config_type} unknown; Forgot to add to CONFIG_TOOL_MAP?"
        )
    return CONFIG_TOOL_MAP[config_type]
