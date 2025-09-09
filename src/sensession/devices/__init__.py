"""
The devices submodule is meant to give a first layer of abstraction
for handling different devices (pertaining to both physical as well
as tool differences).

Every device corresponds to a single entity and should implement common
configuration and setup routines. CSI Tools are then later used to
instrument these devices in whichever ways the corresponding tools
want.
"""

from sensession.util.enum import BaseEnum
from sensession.devices.nic import NICConfig, NetworkInterfaceCard, NetworkInterfaceMode
from sensession.devices.dummy import DummyConfig, DummyDevice
from sensession.devices.esp32 import ESP32, ESP32Config
from sensession.devices.sensei import (
    SenseiNic,
    SenseiDevice,
    SenseiNexmon,
    SenseiRemoteConfig,
    ServerConnectionType,
)
from sensession.devices.viconic import Viconic, ViconicConfig
from sensession.devices.uhd_usrp import UhdUsrp, UhdUsrpConfig, BaseTransmissionConfig
from sensession.devices.ath9k_nic import Ath9kNIC, Ath9kNICConfig
from sensession.devices.nexmon_router import NexmonRouter, NexmonRouterConfig
from sensession.devices.usrpulse_usrp import UsrpulseUsrp, UsrpulseConfig
from sensession.util.frame_generation import (
    Mask,
    IQFrameConfig,
    IQFrameGroupConfig,
    InterleavedIQFrameGroupConfig,
)
from sensession.devices.picoscenes_nic import PSNIC, PSNICConfig
from sensession.devices.picoscenes_usrp import PSUSRP, PSUSRPConfig

DeviceId = str


#######################################################################################
## Mapping between strings and types
## NOTE: Mapping of device to their tools is done in sensession.tools.__init__
## Don't forget to add tool there as well.
#######################################################################################
class DeviceConfigType(BaseEnum):
    """
    Enum map to all available Device Configs
    """

    DUMMY = DummyConfig
    PS_NIC = PSNICConfig
    PS_USRP = PSUSRPConfig
    NEXMON_ROUTER = NexmonRouterConfig
    ATH9K_NIC = Ath9kNICConfig
    RAW_NIC = NICConfig
    ESP_32 = ESP32Config
    UHD_USRP = UhdUsrpConfig
    USRPULSE_USRP = UsrpulseConfig
    SENSEI = SenseiRemoteConfig
