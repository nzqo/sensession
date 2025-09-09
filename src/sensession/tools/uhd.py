"""
Atheros QCA CSI Tool
"""

from typing import Any

from loguru import logger

from sensession.config import Channel, FrameId, BaseTransmissionConfig
from sensession.devices import UhdUsrp, DeviceId
from sensession.tools.tool import CsiTransmitter
from sensession.util.frame_generation import FrameGenerator, GeneratedFrameInfo


class Uhd(CsiTransmitter):
    """
    Uhd class to handle raw USRP transmission
    """

    def __init__(self, frame_generator: FrameGenerator):
        super().__init__()
        self.frame_generator = frame_generator

        # temp variables for transmission
        self.tx_device: DeviceId | None = None
        self.frame: GeneratedFrameInfo | None = None
        self.tx_config: BaseTransmissionConfig | None = None
        self.channel: Channel | None = None

    def setup_transmit(
        self,
        device_id: DeviceId,
        frame_id: FrameId,
        channel: Channel,
        tx_config: BaseTransmissionConfig,
    ):
        """
        Transmit with a Uhd USRP device
        """
        device = self.devices[device_id]
        assert isinstance(device, UhdUsrp), "UHD only handles UhdUsrp objects!"

        self.tx_device = device_id
        logger.trace(f"Retrieving raw frame file for frame {frame_id} ...")
        self.frame = self.frame_generator.retrieve_frame(frame_id)
        self.tx_config = tx_config
        self.channel = channel

    def _run(self):
        """
        Run UHD transmission
        """
        assert self.tx_device, (
            "No transmission device specified; Call `setup_transmit` first!"
        )
        assert self.tx_config, (
            "No transmission config specified; Call `setup_transmit` first!"
        )
        assert self.frame, (
            "No transmission frame config specified; Call `setup_transmit` first!"
        )

        device = self.devices[self.tx_device]
        logger.info(f"Starting transmission with UHD device {self.tx_device}")
        device.transmit(self.frame, self.channel, self.tx_config)

    def _stop(self):
        """
        Stop transmission

        NOTE: Transmission is blocking and there is never any need to stop it
        in code. Hence, this just performs a reset.
        """
        self.tx_device = None
        self.frame = None
        self.tx_config = None
        self.channel = None

    def _device_setup(self, device: Any):
        """
        Perform required setup steps upon adding a device for management, if any.

        Args:
            device: A device to operate on
        """

    def _device_teardown(self, device_id: DeviceId):
        """
        Perform required teardown/reset steps on the device, if any
        """
