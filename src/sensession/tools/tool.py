"""
CSI Tool Interface

CSI Tools are other applications that instrument hardware to capture CSI values.
This interface specifies the minimal set of functionalities required for them to
be used within sensession.
"""

from abc import ABCMeta, abstractmethod
from typing import Any, TypeVar, final
from pathlib import Path
from datetime import datetime
from dataclasses import field, replace, dataclass

import numpy as np
from loguru import logger
from polars._typing import TimeUnit

from sensession.config import Channel, FrameId, BaseFrameConfig, BaseTransmissionConfig
from sensession.devices import DeviceId
from sensession.tools.device_registry import DEVICE_REGISTRY

DEFAULT_TIMESTAMP_UNIT: TimeUnit = "ns"


# fmt: off
@dataclass
class CsiDataPoint:
    """
    A single captured DataPoint
    """
    timestamp    : np.uint64
    sequence_num : np.uint16
    csi          : np.ndarray              # [num_rx_antennas, num_spatial_streams, num_subcarriers]
    rssi         : np.int8                 # single value for whole frame
    antenna_rssi : list[np.int8]           # [num_rx_antennas]
    agc          : np.uint8 | None = None
    fft_gain_esp : np.uint8 | None = None

    def __post_init__(self):
        if isinstance(self.antenna_rssi, np.ndarray):
            self.antenna_rssi = self.antenna_rssi.tolist()

# Group of CSI values
CsiGroup = list[CsiDataPoint]


@dataclass
class CsiMeta:
    """
    A metadata struct describing the setting around which a group of CSI
    values were captured
    """

    collection_start : datetime       = datetime.now()
    collection_stop  : datetime       = datetime.now()
    channel          : Channel | None = None
    antenna_idxs     : list[np.uint8] = field(default_factory=lambda : [np.uint8(0)])
    stream_idxs      : list[np.uint8] = field(default_factory=lambda : [np.uint8(0)])
    subcarrier_idxs  : list[np.int16] = field(default_factory=list)
    receiver_name    : str            = ""
    filter_frame     : str | None     = None                    # Frame that was filtered for
    timestamp_unit   : TimeUnit       = DEFAULT_TIMESTAMP_UNIT  # Time unit for the timestamps in CsiGroup

@dataclass
class CaptureResult:
    """
    Struct to bundle result of a CSI capture. A capture refers to running a tool
    for some amount of time and amassing the collected data.
    """
    receiver_id     : DeviceId                 # Unique ID of the receiver
    csi             : CsiGroup | None = None   # The captured data. May be None if nothing was captured.
    meta            : CsiMeta  | None = None   # Metadata relevant to the capture
# fmt: on


#######################################################################################
## Generic Tool interface. All Tools to be used within sensession must adhere to this
## by inheriting. This will ensure that they are present in the device registry and
## not managed by multiple tools at the same time.
#######################################################################################
class Tool(metaclass=ABCMeta):
    """
    Base class for all Tools possibly participating in the collection
    """

    def __init__(self):
        self.devices = {}
        self._is_running: bool = False

        # Keep temporary CsiMeta object. This object will store shared metadata
        # to allow simpler access to metadata by only changing device-dependent
        # fields. The others are automatically managed by this parent class.
        self._tmp_meta = CsiMeta()

    @final
    def _get_device_meta(
        self,
        antenna_idxs: list[int],
        stream_idxs: list[int],
        subcarrier_idxs: list[int],
        receiver_name: str,
    ) -> CsiMeta:
        """
        Get CsiMeta object adapted to device specific parameters
        """
        return replace(
            self._tmp_meta,
            antenna_idxs=list(map(np.uint8, antenna_idxs)),
            stream_idxs=list(map(np.uint8, stream_idxs)),
            subcarrier_idxs=list(map(np.int16, subcarrier_idxs)),
            receiver_name=receiver_name,
        )

    @abstractmethod
    def _device_setup(self, device: Any):
        """
        Perform required setup steps upon adding a device for management, if any.

        Args:
            device: A device to operate on
        """

    @abstractmethod
    def _device_teardown(self, device_id: DeviceId):
        """
        Perform required teardown/reset steps on the device, if any
        """

    @abstractmethod
    def _run(self) -> None:
        """
        Start running the tool as configued
        """

    @abstractmethod
    def _stop(self) -> None:
        """
        Stop running the tool
        """

    @final
    def run(self) -> None:
        """
        Start running the tool as configued
        """
        # update temporary meta
        self._tmp_meta = replace(
            self._tmp_meta,
            collection_start=datetime.now(),
        )
        self._run()
        self._is_running = True

    @final
    def stop(self) -> None:
        """
        Stop running the tool
        """

        if not self.is_running():
            return

        self._tmp_meta = replace(
            self._tmp_meta,
            collection_stop=datetime.now(),
        )
        self._stop()
        self._is_running = False

    @final
    def is_running(self) -> bool:
        """
        Check whether tool is running
        """
        return self._is_running

    @final
    def reset(self) -> None:
        """
        Reset the tool (including internal resources)
        """
        if not self.devices:
            return

        self.stop()

        for device_id in self.devices.copy():
            self.remove_device(device_id)

    @final
    def add_device(self, device: Any, strict: bool = False) -> DeviceId:
        """
        Public API for registering device

        Args:
            device : The device to add
            strict : If true, will raise error when device is already present

        NOTE: Don't forget to call `remove_device` when finished.

        Returns:
            the unique device id
        """
        device_id = device.get_config().device_id()
        logger.trace(f"Adding device {device_id} to tool ...")
        if device_id in self.devices:
            if strict:
                raise RuntimeError(
                    f"A device with id {device_id} is already registered."
                )
            return device_id

        if DEVICE_REGISTRY.is_registered(device_id):
            raise RuntimeError(
                f"Device {device_id} is already maintained by a different tool!"
            )

        self._device_setup(device)
        self.devices[device_id] = device
        DEVICE_REGISTRY.add_device(device_id, device.config, self)
        return device_id

    @final
    def remove_device(self, device_id: DeviceId, unknown_ok: bool = False):
        """
        Public API for unregistering device

        NOTE: It is important that this method is called. Otherwise, tools are
        never cleared until the EOL of the program, because a reference to them
        is held in the registry.

        Args:
            device_id : Unique device identifier returned on adding device to management
        """
        logger.trace(f"Removing device {device_id} from tool, including teardown ...")
        if not unknown_ok and device_id not in self.devices:
            raise RuntimeError(
                f"Tried to remove {device_id} from tool, but it is not known."
            )

        self._device_teardown(device_id)
        self.devices[device_id].reset()
        del self.devices[device_id]
        DEVICE_REGISTRY.remove(device_id)


#######################################################################################
## CSI specific Receiving Tool Interface
#######################################################################################
class CsiReceiver(Tool, metaclass=ABCMeta):
    """
    Abstract interface base class specifying a CSI Capture-enabled Tool.
    """

    @abstractmethod
    def _setup_capture(
        self,
        devices: list[DeviceId],
        channel: Channel,
        cache_dir: Path,
        frame: BaseFrameConfig | None,
    ):
        """
        Start capturing.

        Args:
            frame     : Frame to listen for
            channel   : Channel to capture on
            cache_dir : Path to put temporary files in

        Warning:
            Must not block!
        """

    @abstractmethod
    def _reap(self) -> list[CaptureResult]:
        """
        Stop the capture and reap the captured data
        """

    @final
    def reap(self) -> list[CaptureResult]:
        """
        Public API for stopping reaping captured data
        """
        if self.is_running():
            self.stop()

        return self._reap()

    @final
    def setup_capture(
        self,
        devices: list[DeviceId] | DeviceId,
        channel: Channel,
        cache_dir: Path,
        filter_frame: BaseFrameConfig | None = None,
    ):
        """
        Public API for capturing. Performs checks and then forwards to implementation.
        """
        logger.trace("Setting up capture...")
        # Ensure that optional frame to filter for is expecting the same bandwidth
        # as the channel we are listening on.
        if filter_frame:
            assert filter_frame.bandwidth == channel.bandwidth, (
                "Bandwidth of channel and frame to filter for don't match"
            )

        # Allow both single or multiple devices to be specified
        if isinstance(devices, DeviceId):
            devices = [devices]

        for device_id in devices:
            if device_id not in self.devices:
                raise ValueError(f"Device {device_id} not known; can't capture!")

        # Ensure directory exists
        cache_dir.mkdir(exist_ok=True, parents=True)

        # update temporary meta
        self._tmp_meta = replace(
            self._tmp_meta,
            channel=channel,
            filter_frame=filter_frame.frame_id() if filter_frame else None,
        )

        # Delegate to implementations
        self._setup_capture(devices, channel, cache_dir, filter_frame)


#######################################################################################
## CSI specific Transmitter Tool Interface
#######################################################################################
AnyTransmissionConfig = TypeVar("AnyTransmissionConfig", bound=BaseTransmissionConfig)


class CsiTransmitter(Tool, metaclass=ABCMeta):
    """
    Abstract interface base class specifying a CSI Transmission-enabled Tool.
    """

    @abstractmethod
    def setup_transmit(
        self,
        device_id: DeviceId,
        frame_id: FrameId,
        channel: Channel,
        tx_config: AnyTransmissionConfig,
    ):
        """
        Start transmission of a frame.

        Args:
            device_id : Device to transmit with
            frame_id  : ID of frame (config) to transmit
            channel   : Channel to transmit on
            tx_config : Transmission config (device specific, e.g. gain, repetitions)
        """
