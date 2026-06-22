"""
ESP32 Tool
"""

from typing import Deque
from pathlib import Path
from collections import deque
from dataclasses import dataclass

from loguru import logger

from sensession.config import Channel, FrameId, BaseTransmissionConfig
from sensession.tools.tool import (
    CsiMeta,
    DeviceId,
    CsiReceiver,
    CaptureResult,
    CsiTransmitter,
    BaseFrameConfig,
)
from sensession.devices.esp32 import ESP32
from sensession.util.temp_file import TempFile
from sensession.csi_file_parser.esp32parser import load_esp32_data


# Is this code duplication? Probably. Is that bad? I really don't think so
# pylint: disable=R0801
@dataclass
class CapturedDataFile:
    """
    Struct to connect receiver to temp file in which data captures are stored
    """

    receiver_name: str  # Name of receiver that captured
    receiver_id: DeviceId  # ID of the receiver
    file: TempFile  # File in which data was captured


class ESP32Tool(CsiReceiver, CsiTransmitter):
    """
    ESP32 Tool Class.
    """

    def __init__(self):
        super().__init__()

        self.capture_files: Deque[CapturedDataFile] = deque()
        self.devices: dict[DeviceId, ESP32] = {}

        # ESP32 gives microsecond timestamp; Overwrite meta accordingly
        self._tmp_meta = CsiMeta(timestamp_unit="us")

        self.tmp_channel: Channel | None = None
        self.tmp_capture_devices: dict = {}

        # TX state: maps device_id -> (device, frame, tx_config)
        self.frames: dict[FrameId, BaseFrameConfig] = {}
        self.tx_devices: dict[
            DeviceId, tuple[ESP32, BaseFrameConfig, BaseTransmissionConfig]
        ] = {}

    def register_frames(self, frames: list[BaseFrameConfig]):
        """
        Register frame configs so they can be referenced by ID in setup_transmit.
        """
        self.frames.update({frame.frame_id(): frame for frame in frames})

    def _device_setup(self, device: ESP32):
        """
        Add a device
        """
        if not isinstance(device, ESP32):
            raise NotImplementedError("ESP Tool only supports ESP Devices.")
        logger.trace("Setting up ESP device")

    def _device_teardown(self, device_id: DeviceId):
        logger.trace(f"Removing ESP device {device_id}")

    def _setup_capture(
        self,
        devices: list[DeviceId],
        channel: Channel,
        cache_dir: Path,
        frame: BaseFrameConfig | None,
    ):
        """
        NOTE: filtering is based on source mac (frame.transmitter_address) and dest mac (frame.receiver_address)
        All other fields in frame are ignored
        """
        capture_devices = {k: self.devices[k] for k in devices}

        # Ensure that queue of capture files is empty in the beginning.
        # Otherwise, if reap isn't called, it will amass partial data from previous runs
        self.capture_files.clear()

        for digest, device in capture_devices.items():
            receiver_name = device.config.short_name

            # Append file for postprocessing queue
            tmp_file = TempFile(f"{receiver_name}.csilog", cache_dir)
            self.capture_files.append(
                CapturedDataFile(
                    receiver_name=receiver_name,
                    receiver_id=digest,
                    file=tmp_file,
                )
            )

            device.set_filepath(tmp_file.path)
            device.connect_device()
            device.change_channel(channel.number)
            if frame:
                device.set_filter_rules(
                    frame.transmitter_address, frame.receiver_address
                )

            self.tmp_channel = channel
            self.tmp_capture_devices = capture_devices

            logger.debug(
                f"ESP32 device {receiver_name} ready, tmp file {tmp_file.path}"
            )

    def setup_transmit(
        self,
        device_id: DeviceId,
        frame_id: FrameId,
        channel: Channel,
        tx_config: BaseTransmissionConfig,
    ):
        device = self.devices[device_id]
        frame = self.frames[frame_id]
        device.connect_device()
        device.change_channel(channel.number)
        self.tx_devices[device_id] = (device, frame, tx_config)

    def _run(self):
        # RX path
        for device in self.tmp_capture_devices.values():
            device.start_receiving_csi()

        # TX path: resync clocks immediately before transmitting to minimize drift
        for device, frame, tx_config in self.tx_devices.values():
            device.sync_esp_clock()
            device.transmit_frame_with_tx_config(frame, tx_config)

    def _stop(self):
        # RX path
        for device in self.tmp_capture_devices.values():
            device.stop_receiving_csi()
            device.close_serial_connection()

        # TX path
        for device, _, _ in self.tx_devices.values():
            device.close_serial_connection()
        self.tx_devices.clear()

    def _reap(self) -> list[CaptureResult]:
        """
        Stop the capture and reap the captured data
        """
        captures = []

        # Go through all devices that were prepared for this session and reap their data.
        while self.capture_files:
            capture = self.capture_files.popleft()
            tmp_file = capture.file
            receiver_name = capture.receiver_name
            receiver_id = capture.receiver_id
            device = self.devices[receiver_id]

            # If the file contains no data or doesnt even exist, we have no data to reap.
            if tmp_file.empty() or tmp_file.path.stat().st_size < 10:
                tmp_file.close()
                continue

            subcarrier_idxs = device.get_csi_subcarrier_idxs()
            subcarrier_mask = device.get_subcarrier_mask()

            # Try to load data and unpack into the capture result if present
            csi = load_esp32_data(tmp_file.path, subcarrier_idxs, subcarrier_mask)

            tmp_file.close()

            if not csi:
                continue

            # Create metadata
            meta = self._get_device_meta(
                antenna_idxs=[0],  # ESP32 only has a single antenna
                stream_idxs=[0],  # and can thus capture from only a single stream
                subcarrier_idxs=sorted(subcarrier_idxs),
                receiver_name=receiver_name,
            )

            captures.append(
                CaptureResult(receiver_id=capture.receiver_id, meta=meta, csi=csi)
            )

            logger.debug(
                f"Loaded ESP32 CSI data for {receiver_name} ({receiver_id}). "
                + f"Found {len(csi)} data points"
            )

        return captures
