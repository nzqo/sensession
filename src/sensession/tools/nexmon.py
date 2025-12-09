"""
Nexmon CSI Tool
"""

from typing import Deque
from pathlib import Path
from collections import deque
from dataclasses import dataclass

from loguru import logger

from sensession.config import Channel
from sensession.tools.tool import (
    CsiMeta,
    DeviceId,
    CsiReceiver,
    CaptureResult,
    BaseFrameConfig,
)
from sensession.util.temp_file import TempFile
from sensession.util.capture_process import CaptureProcess
from sensession.devices.nexmon_router import (
    NexmonRouter,
    NetworkInterfaceMode,
    get_interface,
)
from sensession.csi_file_parser.nexmon import load_nexmon_data


@dataclass
class CapturedDataFile:
    """
    Struct to connect receiver to temp file in which data captures are stored
    """

    receiver_name: str  # Name of receiver that captured
    receiver_id: DeviceId  # ID of the receiver
    file: TempFile  # File in which data was captured


class Nexmon(CsiReceiver):
    """
    Nexmon Tool Class.

    Abstracts handling of multiple network devices with Nexmon-CSI installed and
    exposes a low-res API for their operation.
    """

    def __init__(self):
        super().__init__()

        self.bg_process = CaptureProcess()
        self.capture_files: Deque[CapturedDataFile] = deque()
        self.devices: dict[DeviceId, NexmonRouter] = {}

        # Nexmon gives microsecond timestamp; Overwrite meta accordingly
        self._tmp_meta = CsiMeta(timestamp_unit="us")

        self.tmp_channel: Channel | None = None
        self.tmp_capture_devices: dict = {}

    def _device_setup(self, device: NexmonRouter):
        """
        Add a device for management under Nexmon
        """
        if not isinstance(device, NexmonRouter):
            raise NotImplementedError(
                "Nexmon Tool only supports Nexmon Router Devices."
            )

    def _device_teardown(self, device_id: DeviceId):
        """
        Perform devide teardown
        """
        # Nothing to do that the base class doesnt already do
        logger.trace(f"Removing nexmon device {device_id}")

    def _setup_capture(
        self,
        devices: list[DeviceId],
        channel: Channel,
        cache_dir: Path,
        frame: BaseFrameConfig | None,
    ):
        """
        Start capturing with given devices
        """
        capture_devices = {k: self.devices[k] for k in devices}

        # Ensure that queue of capture files is empty in the beginning.
        # Otherwise, if reap isn't called, it will amass partial data from previous runs
        self.capture_files.clear()

        for digest, device in capture_devices.items():
            receiver_name = device.config.short_name

            # NOTE: Empty address will create csiparams for Nexmon without a specific address
            # to filter for. That is exactly the behavior we want when no frame is specified.
            device.tune(
                NetworkInterfaceMode.MONITOR,
                channel,
                frame.transmitter_address if frame else None,
            )

            # Append file for postprocessing queue
            tmp_file = TempFile(f"{receiver_name}.pcap", cache_dir)
            self.capture_files.append(
                CapturedDataFile(
                    receiver_name=receiver_name,
                    receiver_id=digest,
                    file=tmp_file,
                )
            )

            logger.debug(
                "Starting netcat listen on host to receive CSI ...\n"
                + f" -- receiver ssh name : {receiver_name}\n"
                + f" -- tmp file          : {tmp_file.path}\n"
                + f" -- destination port  : {device.config.netcat_port}\n"
                + f" -- destination ip    : {device.config.host_ip}\n"
            )

            self.bg_process.start_process(
                shell_command=(
                    "./scripts/nexmon/csi_capture_start.sh"
                    + f" {device.config.netcat_port} {tmp_file.path}"
                ),
                cleanup_command="./scripts/nexmon/csi_capture_stop.sh",
            )

        self.tmp_channel = channel
        self.tmp_capture_devices = capture_devices

    def _run(self):
        """
        Start capturing
        """
        # Nexmon routers all use the same interface for capturing, and it solely
        # depends on the channel
        assert self.tmp_channel, "No channel specified; Did you call `setup_capture`?"

        interface = get_interface(self.tmp_channel)

        # Start CSI forwarding stream on Nexmon devices
        for device in self.tmp_capture_devices.values():
            ssh_name = device.config.access_cfg.remote_ssh_hostname
            self.bg_process.start_process(
                shell_command=(
                    "./scripts/nexmon/csi_stream_start.sh "
                    f"{ssh_name} "
                    f"{interface} "
                    f"{device.config.netcat_port} "
                    f"{device.config.host_ip}"
                ),
                cleanup_command=f"./scripts/nexmon/csi_stream_stop.sh {ssh_name}",
            )

    def _stop(self):
        """
        Stop Capturing
        """
        self.bg_process.teardown(ignore_codes=[-15, 130])

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

            assert device.state and device.state.channel, (
                "Device must have been tuned to collect (and then reap)"
            )

            # If the file contains no data or doesnt even exist, we have no data to reap.
            if tmp_file.empty() or tmp_file.path.stat().st_size < 100:
                logger.warning(f"No data found for {receiver_name} ({receiver_id})")
                tmp_file.close()
                continue

            capture_res = CaptureResult(receiver_id=capture.receiver_id)

            # Try to load data and unpack into the capture result if present
            res = load_nexmon_data(
                tmp_file.path,
                device.state.channel.bandwidth,
                device.config.antenna_idxs,
                device.config.stream_idxs,
            )
            tmp_file.close()

            if not res:
                logger.warning(f"No data found for {receiver_name} ({receiver_id})")
                continue

            csi, subcarrier_idxs = res

            # Create metadata
            meta = self._get_device_meta(
                antenna_idxs=device.config.antenna_idxs,
                stream_idxs=device.config.stream_idxs,
                subcarrier_idxs=subcarrier_idxs,
                receiver_name=receiver_name,
            )

            # Save data and metadata
            capture_res.csi = csi
            capture_res.meta = meta

            captures.append(capture_res)

            logger.debug(
                f"Loaded nexmon data for {receiver_name} ({receiver_id}). "
                + f"Found {len(csi)} data points"
            )

        return captures
