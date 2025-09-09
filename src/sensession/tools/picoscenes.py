"""
PicoScenes CSI Tool
"""

import time
from copy import deepcopy
from typing import Deque
from pathlib import Path
from collections import deque
from dataclasses import dataclass

from loguru import logger

from sensession.util import (
    Command,
    TempFile,
    ApiUsageError,
    CaptureProcess,
    SshPasswordless,
    shell_run,
    create_marshall_copy_command,
)
from sensession.config import Channel, FrameId, BaseFrameConfig, BaseTransmissionConfig
from sensession.devices import (
    PSNIC,
    PSUSRP,
    PSNICConfig,
    PSUSRPConfig,
    NetworkInterfaceMode,
)
from sensession.tools.tool import (
    DeviceId,
    CsiReceiver,
    CaptureResult,
    CsiTransmitter,
    AnyTransmissionConfig,
)
from sensession.csi_file_parser.picoscenes import load_picoscenes_data


# fmt: off
@dataclass
class CaptureTask:
    """
    Struct to connect receiver to temp file in which data captures are stored
    """

    receiver_name : str        # Name of receiver that captured
    receiver_id   : DeviceId   # ID of the receiver
    file          : TempFile   # File in which data was captured
    antenna_idxs  : list[int]  # Antenna idxs used for the capture
    stream_idxs   : list[int]  # Streams which to capture
    access_cfg    : SshPasswordless | None # Optional remote location of capture task


@dataclass
class PicoScenesTaskGroup:
    """
    Struct for a group of capture tasks belonging to a group of devices on the same host 
    """
    ps_cmd : str                # Expanded command to execute on host
    tasks  : Deque[CaptureTask]


@dataclass
class PicoscenesTransmissionConfig(BaseTransmissionConfig):
    """
    Tx injection freq config
    """

    tx_startdelay_s: int = 1  # A one-time delay before injection(unit in microseconds)

    def __post_init__(self):
        super().__post_init__()
        self.tx_startdelay_s = int(self.tx_startdelay_s)
# fmt:  on


def ps_filter_options(frame: BaseFrameConfig) -> str:
    """
    Get command options to filter as specified in the frame config.
    """
    assert frame.transmitter_address or frame.receiver_address, (
        "When specifying a frame in capture as filter, "
        + "specify at least one MAC to filter for!"
    )

    options = ""
    if frame and frame.transmitter_address:
        options += f" --source-address-filter {frame.transmitter_address.get()}"

    if frame and frame.receiver_address:
        options += f" --destination-address-filter {frame.receiver_address.get()}"
    return options


def get_target_option(frame: BaseFrameConfig) -> str:
    """
    Get command option set MAC address of the injection target
    """
    assert frame and frame.receiver_address, (
        "Target (receiver) address to send to must be specified!"
    )

    return f" --target-mac-address {frame.receiver_address.get()} "


def get_ps_command_start() -> str:
    """
    Return opening sequence of a Picoscenes command line invocation
    """
    return 'PicoScenes "-d debug; '


class PicoScenesTaskGroupBuilder:
    """
    Picoscenes CLI is issued a single command to capture with potentially more than one
    device. If they are spread between different remotes, we need to group them for each
    of them. This class helps in constructing the necessary task groups, which contain
    the command to execute as well as necessary information for reaping the data.
    """

    def __init__(self, access_cfg: SshPasswordless | None):
        self.access_cfg = access_cfg
        self.picoscenes_cmd = get_ps_command_start()
        self.tasks: Deque[CaptureTask] = deque()

    def add_capture(
        self,
        device_id: DeviceId,
        device: PSNIC | PSUSRP,
        cache_dir: Path,
        frame: BaseFrameConfig | None,
    ):
        """
        Build command to capture with the added devices.
        """

        device_name = device.config.short_name

        # Picoscenes will write to a file and append a ".csi" postfix.
        # It requires a relative path (which is why this is fine for all remotes as well).
        # NOTE: This represents the file on the sensession host where everything will be
        # synced to.
        outfile = f"./{cache_dir.relative_to(Path.cwd())}/{device_name}"
        tmp_file = TempFile(f"{device_name}.csi", cache_dir)

        self.tasks.append(
            CaptureTask(
                receiver_name=device_name,
                receiver_id=device_id,
                file=tmp_file,
                antenna_idxs=device.config.antenna_idxs,
                stream_idxs=device.config.stream_idxs,
                access_cfg=self.access_cfg,
            )
        )

        # Build the picoscenes command part for the currently considered receiver
        if frame:
            self.picoscenes_cmd += ps_filter_options(frame) + " "

        self.picoscenes_cmd += (
            f"{device.get_rxparams()} " + f"--output {outfile} " + "--mode logger; "
        )

    def add_transmit(
        self,
        device: PSNIC | PSUSRP,
        frame: BaseFrameConfig,
        channel: Channel,
        tx_config: AnyTransmissionConfig,
    ):
        """
        Add a transmission part to the command
        """
        if not isinstance(tx_config, PicoscenesTransmissionConfig):
            raise RuntimeError(
                f"Wrong config type! Should be PicoscenesTransmissionConfig, is {type(tx_config)}"
            )

        # Make sure fields of frame are complete
        if not frame.receiver_address:
            raise TypeError(
                "PicoScenes transmssion requires receiver address in frame config"
            )
        if not frame.transmitter_address:
            raise TypeError(
                "PicoScenes transmssion requires transmitter address in frame config"
            )

        # Extract mode specification (VHT, HT, etc.)
        mode_spec = frame.data_rate_mode.value

        # Build the picoscenes command part for the currently considered transmitter
        tx_params = {}

        if isinstance(device, PSUSRP):
            tx_params["txpower"] = tx_config.gain

        self.picoscenes_cmd += f"{device.get_txparams(**tx_params)} "

        # Add frame-specific parts (addresses etc.)
        self.picoscenes_cmd += get_target_option(frame)

        # Add the rest
        self.picoscenes_cmd += (
            f"--preset TX_CBW_{channel.bandwidth.in_mhz()}_{mode_spec} "
            + f"--repeat {tx_config.n_reps} "
            + f"--txpower {tx_config.gain} "
            + f"--delay {tx_config.pause_ms * 1000} "
            + f"--delayed-start {tx_config.tx_startdelay_s} "
            + "--mode injector; "
        )

        # -q Causes the PicoScenes command to quit after the transmission has finished.
        self.picoscenes_cmd += "-q;"

    def reset(self):
        """
        Reset Command builder to clean state
        """
        self.picoscenes_cmd = get_ps_command_start()
        self.tasks.clear()

    def build(self) -> PicoScenesTaskGroup:
        """
        Finalize PicoScenes command
        """
        # Close the quote of the opened up PicoScenes command
        self.picoscenes_cmd += '"'

        task_group = PicoScenesTaskGroup(
            ps_cmd=Command(self.picoscenes_cmd).on_remote(self.access_cfg),
            tasks=deepcopy(self.tasks),
        )

        self.reset()
        return task_group


def get_access_cfg(device: PSNIC | PSUSRP) -> SshPasswordless | None:
    """
    Extract access config from device
    """
    device_cfg = device.get_config()
    if not isinstance(device_cfg, (PSNICConfig | PSUSRPConfig)):
        raise RuntimeError(
            "Capture setup expects PicoScenes devices to have PicoScenes configs!"
        )
    return device_cfg.remote_access_cfg


class PicoScenes(CsiReceiver, CsiTransmitter):
    """
    PicoScenes Tool class.

    PicoScenes allows instrumentation of multiple Network Devices. This class provides
    a simple low-res python binding to that functionality.
    """

    def __init__(self, frames: list[BaseFrameConfig] | None = None):
        super().__init__()
        self.devices: dict[str, PSNIC | PSUSRP] = {}
        self.bg_process = CaptureProcess()
        self.picoscenes_cmd = get_ps_command_start()

        self.task_builder: dict[str, PicoScenesTaskGroupBuilder] = {}
        self.capture_tasks: Deque[CaptureTask] = deque()

        # Store frames as dictionary for direct access
        self.frames: dict[FrameId, BaseFrameConfig] = {}
        if frames:
            self.register_frames(frames)

    def register_frames(self, frames: list[BaseFrameConfig]):
        """
        Register frames so they can be referenced in transmission
        """
        self.frames.update({config.frame_id(): config for config in frames})

    def _get_builder(self, access_cfg: SshPasswordless | None):
        """ """
        remote_name = access_cfg.remote_ssh_hostname if access_cfg else ""

        if remote_name not in self.task_builder:
            self.task_builder[remote_name] = PicoScenesTaskGroupBuilder(access_cfg)
        return self.task_builder[remote_name]

    def _device_setup(self, device: PSNIC | PSUSRP):
        """
        Add a device for management under PicoScenes
        """
        assert isinstance(device, PSNIC | PSUSRP), (
            "PicoScenes only supports PicoScenes NICs and USRPs."
        )

    def _device_teardown(self, device_id: DeviceId):
        """
        Perform devide teardown
        """
        logger.trace(f"Removing picoscenes device {device_id}")
        device = self.devices[device_id]
        if isinstance(device, PSNIC | PSUSRP):
            device.reset()
        else:
            raise NotImplementedError(
                "Currently only NIC and USRP in Picoscenes supported"
            )

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
        self.capture_tasks.clear()

        # Handle all devices to capture with
        for digest, device in capture_devices.items():
            # First perform setup
            if isinstance(device, PSNIC):
                device.tune(mode=NetworkInterfaceMode.MONITOR, channel=channel)
            elif isinstance(device, PSUSRP):
                device.tune(channel=channel)
            else:
                raise NotImplementedError(
                    "Capture only implemented for NICs and USRPs."
                )

            builder = self._get_builder(get_access_cfg(device))
            builder.add_capture(digest, device, cache_dir, frame)

    def _reap(self) -> list[CaptureResult]:
        """
        Stop the capture and reap the captured data
        """
        if not self.capture_tasks:
            logger.trace("No captures started; nothing to reap.")
            return []

        if self._is_bg_running():
            raise ApiUsageError("PicoScenes capture running; Must stop before reaping!")

        # Aggregate data from all receivers and check which of them produced none
        captures = []

        logger.trace("Capture processes stopped, now reaping data from temp files ...")

        while self.capture_tasks:
            capture = self.capture_tasks.popleft()

            if capture.access_cfg:
                copy_cmd = create_marshall_copy_command(
                    capture.access_cfg, capture.file.path, capture.file.path
                )
                rm_cmd = Command(f"rm {capture.file}").on_remote(capture.access_cfg)
                shell_run(copy_cmd)
                shell_run(rm_cmd)

            if capture.file.empty() or capture.file.path.stat().st_size < 10:
                logger.error(
                    f"Capture file {capture.file.path} is empty - Nothing to reap!"
                )
                capture.file.close()
                continue

            capture_res = CaptureResult(receiver_id=capture.receiver_id)

            # Try to load data and unpack into the capture result if present
            res = load_picoscenes_data(
                capture.file.path, capture.antenna_idxs, capture.stream_idxs
            )
            capture.file.close()

            if not res:
                continue

            csi, subcarrier_idxs = res
            meta = self._get_device_meta(
                antenna_idxs=capture.antenna_idxs,
                stream_idxs=capture.stream_idxs,
                subcarrier_idxs=subcarrier_idxs,
                receiver_name=capture.receiver_name,
            )

            capture_res.csi = csi
            capture_res.meta = meta

            captures.append(capture_res)

        return captures

    def setup_transmit(
        self,
        device_id: DeviceId,
        frame_id: FrameId,
        channel: Channel,
        tx_config: AnyTransmissionConfig,
    ):
        """
        Set up for transmission
        """

        if not isinstance(tx_config, PicoscenesTransmissionConfig):
            raise RuntimeError(
                f"Wrong config type! Should be PicoscenesTransmissionConfig, is {type(tx_config)}"
            )

        if not self.frames:
            raise RuntimeError(
                "Must hand a list of Frame Configurations to constructor to "
                + "enable transmission with PicoScenes."
            )

        device = self.devices[device_id]
        full_frame_spec = self.frames[frame_id]

        # First perform setup
        if isinstance(device, PSNIC):
            device.tune(mode=NetworkInterfaceMode.MONITOR, channel=channel)
        elif isinstance(device, PSUSRP):
            device.tune(channel=channel)
        else:
            raise NotImplementedError(
                "Transmission only implemented for NICs and USRPs so far"
            )

        builder = self._get_builder(get_access_cfg(device))
        builder.add_transmit(device, full_frame_spec, channel, tx_config)

    def _run(self):
        if not self.task_builder:
            raise RuntimeError(
                "Nothing to run was configured; Call setup methods. Must be redone between runs!"
            )

        # Start all picoscenes processes
        for builder in self.task_builder.values():
            task_group = builder.build()

            logger.trace("Starting picoscenes subprocess; Awaiting start keyword")

            # If a USRP is configured to run, it needs a lot of startup time. We need to wait for that.
            start_kw = None
            if any(
                isinstance(self.devices[task.receiver_id], PSUSRP)
                for task in task_group.tasks
            ):
                start_kw = "starts receiving baseband signals"

            self.bg_process.start_process(task_group.ps_cmd, wait_for=start_kw)
            logger.trace(
                "Picoscenes started. Waiting one more second for finished startup"
            )
            time.sleep(1)

            self.capture_tasks += deepcopy(task_group.tasks)

        self.task_builder.clear()

    def _stop(self):
        logger.trace("Stopping PicoScenes capture tasks ...")
        self.task_builder.clear()
        self.bg_process.teardown(ignore_codes=[-2])

    def _is_bg_running(self) -> bool:
        """
        Check whether PicoScenes is still running.

        NOTE: Even the capturing processes should be closed after a transmission
        is finished. Using -q at the end of the picoscenes command causes it to
        exit once finished.
        """
        return self.bg_process.is_running()
