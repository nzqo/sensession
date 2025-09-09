"""
Atheros QCA CSI Tool
"""

import time
from pathlib import Path

from loguru import logger

from sensession.config import Channel
from sensession.devices import Ath9kNIC, NetworkInterfaceMode
from sensession.tools.tool import DeviceId, CsiReceiver, CaptureResult, BaseFrameConfig
from sensession.util.shell import shell_run
from sensession.util.temp_file import TempFile
from sensession.util.exceptions import ApiUsageError
from sensession.util.remote_access import Command, create_marshall_copy_command
from sensession.util.capture_process import CaptureProcess
from sensession.csi_file_parser.ath9k import load_ath9k_data


def _reload_driver(device):
    """
    Reload driver to activate it
    """

    access_cfg = device.config.remote_access_cfg
    logger.trace(
        "Reloading ath9k driver. \n"
        + f"Repo path     : {device.config.repo_path} \n"
        + f"Access config : {access_cfg} \n"
    )

    reload_driver = Command(
        f"{device.config.repo_path}/driver/helper-scripts/load_custom_driver.sh --y"
    ).on_remote(access_cfg)

    shell_run(f"{reload_driver}")


class Ath9k(CsiReceiver):
    """
    Ath9k CSI Tool class.

    This class currently exposes only a low-res API in which a single Device with
    the Ath9k CSI tool can be instrumented.
    """

    def __init__(self):
        super().__init__()
        self.bg_process = CaptureProcess()
        self.tmp_file: TempFile | None = None

        self.tmp_capture_cmd: str = ""
        self.tmp_cleanup_cmd: str = ""

    def _device_setup(self, device: Ath9kNIC):
        """
        Setup of Ath9k device on initial registration

        Args:
            device: Ath9k Network Interface Card.

        Note:
            Supports only one card concurrently for now.
        """
        if len(self.devices) != 0:
            raise NotImplementedError(
                "Device already set; Ath9k only supports single device currently."
            )

        if not isinstance(device, Ath9kNIC):
            raise NotImplementedError("Ath9k only supports Ath9kNIC devices.")

        # _reload_driver(device)

    def _device_teardown(self, device_id: DeviceId):
        """
        Teardown of device; Trivial for ath9k, nothing to do that's not done by
        the base class already.
        """
        logger.trace(f"Removing ath9k device {device_id}")

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
        # Ensure background processes are cleaned up!
        if not self.bg_process.processes.empty():
            self._stop()
            time.sleep(0.1)

        device: Ath9kNIC = list(self.devices.values())[0]
        config = device.get_config()

        # Ensure that no temp file is set from previous runs. Otherwise, appending might
        # happen
        if self.tmp_file:
            self.tmp_file.close()
            self.tmp_file = None

        if not frame:
            logger.warning(
                "QCA cards in monitor mode will only capture frames addressed "
                + "to their actual MAC address. Ath9k Tool was started without "
                + "any frame to listen for."
            )
        else:
            assert config.mac_address == frame.receiver_address, (
                "QCA cards can only capture frames addressed to their actual MAC address. \n"
                + f"Expected: {config.mac_address}"
                + f"Got: {frame.receiver_address}"
            )

        access_cfg = config.access

        # Start by putting device into monitor mode
        device.tune(NetworkInterfaceMode.MONITOR, channel)

        # File handling: Remember which file data is in for stop_and_reap
        file_name = f"{config.short_name}.log"
        file_dir = cache_dir
        self.tmp_file = TempFile(file_name, file_dir)

        # If ath9k runs on remote, prepend the proper marshalling
        repo_path = config.repo_path

        if access_cfg:
            # Stream into file on remote and then sync back..
            tmp_file = f"/tmp/{file_name}"
            capture_cmd = f"sudo {repo_path}/extractor/build/csi-extractor {tmp_file}"
            capture_cmd = Command(capture_cmd).on_remote(access_cfg, pseudo_tty=False)
            cleanup_cmd = create_marshall_copy_command(access_cfg, tmp_file, file_dir)

            # NOTE: Remote process is actually kept alive after local background capture process
            # is killed when not using pseudo_tty. That is because SIGINT is not forwarded through
            # ssh. Here, we stash a cleanup command to hackily kill the process on the remote.
            # This command shall only be executed at cleanup time
            shutdown_cmd = Command("sudo pkill csi-extractor").on_remote(
                access_cfg, pseudo_tty=False
            )
            self.bg_process.start_process(
                shell_command=None, cleanup_command=shutdown_cmd
            )
        else:
            # If we run locally, no need to perform any marshalling. Because of sudo
            # cleanup permission issues with subprocess, we wrap capturing in a shell
            # script and clean up with a cleanup command.
            capture_cmd = f"./scripts/capture_qca.sh {repo_path} {self.tmp_file.path}"
            cleanup_cmd = "sudo pkill csi-extractor"

        self.tmp_capture_cmd = capture_cmd
        self.tmp_cleanup_cmd = cleanup_cmd

    def _run(self):
        """
        Start capturing
        """
        if not self.tmp_capture_cmd:
            raise ApiUsageError("No capture command found; Call `setup_capture`")
        if not self.tmp_cleanup_cmd:
            raise ApiUsageError("No cleanup command found; Call `setup_capture`")
        if not self.devices:
            raise ApiUsageError("No device was registered, Call `add_device`")

        self.bg_process.start_process(
            self.tmp_capture_cmd,
            cleanup_command=self.tmp_cleanup_cmd,
        )

    def _stop(self):
        """
        Stop capturing
        """
        # Stop capture processes
        # NOTE:
        # pkill on teardown returns 1 if csi-extractor was already stopped.
        self.bg_process.teardown(ignore_codes=[1, 255])

        self.tmp_capture_cmd = ""
        self.tmp_cleanup_cmd = ""

    def _reap(self) -> list[CaptureResult]:
        """
        Stop the capture and reap the captured data
        """
        if not isinstance(self.tmp_file, TempFile):
            raise ApiUsageError("No temp file configured; Call to `run` missing?")

        assert self.devices, "Device required"
        if len(self.devices) != 1:
            raise NotImplementedError(
                "Ath9k currently supports only single-device mode."
            )

        device_id, device = list(self.devices.items())[0]

        # Load data from the temp file
        tmp_file = self.tmp_file
        logger.trace(f"Reaping data for qca (from file : {tmp_file.path})")

        # Check that it is not empty (or contains just soe header data)
        if tmp_file.empty() or tmp_file.path.stat().st_size < 10:
            tmp_file.close()
            return []

        capture_res = CaptureResult(receiver_id=device_id)

        res = load_ath9k_data(
            tmp_file.path, device.config.antenna_idxs, device.config.stream_idxs
        )
        tmp_file.close()

        if not res:
            return []

        # pylint: disable=duplicate-code
        # I know its "duplicate", but its just a single function call thats long..
        # Unpack parsing result
        csi, subcarrier_idxs = res

        # Extract a corresponding Meta struct
        meta = self._get_device_meta(
            antenna_idxs=device.config.antenna_idxs,
            stream_idxs=device.config.stream_idxs,
            subcarrier_idxs=subcarrier_idxs,
            receiver_name=device.config.short_name,
        )

        capture_res.csi = csi
        capture_res.meta = meta

        return [capture_res]
