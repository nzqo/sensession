"""
Definitions for handling a single USRP by raw UHD.

More precisely, we use the following project to instrument a USRP transmission:
> https://dev.seemoo.tu-darmstadt.de/wisense/usrp-transmit
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from sensession.config import Channel, BaseTransmissionConfig
from sensession.util.shell import shell_run
from sensession.util.remote_access import Command, SshPasswordless
from sensession.util.frame_generation import GeneratedFrameInfo


# fmt: off
@dataclass
class UhdUsrpConfig:
    """
    UHD USRP Config
    """

    name       : str           # A name for readable reference
    serial_num : str           # Globally unique serial address!
    access     : SshPasswordless | None = None # Optional remote configuration

    def device_id(self) -> str:
        """
        Get deterministic device id based on config
        """
        return self.serial_num

    def __post_init__(self):
        if isinstance(self.access, dict):
            self.access = SshPasswordless(**self.access)

    def instantiate_device(self) -> UhdUsrp:
        """
        Create corresponding device to this config
        """
        return UhdUsrp(self)
# fmt: on


class UhdUsrp:
    """
    Class for a single USRP controlled by raw UHD
    """

    def __init__(self, config: UhdUsrpConfig):
        self.config = config

    def transmit(
        self,
        frame_info: GeneratedFrameInfo,
        channel: Channel,
        tx_config: BaseTransmissionConfig,
    ):
        """
        Transmit

        Args:
            frame_info  : Information on generated frame (group) to transmit
            channel     : Channel on which to transmit samples
            gain        : Transmission gain to use
            tx_config   : Transmission configuration
        """
        assert channel.bandwidth == frame_info.frame_config.bandwidth, (
            "Channel bandwidth does not match gnerated frame bandwidth!"
        )

        self.trim_usage()

        freq = channel.center_freq_hz
        rate = frame_info.frame_config.send_rate_hz
        file = frame_info.frame_file

        # Sanity check file to be used for transmission
        assert file.is_file() and file.stat().st_size > 100, (
            f"Sample file {file} does not exist or is broken."
        )

        # If rate is in MS/s, change scale to Samples/s instead
        if rate < 1000:
            rate = int(rate * 1e6)

        filename = str(file)
        if self.config.access:
            # If on remote, rsync the file to the remote and point to it
            remote_name = self.config.access.remote_ssh_hostname
            sync_cmd = f"rsync {file} {remote_name}:/tmp"
            shell_run(sync_cmd)
            filename = f"/tmp/{file.name}"

        # Assemble command, possibly piping it through SSH onto a remote, if the
        # transmitter is not available locally on the current machine
        shell_cmd = Command(
            f"./scripts/transmit_from_sdr.sh {filename} "
            f"{freq} "
            f"{tx_config.gain} "
            f"{rate} "
            f"{tx_config.n_reps} "
            f"{tx_config.pause_ms} "
            f"serial={self.config.serial_num}"
        ).script_through_remote(self.config.access)

        logger.debug(
            "Transmitting dedicated frame with SDR.\n"
            + f" -- sample file : {file}\n"
            + f" -- frequency   : {freq}\n"
            + f" -- gain        : {tx_config.gain}\n"
            + f" -- rate        : {rate}\n"
            + f" -- num repeats : {tx_config.n_reps}\n"
            + f" -- pause       : {tx_config.pause_ms}ms\n"
        )

        shell_run(shell_cmd)

    def trim_usage(self, max_size: int = 10_000_000_000):
        """
        Trim files used for transmissions. This is only relevant if the transmitter
        is on a remote PC to which files are synced for transmission.

        Args:
            max_size : Byte size to trim
        """
        if not self.config.access:
            return

        logger.trace(
            f"Trimming tmp directory to {max_size} bytes to avoid trashing system."
        )

        cmd = Command(f"./scripts/trim_tmp.sh {max_size}").script_through_remote(
            self.config.access
        )
        shell_run(cmd)

    def reset(self):
        """
        Clean up
        """
        self.trim_usage(max_size=10_000)

    def get_config(self) -> UhdUsrpConfig:
        """
        Get the internal config
        """
        return self.config
