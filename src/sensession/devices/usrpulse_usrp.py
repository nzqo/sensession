"""
Definitions for handling a single USRP using the usrpulse library:
> https://dev.seemoo.tu-darmstadt.de/wisense/usrpulse
"""

from __future__ import annotations

import json
import atexit
import threading
from ipaddress import IPv4Address
from dataclasses import dataclass

import requests
import websocket
from tqdm import tqdm
from loguru import logger

from sensession.config import APP_CONFIG, Channel, BaseTransmissionConfig
from sensession.util.shell import shell_run
from sensession.util.remote_access import Command, SshPasswordless
from sensession.util.frame_generation import GeneratedFrameInfo


# fmt: off
@dataclass
class UsrpulseConfig:
    """
    UHD USRP Config
    """

    name     : str           # A name for readable reference
    addr     : IPv4Address   # Address where usrpulse is running
    port     : int           # Port under which to access daemon
    ssh_name : str        # Name for passwordless ssh access

    def device_id(self) -> str:
        """
        Get deterministic device id based on config
        """
        return self.name

    def __post_init__(self):
        if isinstance(self.addr, str):
            self.addr = IPv4Address(self.addr)

    def instantiate_device(self) -> UsrpulseUsrp:
        """
        Create corresponding device to this config
        """
        return UsrpulseUsrp(self)
# fmt: on


class StatusUpdateClient:
    """
    Status update client to receive and log updates from usrpulse transmission
    """

    def __init__(self, uri: str, timeout: int = 300):
        self.progress_bar: tqdm | None = None
        self.timeout: int = timeout
        self.timer: threading.Timer | None = None
        self.ws: websocket.WebSocketApp | None = websocket.WebSocketApp(
            uri, on_message=self.on_message
        )

        atexit.register(self.close)

    def on_message(self, _ws, message):
        """
        Process message
        """
        try:
            # Try to parse the message as JSON
            data = json.loads(message)
        except json.JSONDecodeError:
            return

        steps = 0
        expected = 1
        status = data.get("status", "error")

        # Update progress bar
        if status == "Running":
            steps = data.get("completed_steps", 0)
            expected = data.get("expected_steps", 1)

            # Create progress bar
            if not self.progress_bar:
                self.progress_bar = tqdm(total=expected)

            self.progress_bar.n = steps
            self.progress_bar.refresh()

        # On completion, well, complete.
        if (status == "Completed") or (status == "Running" and (steps == expected)):
            if self.progress_bar:
                self.progress_bar.n = self.progress_bar.total
                self.progress_bar.refresh()
                self.progress_bar.close()
            logger.trace("Completion detected! :)")
            self.close()

    def reset_timer(self):
        """
        Reset internal timeout timer. Timer will cause the opened websocket to
        close within the given period after the blocking run was started.
        """
        if self.timer:
            self.timer.cancel()
        self.timer = threading.Timer(self.timeout, self.on_timeout)
        self.timer.start()

    def on_timeout(self):
        """
        Timeout callback
        """
        logger.error("Timeout reached; Closing websocket, transmission never finished.")
        self.close()

    def close(self):
        """
        Close websocket client
        """
        # Its being called, by whoever, so avoid that it gets called on atexit again.
        atexit.unregister(self.close)

        logger.trace("Closing usrpulse status update client")
        # Close timer
        if self.timer:
            self.timer.cancel()
            self.timer = None
            logger.trace("Timer cancelled")

        # Close websocket
        logger.trace("Closing websocket")
        if self.ws:
            self.ws.close(timeout=0.1)
            self.ws = None
            logger.trace("Websocket closed")

        # Reset state
        self.progress_bar = None

    def run(self):
        """
        Blocking run of websocket until updates are finished.
        """
        if not self.ws:
            raise RuntimeError(
                "Status update client already closed, cant reuse zombie."
            )
        self.reset_timer()
        self.ws.run_forever()


class UsrpulseUsrp:
    """
    Class for a single USRP controlled by raw UHD
    """

    def __init__(self, config: UsrpulseConfig):
        self.config = config
        self.tmp_frame_name: str = ""

    def prepare_frame(self, frame_info: GeneratedFrameInfo):
        """
        Prepare the frame for transmission, i.e. put it on the machine where
        usrpulse daemon is running so it can access it.

        Args:
            frame_info : Info of the frame file to sync over to usrpulse daemon.
        """
        self.trim_usage()
        file = frame_info.frame_file

        # Sanity check file to be used for transmission
        assert file.is_file() and file.stat().st_size > 100, (
            f"Sample file {file} does not exist or is broken."
        )

        # Put file on remote to ready for transmission
        sync_cmd = f"rsync --ignore-existing -z {file} {self.config.ssh_name}:/tmp"
        shell_run(sync_cmd)
        self.tmp_frame_name = f"/tmp/{file.name}"

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
            tx_config   : Transmission configuration
        """
        logger.trace("Configuring usrpulse transmission ...")
        assert channel.bandwidth == frame_info.frame_config.bandwidth, (
            "Channel bandwidth does not match gnerated frame bandwidth!"
        )

        freq = channel.center_freq_hz
        rate = frame_info.frame_config.send_rate_hz
        file = frame_info.frame_file

        if not self.tmp_frame_name.endswith(file.name):
            self.prepare_frame(frame_info)

        # If rate is in MS/s, change scale to Samples/s instead
        if rate < 1000:
            rate = int(rate * 1e6)

        # Generic parameters for the upcoming requests
        base_url = f"http://{self.config.addr}:{self.config.port}"
        headers = {"Content-Type": "application/json"}

        # Build a tuning request (that only runs when parameters change or a
        # 10 minute timeframe has passed).
        tune_request_payload = {
            "out_of_date": {
                "secs": 60000,
                "nanos": 0,
            },
            "parameters": {
                "frequency_hz": freq,
                "gain_db": tx_config.gain,
                "integer_n_tuning": True,
                "lo_offset": 0,
                "sample_rate_hz": rate,
            },
        }

        url = f"{base_url}/tune"
        logger.trace(
            f"Requesting tuning from usrpulse at {url}. Params:\n"
            + f"{json.dumps(tune_request_payload, indent=4)}"
        )
        response = requests.post(
            url,
            json=tune_request_payload,
            headers=headers,
            timeout=APP_CONFIG.shell_timeout_s,
        )
        logger.debug(f"Response from usrpulse: {response.json()}")
        if response.status_code != 202:
            raise RuntimeError(
                f"Received non-accept response from usrpulse: {response.json()}"
            )

        # Use a websocket to receive updates.
        url = f"ws://{self.config.addr}:{self.config.port}/status"
        logger.trace("Connecting to status endpoint ...")
        updates_client = StatusUpdateClient(url)

        # Now transmit
        tx_request_payload = {
            "chunk_size": 10000,  # heuristic
            "sample_file": self.tmp_frame_name,
            "num_reps": tx_config.n_reps,
            "pause_ms": tx_config.pause_ms,
        }

        if tx_config.start_at:
            tx_request_payload["start_time"] = tx_config.start_at

        url = f"{base_url}/transmit"
        logger.trace(
            f"Requesting transmission from usrpulse at {url}. Params:\n"
            + f"{json.dumps(tx_request_payload, indent=4)}"
        )
        response = requests.post(
            url,
            json=tx_request_payload,
            headers=headers,
            timeout=APP_CONFIG.shell_timeout_s,
        )
        logger.debug(f"Response from usrpulse: {response}")
        if response.status_code != 202:
            raise RuntimeError(
                f"Received non-accept response from usrpulse: {response.json()}"
            )

        # Blocking run. This will print out a progress bar of the transmission
        # and stop when the updates have finished
        updates_client.run()
        logger.trace("Usrpulse transmission finished!")

    def trim_usage(self, max_size: int = 10_000_000_000):
        """
        Trim files used for transmissions. This is only relevant if the transmitter
        is on a remote PC to which files are synced for transmission.

        Args:
            max_size : Byte size to trim
        """

        logger.trace(
            f"Trimming usrpulse tmp directory to {max_size} bytes to avoid trashing system."
        )

        cmd = Command(f"./scripts/trim_tmp.sh {max_size}").script_through_remote(
            SshPasswordless(self.config.ssh_name)
        )
        shell_run(cmd)

    def reset(self):
        """
        Clean up device usage
        """
        self.trim_usage(max_size=10_000)
        self.tmp_frame_name = ""

    def get_config(self) -> UsrpulseConfig:
        """
        Get the internal config
        """
        return self.config
