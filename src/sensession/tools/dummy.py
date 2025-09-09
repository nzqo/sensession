"""
Dummy CSI Tool
"""

import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from loguru import logger

from sensession.util import ApiUsageError
from sensession.config import Channel, FrameId, Bandwidth, DataRateMode, BaseFrameConfig
from sensession.tools.tool import (
    CsiGroup,
    DeviceId,
    CsiReceiver,
    CsiDataPoint,
    CaptureResult,
    CsiTransmitter,
    AnyTransmissionConfig,
)
from sensession.devices.dummy import DummyConfig, DummyDevice, get_subcarriers


def get_csi(
    config: DummyConfig, bandwidth: Bandwidth, data_rate_mode: DataRateMode
) -> CsiGroup:
    """
    Get a dummy CSI
    """
    num_csi = len(get_subcarriers(bandwidth, data_rate_mode))
    csi = np.ones(
        (len(config.antenna_idxs), len(config.stream_idxs), num_csi),
        dtype=np.complex64,
    )

    timestamp_ns = int(time.time() * 1e9)
    csi_data = [
        CsiDataPoint(
            timestamp=np.uint64(timestamp_ns + i * 1000),
            sequence_num=np.uint16(i),
            csi=csi,
            rssi=np.int8(1),
            antenna_rssi=[np.int8(1)] * len(config.antenna_idxs),
        )
        for i in range(config.num_captures)
    ]

    return csi_data


class Dummy(CsiReceiver, CsiTransmitter):
    """
    Ath9k CSI Tool class.

    This class currently exposes only a low-res API in which a single Device with
    the Ath9k CSI tool can be instrumented.
    """

    def __init__(self):
        super().__init__()
        self.capture_devices: dict[DeviceId, DummyDevice] = {}
        self.tmp_data: dict[DeviceId, CaptureResult] = {}

        self.bandwidth = Bandwidth.TWENTY
        self.data_rate_mode = DataRateMode.HIGH_THROUGHPUT
        logger.trace("Created dummy tool")

    def _device_setup(self, device: DummyDevice):
        """
        Setup of Ath9k device on initial registration

        Args:
            device: Ath9k Network Interface Card.

        Note:
            Supports only one card concurrently for now.
        """
        logger.trace(f"Set up dummy device: {device.get_config().name}")

    def _device_teardown(self, device_id: DeviceId):
        """
        Teardown of device; Trivial for ath9k, nothing to do that's not done by
        the base class already.
        """
        logger.trace(f"Removing dummy device {device_id}")

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
        self.capture_devices = {k: self.devices[k] for k in devices}

        self.bandwidth = channel.bandwidth
        if frame:
            self.data_rate_mode = frame.data_rate_mode

    def _run(self):
        """
        Start capturing
        """
        self.tmp_data.clear()

        for device_id, device in self.capture_devices.items():
            config = device.get_config()
            csi = get_csi(config, self.bandwidth, self.data_rate_mode)
            meta = self._get_device_meta(
                antenna_idxs=config.antenna_idxs,
                stream_idxs=config.stream_idxs,
                subcarrier_idxs=get_subcarriers(self.bandwidth, self.data_rate_mode),
                receiver_name=config.short_name,
            )
            self.tmp_data[device_id] = CaptureResult(
                receiver_id=device_id,
                csi=csi,
                meta=meta,
            )

    def _stop(self):
        """
        Stop capturing
        """
        self.capture_devices.clear()

    def _reap(self, executor: ThreadPoolExecutor | None = None) -> list[CaptureResult]:
        """
        Stop the capture and reap the captured data
        """
        if self.capture_devices:
            raise ApiUsageError(
                "Capture still running! Always call `stop` before `reap`!"
            )

        if executor:
            futures = [executor.submit(self._reap_once, rcv) for rcv in self.tmp_data]
            results = [future.result() for future in futures]
        else:
            results = [self._reap_once(dev) for dev in self.tmp_data]

        self.tmp_data.clear()
        return results

    def _reap_once(self, device_id):
        """
        A single dummy reap
        """
        logger.trace("Sleeping for 5 seconds")
        time.sleep(5)
        return self.tmp_data[device_id]

    def setup_transmit(
        self,
        device_id: DeviceId,
        frame_id: FrameId,
        channel: Channel,
        tx_config: AnyTransmissionConfig,
    ):
        logger.trace(
            f"Set up DUMMY transmission with device {device_id} (won't do anything)"
        )
