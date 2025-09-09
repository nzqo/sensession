"""
Sensei tool
"""

import time
import threading
from queue import Queue
from typing import cast
from pathlib import Path

import numpy as np
from loguru import logger

from sensession.config import Channel, Bandwidth
from sensession.devices import SenseiDevice, SenseiRemoteConfig
from sensession.tools.tool import (
    DeviceId,
    CsiReceiver,
    CsiDataPoint,
    CaptureResult,
    BaseFrameConfig,
)
from sensession.csi_file_parser.nexmon import get_subcarrier_idxs

try:
    import sensei

    SENSEI_AVAILABLE = True
except ImportError:
    SENSEI_AVAILABLE = False


class Sensei(CsiReceiver):
    """
    Ath9k CSI Tool class.

    This class currently exposes only a low-res API in which a single Device with
    the Ath9k CSI tool can be instrumented.
    """

    def __init__(self):
        if not SENSEI_AVAILABLE:
            raise ImportError("Sensei feature not enabled; Check pyproject.toml")
        super().__init__()

        self.collector = sensei.PyDataCollector()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.data_queues: dict[str, tuple[str, Queue]] = {}
        self.tmp_channel: Channel | None = None

    def _device_setup(self, device: SenseiDevice):
        logger.trace("Performing device setup for sensei device")
        config = device.config
        if not isinstance(config, SenseiRemoteConfig):
            raise NotImplementedError("Only nexmon live implemented")

        addr = f"{config.addr}:{config.port}"
        try:
            self.collector.setup_from_remote(
                config.short_name,
                config.remote_resource_id,
                addr,
                config.connection_type.to_sensei(),  # Connection type tag (e.g. TCP)
                config.source_cfg.to_sensei(),  # Source type tag (e.g. nexmon or iwl)
            )
        except ValueError as e:
            if "connection refused" in str(e).lower():
                raise RuntimeError(
                    f"Couldn't connect to {config.short_name} at "
                    + f"{addr}/{config.remote_resource_id}. "
                    + "Likely no remote sensei instance running or firewall problem."
                )
            raise RuntimeError("Couldn't connect to remote sensei server") from e
        self.data_queues[config.device_id()] = (config.short_name, Queue())

    def _device_teardown(self, device_id: DeviceId):
        """
        Teardown of device
        """
        logger.trace(f"Removing sensei device {device_id}")
        del self.data_queues[device_id]

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
        self.tmp_channel = channel
        for receiver_id in devices:
            self.devices[receiver_id].tune(channel, frame, self.collector)

        # Ensure queues are emptied and dont contain leftovers from prior runs.
        logger.trace("Emptying data point queues")
        for _, queue in self.data_queues.values():
            queue.queue.clear()

    def _run(self):
        self.stop_event.clear()  # Reset the stop event
        self.thread = threading.Thread(target=self.collect)
        self.thread.start()

    def _stop(self):
        logger.trace("Sleeping for 0.5 seconds to ensure everything is streamed")
        time.sleep(0.5)
        self.stop_event.set()
        if self.thread:
            self.thread.join()
            self.thread = None

        logger.trace("Sensei collection stopped.")

    def _reap(self) -> list[CaptureResult]:
        """
        Stop the capture and reap the captured data
        """
        logger.trace("Reaping data from sensei ...")
        assert self.tmp_channel, "Temporary channel must be present until after reap"
        res = []
        for receiver_id, (name, queue) in self.data_queues.items():
            csi_list = []
            while not queue.empty():
                csi_list.append(queue.get())
                queue.task_done()

            config = self.devices[receiver_id].config

            # Note: `name` is config.short_name
            meta = self._get_device_meta(
                antenna_idxs=config.antenna_idxs,
                stream_idxs=config.stream_idxs,
                subcarrier_idxs=get_sensei_subcarrier_idxs(
                    name, self.tmp_channel.bandwidth
                ),
                receiver_name=name,
            )

            if len(csi_list) == 0:
                continue

            res.append(
                CaptureResult(
                    receiver_id=receiver_id,
                    csi=csi_list,
                    meta=meta,
                )
            )

        logger.trace("Finished reaping!")
        return res

    def collect(self):
        """
        Collect data from sensei
        """
        resource_ids = [resource_id for (resource_id, _) in self.data_queues.values()]
        logger.trace(f"Starting collection with sensei for: {resource_ids}")

        # We will keep running until the stop event is set and no more items are buffered.
        running = {res: True for res in resource_ids}

        # Start collection!
        for resource_id in resource_ids:
            self.collector.start(resource_id)

        while any(running.values()):
            for resource_id, queue in self.data_queues.values():
                if data := self.collector.poll(resource_id):
                    queue.put(
                        CsiDataPoint(
                            timestamp=np.uint64(data.timestamp * 1e9),
                            sequence_num=np.uint16(data.sequence_number),
                            csi=data.csi,
                            antenna_rssi=cast(list[np.int8], data.rssi.tolist()),
                            rssi=estimate_rssi(data.rssi),
                        )
                    )

                elif self.stop_event.is_set():
                    running[resource_id] = False

        # And stop of course
        logger.trace("Stopping sensei collector...")
        for resource_id in resource_ids:
            self.collector.stop(resource_id)


def estimate_rssi(antenna_rssi: np.ndarray):
    """
    Estimate combined rssi given RSSI from a set of RX chains
    """
    return (
        (10 * np.log10(np.mean(np.power(10, antenna_rssi / 10)))).astype(int).tolist()
    )


def get_sensei_subcarrier_idxs(receiver_name: str, bandwidth: Bandwidth) -> list[int]:
    """
    Get the subcarrier indices for the given receiver
    """
    # NOTE: Sensei should somehow announce the type of the device.
    # Currently we hardcode this by name
    if "iwl" in receiver_name:
        if bandwidth == Bandwidth.TWENTY:
            return list(range(-28, -1, 2)) + list(range(-1, 28, 2)) + [28]
        if bandwidth == Bandwidth.FOURTY:
            return list(range(-58, 59, 4))
        raise ValueError("Unsupported bandwidth for iwl")

    subcs = cast(list[int], get_subcarrier_idxs(bandwidth).tolist())
    return subcs
