"""
A schedule refers to a sequence of events that should happen over the course
of a sensing procedure.
"""

from __future__ import annotations

import json
from typing import Iterable
from pathlib import Path
from datetime import timedelta
from dataclasses import field, asdict, dataclass

from loguru import logger

from sensession.config import (
    APP_CONFIG,
    Channel,
    FrameId,
    BaseFrameConfig,
    BaseTransmissionConfig,
)
from sensession.devices import DeviceId
from sensession.util.enum import BaseEnum
from sensession.util.serialize import TransmissionConfig, convert


# fmt: off
@dataclass
class TxPrepEvent:
    """
    Event of transmitting with a device so that others can collect
    """
    device_id : DeviceId
    frame_id  : FrameId
    tx_config : TransmissionConfig

    def __post_init__(self):
        if isinstance(self.tx_config, dict):
            self.tx_config = (
                TransmissionConfig(**self.tx_config) if isinstance(self.tx_config, dict) else                  # from deserializing # pylint: disable=not-a-mapping
                TransmissionConfig(self.tx_config) if not isinstance(self.tx_config, TransmissionConfig) else  # from not wrapping in wrapper type
                self.tx_config
            )

@dataclass
class CsiRxPrepEvent:
    """
    Event of starting collection with a set of devices
    """
    device_ids      : list[DeviceId]             # IDs of devices to collect with
    collection_name : str                        # Name of collection for later identification
    filter_frame    : BaseFrameConfig | None = None      # Frame to filter for

@dataclass
class PauseEvent:
    """
    Pausing
    """
    delay : timedelta

@dataclass
class RunEvent:
    """
    Run what was configured!
    """
    collecting_devices   : list[DeviceId]  # All collecting devices
    transmitting_devices : list[DeviceId]  # All transmitting devices
    collection_name      : str             # Name of the collection that's being run
    cooldown             : timedelta       # Time to wait between finishing run and reaping results
    min_capture_num      : int             # Minimum number of captures per receiver to consider run successful
    max_capture_num      : int = int(1e9)  # Maximum capture number

SessionEvent = TxPrepEvent | PauseEvent | CsiRxPrepEvent | RunEvent

class ScheduleEventType(BaseEnum):
    """
    Enumeration of all available transmission config types
    """

    TX = TxPrepEvent
    RX = CsiRxPrepEvent
    PAUSE = PauseEvent
    RUN = RunEvent

@dataclass
class ScheduleEvent:
    """
    Wrapper class for events to force custom serialization
    """
    event      : SessionEvent
    event_type : str = ""

    def __post_init__(self):
        if isinstance(self.event, dict):
            assert self.event_type, (
                "Config type should be automatically deduced; "
                + "Maybe serialization went wrong and didn't write it?"
            )

            self.event = ScheduleEventType[self.event_type].value(**self.event)
        else:
            event_type = type(self.event)
            if not ScheduleEventType.has_value(event_type):
                raise RuntimeError(f"Frame Config type {event_type} unknown.")

            self.event_type = ScheduleEventType(event_type).name

@dataclass
class Schedule:
    """
    A schedule describes the actions to be taken as a sensing session. For example:
      |> StartCollecting(device1)
      |> StartTransmitting(device2, Frame)
      |> StopCollecting
    """
    name      : str                 # Name to distinguish this schedule
    events    : list[ScheduleEvent] # Sequential events in the schedule
    channel   : Channel             # The channel on which to perform the sensing procedure
    cache_dir : Path                # Directory in which to store temp files
    extra_labels : dict = field(default_factory=dict) # A dictionary of additional labels to store

    def __post_init__(self):
        self.events = [ScheduleEvent(**x) if isinstance(x, dict) else x for x in self.events]

        if isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)

        if isinstance(self.channel, dict):
            self.channel = Channel.from_dict(self.channel)

    def to_json(self) -> str:
        """
        Serialize to json
        """
        logger.trace("Converting schedule object to json ...")
        return json.dumps(asdict(self), indent=4, default=convert)
# fmt: on


@dataclass
class BuilderState:
    """
    Builder internal state to help checking allowed builds
    """

    events: list[ScheduleEvent] = field(default_factory=list)
    collection_name: str | None = None
    collecting_devices: set[DeviceId] = field(default_factory=set)
    transmitting_devices: set[DeviceId] = field(default_factory=set)
    is_running: bool = False

    def has_collecting_devices(self) -> bool:
        """
        Check whether state is currently collecting
        """
        return self.collection_name is not None

    def reset(self):
        """
        Reset collecting state
        """
        self.collection_name = None
        self.collecting_devices = set()
        self.transmitting_devices = set()
        self.is_running = False


class ScheduleBuilder:
    """
    Schedule Builder to help build a schedule with a fluent API
    """

    def __init__(
        self,
        name: str,
        channel: Channel,
        cache_dir: Path = APP_CONFIG.cache_dir,
        extra_labels: dict | None = None,
    ):
        self._name = name
        self._channel = channel
        self._cache_dir = Path(cache_dir)

        # Temporary variables for state
        self._state = BuilderState()
        self._extra_labels = extra_labels or {}

    def prepare_collect_csi(
        self,
        device_ids: Iterable[DeviceId],
        collection_name: str = "",
        filter_frame: BaseFrameConfig | None = None,
    ) -> ScheduleBuilder:
        """
        Start collecting CSI with given devices

        Args:
            device_ids      : Devices to put into collection mode
            collection_name : Name to give the collection for later identification
            filter_frame    : An optional frame to filter for
        """
        if isinstance(device_ids, DeviceId):
            device_ids = [device_ids]

        if self._state.is_running:
            raise ValueError(
                "Collections must be stopped before a new one is configured"
            )

        # This function may be called multiple times, but only when the same collection
        # is specified
        if self._state.has_collecting_devices():
            assert self._state.collection_name == collection_name, (
                "Can not start two different collections at one; Please stop the active one first."
            )
        else:
            self._state.collection_name = collection_name

        # Add the new devices into collecting state
        self._state.collecting_devices.update(device_ids)

        # Append start event
        self._state.events.append(
            ScheduleEvent(
                CsiRxPrepEvent(list(device_ids), collection_name, filter_frame)
            )
        )

        return self

    def prepare_transmit(
        self,
        device_id: DeviceId,
        frame_id: FrameId,
        tx_config: BaseTransmissionConfig,
    ) -> ScheduleBuilder:
        """
        Add a frame to be transmitted with given device.

        NOTE: Transmission is currently blocking, hence we also only allow concurrent
        transmission with a single device.

        Args:
            device_id  : Id of transmitter device
            frame_id   : Unique ID of frame to transmit
            tx_config  : Transmission configuration parameters (may be subclass of BaseTransmissionConfig)
        """
        self._state.events.append(
            ScheduleEvent(
                TxPrepEvent(device_id, frame_id, TransmissionConfig(tx_config))
            )
        )
        self._state.transmitting_devices.add(device_id)
        return self

    def wait(self, delay: timedelta) -> ScheduleBuilder:
        """
        Add a delay before the next transmission

        Args:
            delay : Delay to wait for
        """
        self._state.events.append(ScheduleEvent(PauseEvent(delay)))
        return self

    def run(
        self,
        cooldown: timedelta = timedelta(seconds=1),
        min_capture_num: int = 1,
        max_capture_num: int = int(1e9),
    ) -> ScheduleBuilder:
        """
        Run what was set up
        """
        self._state.is_running = True

        # First create a Run event that will run all the tools
        self._state.events.append(
            ScheduleEvent(
                RunEvent(
                    collecting_devices=list(self._state.collecting_devices),
                    transmitting_devices=list(self._state.transmitting_devices),
                    collection_name=self._state.collection_name or "",
                    cooldown=cooldown,
                    min_capture_num=min_capture_num,
                    max_capture_num=max_capture_num,
                )
            )
        )

        self._state.reset()
        return self

    def build(self) -> Schedule:
        """
        Finalize the building process to create the schedule!
        """
        if len(self._state.collecting_devices) != 0:
            logger.warning(
                "Final configured run doesn't seem to have been explicitly started. "
                + "Will add another `run` call automatically..."
            )
            self.run()

        schedule = Schedule(
            name=self._name,
            channel=self._channel,
            events=self._state.events,
            cache_dir=self._cache_dir,
            extra_labels=self._extra_labels,
        )

        # Reset the state to allow reusing the builder object
        self._state = BuilderState()

        return schedule
