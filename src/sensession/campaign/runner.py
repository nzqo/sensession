"""
Implementation of the CampaignRunner which, well, runs the steps specified in a campaign.
"""

from __future__ import annotations

import time
import traceback
from typing import TypeVar, get_args
from collections import Counter
from dataclasses import field, dataclass
from collections.abc import Callable  # pylint: disable=ungrouped-imports

import polars as pl
from loguru import logger

from sensession.tools import (
    DEVICE_REGISTRY,
    Tool,
    ToolType,
    CsiReceiver,
    CaptureResult,
    CsiTransmitter,
    get_tool_type,
)
from sensession.config import APP_CONFIG, FrameId
from sensession.devices import DeviceId, DeviceConfigType
from sensession.database import Database
from sensession.util.exceptions import (
    MatlabError,
    NoDataError,
    SubprocessError,
    MultipleDataError,
    InsufficientDataError,
)
from sensession.campaign.campaign import Campaign
from sensession.campaign.schedule import (
    RunEvent,
    Schedule,
    PauseEvent,
    TxPrepEvent,
    SessionEvent,
    CsiRxPrepEvent,
)
from sensession.util.frame_generation import FrameGenerator, USRPFrameConfig

K = TypeVar("K")
V = TypeVar("V")


def extract_frame_schedule(campaign: Campaign):
    """
    Extract frames in access order according to schedules

    Args:
        campaign : The campaign, containing a sequential list of schedules
    """
    return [
        event.event.frame_id
        for schedule in campaign.schedules
        for event in schedule.events
        if isinstance(event.event, TxPrepEvent)
    ]


def invert_map(data: dict[K, V]) -> dict[V, list[K]]:
    """
    Invert mapping

    Args:
        data : Dictionary

    Example:
        {a : 1, b : 1, c : 2} -> {1 : [a, b], 2 : [c]}
    """
    inv_map: dict[V, list[K]] = {}
    for key, value in data.items():
        inv_map[value] = inv_map.get(value, []) + [key]
    return inv_map


@dataclass
class CollectionResult:
    """
    Results from a single collection sequence of possibly multiple devices
    """

    name: str
    data: list[CaptureResult]  # List of data from runs


@dataclass
class ScheduleResult:
    """
    Result aggregates for a schedule; i.e. one or multiple collections
    """

    name: str
    data: list[CollectionResult] = field(default_factory=list)
    error: str = ""
    extra_labels: dict = field(default_factory=dict)


@dataclass
class CampaignResult:
    """
    Result aggregates for a whole campaign
    """

    name: str  # Campaign name
    data: list[ScheduleResult] = field(default_factory=list)


def get_participating_devices(schedule: Schedule) -> set[DeviceId]:
    """
    Get all devices that are participating in a schedule
    """
    devices = set()
    for event_wrapper in schedule.events:
        event = event_wrapper.event
        if isinstance(event, CsiRxPrepEvent):
            devices.update(event.device_ids)
        if isinstance(event, TxPrepEvent):
            devices.add(event.device_id)

    return devices


def get_used_frames(schedule: Schedule) -> set[FrameId]:
    """
    Get all frames used in a schedule
    """
    return {
        event.event.frame_id
        for event in schedule.events
        if isinstance(event.event, TxPrepEvent)
    }


def _campaign_extract(
    campaign: Campaign,
    results: CampaignResult,
    predicate: Callable[[ScheduleResult], bool],
) -> Campaign:
    """
    Extract a sub-campaign based on a predicate on its containing ScheduleResults.
    """
    new_schedules = []
    relevant_devices = set()
    relevant_frames = set()

    for schedule, result in zip(campaign.schedules, results.data):
        if predicate(result):
            new_schedules.append(schedule)

            # In case some frames or devices are not occuring in affected schedules, we don't
            # need them in the new Campaign. For that reason, we track which ones _are_ used.
            relevant_devices.update(get_participating_devices(schedule))
            relevant_frames.update(get_used_frames(schedule))

    # We now know the relevant IDs of frames and device configs; Retrieve the configs
    cfgs = [
        cfg
        for cfg in campaign.device_cfgs
        if cfg.config.device_id() in relevant_devices
    ]

    frames = []
    for frame in campaign.frames:
        if frame.config.frame_id() in relevant_frames:
            frames.append(frame)

            # NOTE: This ensures that we don't duplicate frames, even if they are duplicated
            # in the original list.
            relevant_frames.remove(frame.config.frame_id())

    # Build the campaign
    return Campaign(
        name=campaign.name, schedules=new_schedules, frames=frames, device_cfgs=cfgs
    )


def extract_succeeded_campaign(campaign: Campaign, results: CampaignResult) -> Campaign:
    """
    Create a new campaign containing just successful runs of a given one

    Args:
        campaign : The original campaign that was run
        results  : The results that came out of this run

    Returns:
        A new campaign with just the successfully ran schedules
    """
    return _campaign_extract(
        campaign, results, predicate=lambda sched_result: not sched_result.error
    )


def extract_failed_campaign(campaign: Campaign, results: CampaignResult) -> Campaign:
    """
    Create a new campaign from failed runs of a given one.

    Args:
        campaign : The original campaign that was run
        results  : The results that came out of this run

    Returns:
        A new campaign with just the non-successfully ran schedules
    """
    return _campaign_extract(
        campaign, results, predicate=lambda sched_result: bool(sched_result.error)
    )


class CampaignRunner:
    """
    Class to manage running of a campaign
    """

    def __init__(self, campaign: Campaign, retry_count: int = 0):
        self.tools: dict[ToolType, Tool] = {}
        self.campaign = campaign
        self.retry_count = retry_count

        self.result = CampaignResult(name=self.campaign.name)

        # For easier frame access, convert frame list to a dictionary with keys
        # being the frame IDs
        self.frames = {
            frame_cfg.config.frame_id(): frame_cfg.config
            for frame_cfg in campaign.frames
        }

        # To keep track in which retry we are
        self.retry_idx: int = 0
        # to keep track which schedule we need to execute
        self.current_schedule_idx: int = 0

        self.errors: list[dict] = []

    def get_errors(self) -> list[dict]:
        """
        Remember errors in a file for later inspection
        """
        return self.errors

    def __enter__(self) -> CampaignRunner:
        logger.trace("Preparing for running campaigns ...")
        self._prepare_campaign(self.campaign)
        return self

    def __exit__(self, exc_type, exc_value, tb) -> bool:
        logger.trace("Wrapping up campaign runner ...")
        self._finish_campaign()

        if exc_type is not None:
            trace = "".join(traceback.format_exception(exc_type, exc_value, tb))
            logger.error(
                f"Encountered exception in CampaignRunner context! Traceback: {trace}"
            )

            return False

        logger.trace("Campaign Runner finished.")
        return True

    def __iter__(self) -> CampaignRunner:
        return self

    def __next__(self) -> ScheduleResult:
        """
        Run next schedule in line
        """
        if self.current_schedule_idx >= len(self.campaign.schedules):
            raise StopIteration()

        schedule = self.campaign.schedules[self.current_schedule_idx]
        logger.trace(
            "Executing schedule \n"
            + f" -- name         : {schedule.name} \n"
            + f" -- num steps    : {len(schedule.events)}\n"
            + f" -- extra labels : {schedule.extra_labels}"
        )

        result = ScheduleResult(
            name=schedule.name, data=[], extra_labels=schedule.extra_labels
        )

        try:
            # Execute all events in the schedule
            for event_wrapper in schedule.events:
                event = event_wrapper.event
                event_res = self._handle_event(event, schedule)
                if event_res is not None:
                    result.data.append(event_res)

        except* (
            RuntimeError,
            SubprocessError,
            NoDataError,
            InsufficientDataError,
            MatlabError,
        ) as e:
            tb = "".join(traceback.format_exception(e))
            logger.critical(f"Session failed due to a tool error! Exception: {tb}")

            tb = traceback.format_exc()
            result = ScheduleResult(
                name=schedule.name,
                data=[],
                error=tb,
                extra_labels=schedule.extra_labels,
            )

            self.errors.append(
                {
                    "schedule_name": schedule.name,
                    "event": str(type(event)),
                    "error": str(e),
                    "traceback": tb,
                }
            )

            # Handle retries -- Either set up for retrying the same schedule again ..
            # Or advance to the next schedule in line.
            if self.retry_idx < self.retry_count:
                logger.info(
                    "Retrying same schedule again next run "
                    + f" ({self.retry_idx} / {self.retry_count})"
                )
                self.retry_idx += 1

                # decrease to offset stepping to next schedule
                self.current_schedule_idx -= 1
            else:
                self.retry_idx = 0

            if APP_CONFIG.wait_after_logs:
                time.sleep(APP_CONFIG.wait_after_logs)
        finally:
            self.current_schedule_idx += 1

        return result

    def run(self) -> CampaignResult:
        """
        Run the campaign
        """
        logger.trace(f"Running campaign {self.campaign.name}")

        for schedule_res in self:
            self.result.data.append(schedule_res)

        return self.result

    def _prepare_campaign(self, campaign: Campaign):
        """
        Instantiate campaign objects from configuration spec

        Args:
            campaign : Description of the campaign to run
        """
        self._instantiate_tools(campaign)
        self._instantiate_devices(campaign)

    def _finish_campaign(self):
        """
        Tie up loose ends; end the campaign

        Args:
            campaign : Description of the campaign to run
        """
        self._clear_tools()

    def _handle_event(
        self, event: SessionEvent, schedule: Schedule
    ) -> CollectionResult | None:
        """
        Handle events of different types
        """
        result = None

        if isinstance(event, PauseEvent):
            self._handle_pause(event)
        elif isinstance(event, TxPrepEvent):
            self._handle_tx_setup(event, schedule)
        elif isinstance(event, CsiRxPrepEvent):
            self._handle_csi_rx_setup(event, schedule)
        elif isinstance(event, RunEvent):
            result = self._handle_run(event)
        else:
            raise TypeError(f"Event of type {type(event)} is not handled!")

        return result

    def _instantiate_tools(self, campaign: Campaign):
        """
        Instantiate all tools that are required to manage devices in the campaign

        Args:
            campaign : Campaign to run
        """
        config_types = set(
            DeviceConfigType[x.config_type] for x in campaign.device_cfgs
        )
        for cfg_type in config_types:
            tool_type = get_tool_type(cfg_type)

            if cfg_type in [DeviceConfigType.UHD_USRP, DeviceConfigType.USRPULSE_USRP]:
                # For the transmitter SDRs, we need to create IQ frames to send.
                # To do so, we instantiate a frame generator for these tools and
                # pass it in the constructor
                usrp_frame_manager = FrameGenerator()
                for frame in campaign.frames:
                    if isinstance(frame.config, get_args(USRPFrameConfig)):
                        usrp_frame_manager.add_frame(frame.config)
                    else:
                        raise NotImplementedError(
                            "Campaign does not support any other transmission than UHD for now"
                        )

                frames = extract_frame_schedule(campaign)
                logger.trace(f"Enabling frame pre-generation with schedule : {frames}")
                usrp_frame_manager.enable_pregen(frames)

                self.tools[tool_type] = tool_type.value(usrp_frame_manager)
            else:
                # Default construct the tool
                self.tools[tool_type] = tool_type.value()

    def _instantiate_devices(self, campaign: Campaign):
        """
        Instantiate the devices specified in the campaign

        Args:
            campaign : Campaign to run
        """
        logger.trace("Instantiating specified device for campaign ...")
        for config in campaign.device_cfgs:
            device = config.config.instantiate_device()
            cfg_type = DeviceConfigType[config.config_type]
            tool_type = get_tool_type(cfg_type)
            self.tools[tool_type].add_device(device)

    def _clear_tools(self):
        """
        Clear out all tools
        """
        logger.trace("Clearing out all tools ...")
        for tool in self.tools.values():
            tool.reset()
        self.tools.clear()

    def _handle_pause(self, event: PauseEvent):
        """
        Perform a pause
        """
        logger.trace("Executing Pause event ...")
        time.sleep(event.delay.total_seconds())

    def _handle_tx_setup(self, event: TxPrepEvent, schedule: Schedule):
        """
        Execute transmission phase
        """
        logger.trace("Executing Transmission event ...")

        device_id = event.device_id
        tool = DEVICE_REGISTRY.get(device_id).tool

        if not isinstance(tool, CsiTransmitter):
            raise RuntimeError(
                "Transmission event caught, but tool does not implement `CsiTransmitter` interface"
            )

        tool.setup_transmit(
            device_id,
            event.frame_id,
            channel=schedule.channel,
            tx_config=event.tx_config.config,
        )

    def _handle_csi_rx_setup(self, event: CsiRxPrepEvent, schedule: Schedule):
        """
        Execute the start of a collection phase
        """
        logger.trace("Executing Collection Start Event ...")

        tools = {
            device_id: DEVICE_REGISTRY.get(device_id).tool
            for device_id in event.device_ids
        }

        tool_device_map = invert_map(tools)

        for tool, device_ids in tool_device_map.items():
            if not isinstance(tool, CsiReceiver):
                raise RuntimeError(
                    "Transmission event caught, but tool does not implement `CsiTransmitter` interface"
                )

            cache_base_dir = schedule.cache_dir
            cache_dir = (
                cache_base_dir
                / f"sched_{schedule.name}"
                / f"coll_{event.collection_name}"
            )
            cache_dir.mkdir(exist_ok=True, parents=True)
            try:
                tool.setup_capture(
                    device_ids,
                    schedule.channel,
                    cache_dir=cache_dir,
                    filter_frame=event.filter_frame,
                )
            except Exception as e:  # pylint: disable=broad-exception-caught
                # Stop all started thus far and raise exceptions
                logger.error(
                    "Exception during tool setup occured; Stopping tools and reraising."
                )
                stop_errs = safe_stop_tools(list(tools.values()))
                if stop_errs:
                    raise ExceptionGroup(
                        "Tool preparation failed; Stop failed as well: ", stop_errs
                    ) from e
                raise RuntimeError("Tool run failed!") from e

    def _handle_run(self, event: RunEvent) -> CollectionResult:
        """
        Handle a collection execution (running and stopping!)
        """
        logger.trace(
            f"Executing Collection Run Event for collection name: {event.collection_name} ..."
        )

        # To get the affected tools, we can invert the map of devices
        tools = invert_map(
            {
                device_id: DEVICE_REGISTRY.get(device_id).tool
                for device_id in event.collecting_devices + event.transmitting_devices
            }
        )

        # ------------------------------------------------------------------
        # Run, wait, stop.
        # NOTE: We want to make sure stop is called for each of the receiver
        # to minimize potential problems since we change persistent hardware
        # state here. Potential errors are collected and rethrown afterwards.
        for tool in tools:
            try:
                tool.run()
            except Exception as e:  # pylint: disable=broad-exception-caught
                # Stop all started thus far and raise exceptions
                logger.error(
                    "Exception during tool run occured; Stopping tools and reraising."
                )
                stop_errs = safe_stop_tools(list(tools))
                if stop_errs:
                    raise ExceptionGroup(
                        "Tool run failed; Stop failed as well: ", stop_errs
                    ) from e
                raise RuntimeError("Tool run failed!") from e

        # Cooldown
        seconds = event.cooldown.total_seconds()
        logger.trace(
            f"Run finished, waiting {seconds}s before stopping (to clear buffers)"
        )
        time.sleep(seconds)

        # Stop
        errors = safe_stop_tools(list(tools))
        if errors:
            raise ExceptionGroup("Exception(s) occured during tool stop: ", errors)

        # Reap results from receivers
        logger.trace("Run finished; Reaping data ...")
        receivers = [tool for tool in tools if isinstance(tool, CsiReceiver)]
        data = reap(receivers)

        check_results_and_log(data, event)
        return CollectionResult(event.collection_name, data)

    def store_results(self, database: Database):
        """
        Persist data in database.
        """
        for sched_res in self.result.data:
            if sched_res.error:
                logger.warning(f"Schedule {sched_res.name} failed; Consider rerunning.")
                continue

            logger.trace(
                f"Persisting data from schedule {sched_res.name}: \n"
                + f" -- Extra Metadata : {sched_res.extra_labels}"
            )

            for collection_res in sched_res.data:
                for data in collection_res.data:
                    database.add_data(
                        data,
                        collection_name=(collection_res.name, pl.String),
                        schedule_name=(sched_res.name, pl.String),
                        campaign_name=(self.campaign.name, pl.String),
                        **sched_res.extra_labels,
                    )

            database.add_errors(self.errors)
            self.errors.clear()


def reap(receivers: list[CsiReceiver]) -> list[CaptureResult]:
    """
    Reap data from a list of receivers
    """
    results = []
    exceptions = []

    for receiver in receivers:
        # NOTE: We catch general exceptions which are reraised in an exception group.
        # Down the line, we handle explicit exceptions, or expect a program crash.
        try:
            results.extend(receiver.reap())
        except Exception as e:  # pylint: disable=broad-exception-caught
            exceptions.append(e)

    if exceptions:
        raise ExceptionGroup("Errors occurred while reaping:", exceptions)

    return results


def check_results_and_log(data: list[CaptureResult], event: RunEvent):
    """
    Checks for consistency of received data with the specified event.
    If problems are detected, such as a receiver not having collected
    sufficient data, an error is thrown.

    Also logs number of collected data points per receiver.
    """
    # Log how much data was collected in this run
    collected_nums = f"Data collected in this run ({event.collection_name}): \n"
    for res in data:
        name = DEVICE_REGISTRY.get_short_name(res.receiver_id)
        collected_nums += f"  -- {name}: {len(res.csi) if res.csi else 0}\n"

    logger.info(collected_nums)
    if APP_CONFIG.wait_after_logs:
        time.sleep(APP_CONFIG.wait_after_logs)

    # Check whether all the configured receivers have collected data
    # NOTE: collecting_devices are the ones we desire data from.
    captured_receivers = [res.receiver_id for res in data if res.csi]
    failed_receivers = set(event.collecting_devices) - set(captured_receivers)
    if failed_receivers:
        names = [DEVICE_REGISTRY.get(dev).config.name for dev in failed_receivers]
        msg = f"Receivers {names} failed to report ANY data!"
        raise NoDataError(msg)

    # Check if any of the receivers failed to meet the desired quota
    failed_receivers = set(
        receiver_data.receiver_id
        for receiver_data in data
        if receiver_data.csi is not None
        and (
            len(receiver_data.csi or []) < event.min_capture_num
            or len(receiver_data.csi or []) > event.max_capture_num
        )
    )
    if failed_receivers:
        names = [DEVICE_REGISTRY.get(dev).config.name for dev in failed_receivers]
        msg = f"Receivers {names} failed to meet capture quota of {event.min_capture_num}!"
        raise InsufficientDataError(msg)

    # Compute whether any of the receivers reported more than one data group
    dupes = set(Counter(captured_receivers) - Counter(set(captured_receivers)))
    if dupes:
        raise MultipleDataError(f"Receivers {dupes} reported multiple data groups.")


def safe_stop_tools(tools: list[Tool]) -> list[Exception]:
    """
    Stop all tools, raising an error group of all that have failed to stop.
    """
    errors = []
    for tool in tools:
        try:
            tool.stop()
        except Exception as e:  # pylint: disable=broad-exception-caught
            errors.append(e)

    # ------------------------------------------------------------------
    # If any errors occured, raise them.
    return errors
