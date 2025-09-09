"""
A campaign refers to a larger set of experiments. This is a convenience abstraction
to allow defining longer running experiments.

By construction campaigns from failed scheduled sessions, we also have a good idea
of experiments to rerun to finish a defined campaign. For this reason, we implement
serialization and deserialization of campaigns.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from dataclasses import asdict, dataclass

from loguru import logger

from sensession.util.enum import BaseEnum
from sensession.util.serialize import FrameConfig, DeviceConfig, convert, deconvert
from sensession.campaign.schedule import Schedule


#######################################################################################
## Implementation of a campaign
#######################################################################################
# fmt: off
@dataclass
class Campaign:
    """
    A campaign refers to a group of sessions with the same participating receivers.
    """

    schedules   : list[Schedule]      # Schedules to run
    frames      : list[FrameConfig]   # Configs of frames
    device_cfgs : list[DeviceConfig]  # Configs of devices
    name        : str = ""            # A name for the campaign

    def to_json(self) -> str:
        """
        Serialize to json
        """
        logger.trace("Serializing Campaign object to json ...")
        return json.dumps(asdict(self), indent=4, default=convert)

    @staticmethod
    def from_json(json_str: str) -> Campaign:
        """
        Deserialize from json
        """
        logger.trace("Deserializing Campaign object from json ...")
        data = json.loads(json_str, object_hook=deconvert)
        return Campaign(**data)

    def is_empty(self) -> bool:
        """
        Check whether campaign is empty/trivial, i.e. contains no schedules to execute
        """
        return len(self.schedules) == 0

    def __post_init__(self):
        self.schedules = [
            Schedule(**sched) if isinstance(sched, dict) else
            sched for sched in self.schedules
        ]
        self.frames = [
            FrameConfig(**frame) if isinstance(frame, dict) else          # from deserialization
            FrameConfig(frame) if not isinstance(frame, FrameConfig) else # from not wrapping config in FrameConfig wrapper type
            frame                                                         # if type is fine
            for frame in self.frames
        ]
        self.device_cfgs = [
            DeviceConfig(**cfg) if isinstance(cfg, dict) else           # from deserialization
            DeviceConfig(cfg) if not isinstance(cfg, DeviceConfig) else # from not wrapping config in DeviceConfig wrapper type
            cfg for cfg in self.device_cfgs
        ]
# fmt: on


class SerializationMode(str, BaseEnum):
    """
    Supported serialization modes for writing to disk.
    """

    JSON = ".json"
    PICKLE = ".pickle"


def write_to_disk(
    campaign: Campaign,
    file_path: Path,
    mode: SerializationMode = SerializationMode.JSON,
    append: bool = False,
):
    """
    Write the campaign to a file

    NOTE: Will append a corresponding path ending depending on the write mode.

    Args:
        campaign  : Campaign to serialize and persist
        file_path : Path to file in which to store campaign
        mode      : Mode with which to perform serialization
        append    : If true, will append if there are already campaigns in the file
    """
    # Append file ending corresponding to serialization mode file
    file_path = file_path.parent / f"{file_path.stem}{mode.value}"
    file_path.parent.mkdir(exist_ok=True, parents=True)

    logger.trace(
        f"Writing campaign {campaign.name} to file \n"
        + f" -- file      : {file_path} \n"
        + f" -- mode      : {mode.name} \n"
        + f" -- appending : {append} \n"
    )

    # If append mode is chosen and campaigns are already persistent in the file,
    # load them.
    persist_campaigns = read_from_disk(file_path, strict=False) if append else []
    persist_campaigns.append(campaign)

    # Handle the different serialization modes
    if mode == SerializationMode.JSON:
        with file_path.open("w", encoding="UTF-8") as f:
            json.dump(
                persist_campaigns, f, default=convert, indent=4, ensure_ascii=False
            )

    elif mode == SerializationMode.PICKLE:
        with open(file_path, "wb") as f:
            pickle.dump(persist_campaigns, f)


def read_from_disk(file_path: Path, strict: bool = True) -> list[Campaign]:
    """
    Read campaigns from disk.

    NOTE: Will automatically resolve path ending based on serialization on first-come
    first-serve basis, if it is not appended already.

    Args:
        file_path : Path from which to read
    """
    # Resolve the file ending automatically
    if file_path.suffix not in SerializationMode.list():
        for ending in SerializationMode.list():
            extend_path = file_path.parent / f"{file_path.name}{ending}"
            if extend_path.exists():
                file_path = extend_path

    # Test whether we were able to resolve to a campaign cache file.
    if not SerializationMode.has_value(file_path.suffix):
        logger.warning(
            f"{file_path} contains unknown suffix; possibly reading not handled!"
        )
        if strict:
            raise ValueError(
                f"Given path {file_path} does not point to a valid file type for reading a campaign"
            )
        return []

    if not file_path.is_file():
        logger.trace("File doesn't exist, returning empty list of campaigns.")
        return []

    mode = SerializationMode(file_path.suffix)
    logger.trace(
        "Loading campaigns stored to file ... \n"
        + f" -- file : {file_path} \n"
        + f" -- mode : {mode.name} \n"
    )

    # Handle deserializations
    if mode == SerializationMode.JSON:
        # First load the json
        data = json.loads(file_path.read_text(encoding="UTF-8"))
        assert isinstance(data, list), f"{file_path} didnt contain list of campaigns!"

        # This looks weird, but we take every campaign object (a dictionary), dump it to a json
        # string and then back. The reason being that now we can use the from_json method, which
        # includes the proper deserialization hooks.
        data = [Campaign.from_json(json.dumps(camp)) for camp in data]

    elif mode == SerializationMode.PICKLE:
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        assert isinstance(data, list), f"{file_path} didnt contain list of campaigns!"

    return data
