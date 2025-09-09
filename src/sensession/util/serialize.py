"""
We want to be able to serialize configs, so we can check which experiments
we ran and also possibly reuse that to rerun failed experiments.

Proper serialization is annoying in python.
Could have used pickles, but those are not readable on other machines.
The problem is that nested dataclasses and some types do not behave nicely
in e.g. json serialization.

For every type we want to deserialize, we provide a dict unwrapping in
their __post_init__() method. Currently, this leaves us with handling
numpy arrays, which we do here.
"""

from __future__ import annotations

from enum import Enum
from typing import Any
from pathlib import Path
from datetime import timedelta
from ipaddress import IPv4Address
from dataclasses import asdict, dataclass

import numpy as np
import polars as pl

from sensession.devices import DeviceConfigType, BaseTransmissionConfig
from sensession.util.enum import BaseEnum
from sensession.devices.esp32 import (
    CSISelection,
    ESPOperationMode,
    ESP40MHzSecondaryChannel,
)
from sensession.tools.picoscenes import PicoscenesTransmissionConfig
from sensession.util.frame_generation import (
    IQFrameConfig,
    IQFrameGroupConfig,
    InterleavedIQFrameGroupConfig,
)


class FrameConfigType(BaseEnum):
    """
    Enumeration of all available types of WiFi frame configurations
    """

    USRP_IQ = IQFrameConfig
    USRP_IQ_GROUP = IQFrameGroupConfig
    USRP_IQ_INTERLEAVED = InterleavedIQFrameGroupConfig


class TransmissionConfigType(BaseEnum):
    """
    Enumeration of all available transmission config types
    """

    USRP = BaseTransmissionConfig
    PICOSCENES = PicoscenesTransmissionConfig


enum_registry = {
    "ESPOperationMode": ESPOperationMode,
    "CSISelection": CSISelection,
    "ESP40MHzSecondaryChannel": ESP40MHzSecondaryChannel,
}


#######################################################################################
## Config wrappers to aid in serialization and simpler "config-based" execution
#######################################################################################
@dataclass
class DeviceConfig:
    """
    Wrapper around generic config types.

    Aids serialization and automatic typing of config based on the `DeviceConfigType`
    Enum.
    """

    config: Any
    config_type: str = ""

    def __post_init__(self):
        if isinstance(self.config, dict):
            assert self.config_type, (
                "Config type should be automatically deduced; "
                + "Maybe serialization went wrong and didn't write it?"
            )

            self.config = DeviceConfigType[self.config_type].value(**self.config)
        else:
            cfg_type = type(self.config)
            if not DeviceConfigType.has_value(cfg_type):
                raise RuntimeError(f"Device Config type {cfg_type} unknown.")

            self.config_type = DeviceConfigType(cfg_type).name


@dataclass
class FrameConfig:
    """
    Wrap device configs to force custom serialization
    """

    config: Any
    config_type: str = ""

    def __post_init__(self):
        if isinstance(self.config, dict):
            assert self.config_type, (
                "Config type should be automatically deduced; "
                + "Maybe serialization went wrong and didn't write it?"
            )

            self.config = FrameConfigType[self.config_type].value(**self.config)
        else:
            cfg_type = type(self.config)
            if not FrameConfigType.has_value(cfg_type):
                raise RuntimeError(f"Frame Config type {cfg_type} unknown.")

            self.config_type = FrameConfigType(cfg_type).name


@dataclass
class TransmissionConfig:
    """
    Wrap transmission configs to force custom serialization
    """

    config: Any
    config_type: str = ""

    def __post_init__(self):
        if isinstance(self.config, dict):
            assert self.config_type, (
                "Config type should be automatically deduced; "
                + "Maybe serialization went wrong and didn't write it?"
            )

            self.config = TransmissionConfigType[self.config_type].value(**self.config)
        else:
            cfg_type = type(self.config)
            if not TransmissionConfigType.has_value(cfg_type):
                raise RuntimeError(f"Frame Config type {cfg_type} unknown.")

            self.config_type = TransmissionConfigType(cfg_type).name


@dataclass
class Complex:
    """
    Complex numbers are also not json serializable, for that we create a simple
    custom type
    """

    real: float
    imag: float

    @staticmethod
    def from_complex(num: np.complex64) -> Complex:
        """
        Create from complex number
        """
        return Complex(float(num.real), float(num.imag))

    def to_complex(self) -> np.complex64:
        """
        Convert back to complex number
        """
        return np.complex64(self.real + 1.0j * self.imag)


def convert(x: Any):  # pylint: disable=too-many-return-statements
    """
    json conversion hook to handle odd types
    """
    if isinstance(x, np.ndarray):  # numpy arrays have this
        return {"__ndarray": list(x)}
    if isinstance(x, np.complex64):
        return {"__complex": asdict(Complex.from_complex(x))}
    if isinstance(x, timedelta):
        return {"__milliseconds": x / timedelta(milliseconds=1)}
    if isinstance(x, IPv4Address):
        return str(x)
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, pl.datatypes.classes.DataTypeClass):
        return {"__polars_datatype": str(x)}
    if isinstance(x, Enum):
        return {"__enum": x.__class__.__name__, "value": x.value}
    try:
        a = asdict(x)
        return a
    except TypeError:
        pass

    raise TypeError(f"Dictionary conversion for {type(x)} not handled")


def deconvert(x: dict):
    """
    json deconversion hook to handle deserialization
    """
    if "__complex" in x:
        return Complex(**x["__complex"]).to_complex()

    if "__ndarray" in x:
        return np.array(x["__ndarray"])

    if "__milliseconds" in x:
        return timedelta(milliseconds=x["__milliseconds"])

    if "__polars_datatype" in x:
        return getattr(pl, x["__polars_datatype"])

    if "__enum" in x:
        enum_class = enum_registry[x["__enum"]]
        return enum_class(x["value"])

    return x
