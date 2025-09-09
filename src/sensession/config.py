"""
Collection of configurations used throughout the library.

Most importantly, contains all the experiment and sensing session participant
configuration structs.
"""

import re
import json
import hashlib
import logging
from enum import Enum, IntEnum
from typing import Type
from pathlib import Path
from ipaddress import IPv4Address
from dataclasses import field, asdict, dataclass

from loguru import logger
from rich.logging import RichHandler

try:
    import tomllib  # type: ignore
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore


# A dictionary of WiFi channels and their corresponding frequencies.
# NOTE: This may not be complete and it does not matter, since we are not looking to have
# a full WiFi-featured project, but just support enough to be able to test a wider range
# of channels with little configuration overhead and error possibility.
WIFI_CHANNELS = {
    1: 2_412_000_000,
    2: 2_417_000_000,
    3: 2_422_000_000,
    4: 2_427_000_000,
    5: 2_432_000_000,
    6: 2_437_000_000,
    7: 2_442_000_000,
    8: 2_447_000_000,
    9: 2_452_000_000,
    10: 2_457_000_000,
    11: 2_462_000_000,
    12: 2_467_000_000,
    13: 2_472_000_000,
    14: 2_484_000_000,
    32: 5_160_000_000,
    36: 5_180_000_000,
    40: 5_200_000_000,
    44: 5_220_000_000,
    48: 5_240_000_000,
    52: 5_260_000_000,
    56: 5_280_000_000,
    60: 5_300_000_000,
    64: 5_320_000_000,
    68: 5_340_000_000,
    96: 5_480_000_000,
    100: 5_500_000_000,
    104: 5_520_000_000,
    108: 5_540_000_000,
    112: 5_560_000_000,
    116: 5_580_000_000,
    120: 5_600_000_000,
    124: 5_620_000_000,
    128: 5_640_000_000,
    132: 5_660_000_000,
    136: 5_680_000_000,
    140: 5_700_000_000,
    144: 5_720_000_000,
    149: 5_745_000_000,
    153: 5_765_000_000,
    157: 5_785_000_000,
    161: 5_805_000_000,
    165: 5_825_000_000,
    169: 5_845_000_000,
    173: 5_865_000_000,
    177: 5_885_000_000,
}


#######################################################################################
## Modes and discrete choices
#######################################################################################
# fmt: off
class Bandwidth(IntEnum):
    """
    Generally allowed WiFi bandwidth values
    """
    TWENTY               = 20
    FOURTY               = 40
    EIGHTY               = 80
    HUNDRED_SIXTY        = 160
    THREE_HUNDRED_TWENTY = 320

    def in_mhz(self) -> int:
        """
        Get bandwidth value in Megahertz
        """
        return self.value

    def in_hz(self) -> int:
        """
        Get bandwidth value in Hertz
        """
        return int(self.value * 1e6)


class DataRateMode(str, Enum):
    """
    802.11 data rate mode
    """
    NON_HIGH_THROUGHPUT          = "Non-HT"
    HIGH_THROUGHPUT              = "HT"     # 802.11n
    VERY_HIGH_THROUGHPUT         = "VHT"    # 802.11ac
    HIGH_EFFICIENCY_SINGLE_USER  = "HE-SU"   # 802.11ax
    EXTREMELY_HIGH_THROUGHPUT    = "EHT-SU"  # 802.11be


@dataclass(frozen=True, eq=True)
class MacAddr:
    """
    Struct for strongly typed Mac Address.
    """
    addr: str

    def __post_init__(self):
        if not self._is_valid():
            raise ValueError(f"Address {self.addr} is not valid MAC address.")

    def get(self) -> str:
        """
        Get the string mac address value
        """
        return self.addr.lower()

    def strip(self) -> str:
        """
        Get stripped version of address
        """
        return self.addr.replace(":", "").replace("-", "")

    def with_separator(self, separator :str = ":"):
        """
        Return MAC with a custom separator between the 6 groups.
        """
        stripped = self.strip()
        return separator.join(stripped[i:i+2] for i in range(0, len(stripped), 2))

    def _is_valid(self) -> bool:
        """
        Check whether the stored string address is actually in valid MAC address format
        
        NOTE: Valid Mac Address is six groups of two byte each, separated by either
        a dash, a colon, or nothing.
        """
        return re.match("[0-9a-f]{2}([-:]?)[0-9a-f]{2}(\\1[0-9a-f]{2}){4}$", self.get()) is not None


#######################################################################################
## CSI Capture Tool specific options
#######################################################################################
@dataclass()
class MocapConfig:
    """
    Vicon Motion Capture Management Tool Config 
    """
    ip       : IPv4Address
    port     : int
    host_dir : str

    def __post_init__(self):
        if not isinstance(self.ip, IPv4Address):
            self.ip = IPv4Address(self.ip)


#######################################################################################
## Channel configuration, i.e. parameters dictating where the frame is sent
#######################################################################################
@dataclass
class Channel:
    """
    Description of a WiFi channel
    """

    number          : int
    bandwidth       : Bandwidth       = Bandwidth(20)
    center_freq_hz  : int             = 0                   # Only required in > 20 MHz cases
    control_freq_hz : int             = field(init=False)

    def __post_init__(self):
        if isinstance(self.bandwidth, int):
            self.bandwidth = Bandwidth(self.bandwidth)

        if self.number not in WIFI_CHANNELS:
            raise ValueError(f"Channel number {self.number} unknown or unsupported")

        control_freq_hz = WIFI_CHANNELS[self.number]

        if self.bandwidth == Bandwidth.TWENTY:
            if self.center_freq_hz > 0:
                logger.warning("Center frequency is automatically deduced from channel number in 20 MHz settings.")
            self.center_freq_hz = control_freq_hz
        elif self.center_freq_hz == 0:
            # Bonding takes an even number of non-overlapping 20 MHz channels. Thus,
            # the center frequency always falls between two such non-overlapping channels.
            # By going down 10 MHz, we choose the closest lower sidechannel as control.
            self.center_freq_hz = int(control_freq_hz + 10e6)
            logger.warning(f"No center frequency given for a bonded channel! Will use a default: {self.center_freq_hz}")

        # Some notes here: Larger channels > 20 MHz are made up of multiple non-overlapping 20 MHz
        # ones. In that case, we require for configuration the center frequency as well as a control
        # channel. The latter must be one of the smaller 20 MHz ones.
        self.control_freq_hz = control_freq_hz

    @staticmethod
    def from_dict(data : dict):
        """
        Create a Channel object from dictionary values
        """
        return Channel(
            number = data["number"],
            bandwidth= data["bandwidth"],
            center_freq_hz=data["center_freq_hz"]
        )

    def to_string(self):
        """
        Convert channel to string
        """
        return (
            f"{self.number} (bw : {self.bandwidth}, center freq: {self.center_freq_hz / 1e6} MHz, control freq: {self.control_freq_hz / 1e6} MHz)"
        )

#######################################################################################
## Configuration of the frame to be generated for and sent with the SDR. This frame is
## detected at the receiver(s) and used to extract CSI. As such, PHY and MAC headers
## contain some relevant information.
#######################################################################################
FrameId = str

@dataclass
class BaseTransmissionConfig:
    """
    Basic transmission settings.
    
    NOTE: Inherit from this class to extend for specified tools.
    """
    n_reps   : int = 1            # Number of repetitions (how often to repeat the frame transmission process)
    pause_ms : int = 0            # Pause time (in milliseconds) between repetitions
    gain     : int | float = 25   # Transmission gain
    start_at : int | None = None  # When to start transmission (seconds since epoch, None = now)

    def __post_init__(self):
        self.n_reps = int(self.n_reps)
        self.pause_ms = int(self.pause_ms)
        self.gain = int(self.gain)


@dataclass(eq=True)
class BaseFrameConfig:
    """
    Basic frame configuration, i.e. parameters used to specify a single simple
    WiFi 802.11 frame
    
    NOTE: We allow empty fields because this config can also be used to specify
    frames to filter for. When an address is None, it means to not apply a filter
    for that address.
    """
    receiver_address    : MacAddr | None = None      # Mac address of receiver of the frame
    transmitter_address : MacAddr | None = None      # Mac address of the transsmitter (e.g. AP)
    bssid_address       : MacAddr | None = None      # BSSID address, usually equal transmitter_address
    bandwidth           : Bandwidth = Bandwidth(20)  # Bandwidth of underlying channel
    data_rate_mode      : DataRateMode = DataRateMode.HIGH_THROUGHPUT # Data rate mode (i.e. protocol) of frame
    _frame_id           : str | None = None          # Cached frame id for lazy load

    def __post_init__(self):
        # string conversion
        if isinstance(self.receiver_address, str):
            self.receiver_address = MacAddr(self.receiver_address)
        if isinstance(self.transmitter_address, str):
            self.transmitter_address = MacAddr(self.transmitter_address)
        if isinstance(self.bssid_address, str):
            self.bssid_address = MacAddr(self.bssid_address)

        if isinstance(self.receiver_address, dict):
            self.receiver_address = MacAddr(**self.receiver_address)
        if isinstance(self.transmitter_address, dict):
            self.transmitter_address = MacAddr(**self.transmitter_address)
        if isinstance(self.bssid_address, dict):
            self.bssid_address = MacAddr(**self.bssid_address)
        if isinstance(self.bandwidth, int):
            self.bandwidth = Bandwidth(self.bandwidth)
        if not isinstance(self.data_rate_mode, DataRateMode):
            self.data_rate_mode = DataRateMode(self.data_rate_mode)

    def frame_id(self) -> FrameId:
        """
        Get a hash for the IQ frame config
        """
        if self._frame_id:
            return self._frame_id

        hash_id = hashlib.md5()
        hash_id.update(repr(self).encode("utf-8"))
        self._frame_id = hash_id.hexdigest()
        return self._frame_id

    def ensure_specified(self):
        """
        Ensure frame is fully specified.
        """
        if not self.receiver_address:
            raise TypeError("Group Config must be fully specified but has no receiver addr")
        if not self.transmitter_address:
            raise TypeError("Group Config must be fully specified but has no transmitter addr")
        if not self.bssid_address:
            raise TypeError("Group Config must be fully specified but has no bssid addr")


#######################################################################################
## Global configuration, containing:
##   Hardware Setup; Assembling all hardware participant configuration
##   Global switches
#######################################################################################
@dataclass
class EnvironmentConfig:
    """
    Environment config, i.e. the bundled information on all receivers and transmitters
    available in the actual physical system.
    """
    loglevel            : str  = "TRACE"
    logfile             : str  = ""
    log_rotation        : str  = "500 MB"
    log_retention       : int  = 3
    keep_tmp_files      : bool = False                   # Keep temporary files (e.g. nexmon pcap captures)
    overwrite_tmp_files : bool = True                    # Whether to overwrite existing tmp files
    cache_dir           : Path = Path.cwd() / ".cache"   # Directory of cache
    cache_trim          : int  = int(1e10)               # Byte size after which to trim frame cache to avoid blow up
    matlab_batch_size   : int  = 10                      # Number of frames to generate in parallel during generation time
    max_matlab_worker   : int  = 4                       # Max number of parallel matlab instances for parallelization pool
    matlab_lazy_init    : bool = True                    # Whether to create matlab subprocesses in pool on demand or ahead
    suppress_subprocs   : bool = True                    # Whether to suppress subprocess terminal output
    shell_timeout_s     : int  = 600                     # Maximum time to wait for a shell subprocess
    wait_after_logs     : int | None = None              # For some logs, for readability, it makes sense to wait after. Only in interactive sessions.
    lazy_database       : bool = False                   # Whether to use/write DataFrame in lazy mode. Makes sense for very large experiments.
# fmt: on


#######################################################################################
## Convenience helpers
#######################################################################################
def get_pretty_config(config) -> str:
    """
    Convert dataclass config to a pretty printed json object

    Args:
        config : Dataclass object to convert
    """
    return json.dumps(asdict(config), indent=4, default=vars)


def _load_config(config_file_path: Path, config_type: Type, default_ok: bool = False):
    """
    Parse config from toml file into dataclass type

    Args:
        config_file_path : Path to configuration file
        config_type      : Type to parse dict into
    """

    if not config_file_path.is_file():
        if default_ok:
            logger.warning(
                f"Config file {config_file_path} not provided; Using default"
            )
            return config_type()

        raise FileNotFoundError(
            f"Config file {config_file_path} not found; Check path again!"
        )

    if not config_file_path.suffix == ".toml":
        raise FileNotFoundError(
            f"Config file {config_file_path} is not toml; Wrong format."
        )

    with open(config_file_path, "rb") as f:
        data = tomllib.load(f)
        config = config_type(**data)

    return config


# Load a global config.
CONFIG_FILE = Path.cwd() / "sense_config.toml"
APP_CONFIG: EnvironmentConfig = _load_config(
    CONFIG_FILE,
    config_type=EnvironmentConfig,
    default_ok=True,
)


def configure_logger():
    """
    Configure logger from global config
    """
    logging.addLevelName(5, "TRACE")
    logger.remove()

    sink = RichHandler(markup=True, show_time=False)
    fmt = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{message}</level>"
    logger.add(sink=sink, level=APP_CONFIG.loglevel.upper(), format=fmt)

    if APP_CONFIG.logfile != "":
        logger.add(
            APP_CONFIG.logfile,
            rotation=APP_CONFIG.log_rotation,
            retention=APP_CONFIG.log_retention,
            level=APP_CONFIG.loglevel,
            format="<level>{level: <8}</level> | " + fmt,
        )

    logger.trace("Logger (re-)configured from provided application config.")


configure_logger()
