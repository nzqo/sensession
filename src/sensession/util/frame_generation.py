"""
Generation of WiFi Frames, specifically down to the waveform (IQ-samples)
"""

from __future__ import annotations

import shelve
import hashlib
from enum import Enum
from typing import Union
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass

import numpy as np
from loguru import logger
from scipy.io import loadmat
from scipy.signal import resample

from sensession.config import (
    APP_CONFIG,
    FrameId,
    DataRateMode,
    BaseFrameConfig,
    get_pretty_config,
)
from sensession.util.exceptions import ApiUsageError
from sensession.util.matlab_parallelizer import Task, MatlabParallelizer


#######################################################################################
## Configuration of the frame to be generated for and sent with the SDR. This frame is
## detected at the receiver(s) and used to extract CSI. As such, PHY and MAC headers
## contain some relevant information.
#######################################################################################
# fmt: off
class GuardIntervalMode(str, Enum):
    """
    Mode of 802.11 guard intervals
    """
    SHORT = "Short"   # 0.4 ns guard interval length between symbols
    LONG  = "Long"    # 0.8 ns guard interval length between symbols

FrameDelay = Union[int, timedelta] # Either number of samples or a nanosecond-level delay between frames


@dataclass()
class Mask:
    """
    OFDM Precoding Mask
    """

    precoding_symbols : np.ndarray       # A [num_mask, num_subcarrier] array of masks
    name : str                      = "" # Descriptive name of mask


    def hash(self) -> str:
        """
        Get a hash for a mask
        """
        hash_id = hashlib.sha256(self.precoding_symbols.data)
        return hash_id.hexdigest()

    def is_trivial(self) -> bool:
        """
        Check whether mask is trivial, i.e. does not change any symbols
        """
        return bool(np.all(self.precoding_symbols == 1))

    def __post_init__(self):
        if not isinstance(self.precoding_symbols, np.ndarray):
            self.precoding_symbols = np.ndarray(self.precoding_symbols, dtype=np.complex64)
        self.precoding_symbols = self.precoding_symbols.astype(np.complex64)


@dataclass()
class IQFrameConfig(BaseFrameConfig):
    """
    Extended frame configuration for USRP, where frames are created via matlab with high
    configurability options
    """
    send_rate_hz        : int  = int(20e6)  # The rate with which this frame is to be sent upsampled to avoid congruency issues.
    enable_sounding     : bool = False      # Whether to force sounding bit in PHY preamble
    rescale_factor      : int  = 25000      # Max value to scale int16 sample outputs in file to (applied on base frame before precoding)
    guard_iv_mode       : GuardIntervalMode = GuardIntervalMode.SHORT         # Specifies length of guard interval in PPDU
    data_rate_mode      : DataRateMode      = DataRateMode.HIGH_THROUGHPUT    # HT mode

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.guard_iv_mode, str):
            self.guard_iv_mode = GuardIntervalMode(self.guard_iv_mode)
        if isinstance(self.data_rate_mode, str):
            self.data_rate_mode = DataRateMode(self.data_rate_mode)

        if self.send_rate_hz < self.bandwidth.in_hz():
            raise ValueError("Please choose a send_rate_hz larger than the bandwidth. "
                             + "If in doubt, just set them equal.")

        # Make sure base frame is fully specified (no empty fields)
        self.ensure_specified()


@dataclass()
class IQFrameGroupConfig:
    """
    Configuration of a group of (possibly masked) frames, one after the other.
    All frames in a group are based off of the same basic frame, and differ only
    through precoding masking.
    """
    base_frame        : IQFrameConfig
    group_repetitions : int           = 1     # Number of times to repeat masked-frame-group for final sample file
    interframe_delay  : FrameDelay    = 0     # Number of interframe zero-padding samples
    mask              : Mask | None   = None  # OFDM symbol mask for precoding

    def __post_init__(self):
        if self.group_repetitions < 1:
            raise ValueError("FrameGroupConfig group_repetitions must be positive. ")

        if isinstance(self.mask, dict):
            self.mask = Mask(**self.mask) # pylint: disable=not-a-mapping
        if isinstance(self.mask, np.ndarray):
            self.mask = Mask(precoding_symbols=self.mask)

        if self.interframe_delay != 0 and self.interframe_delay != timedelta(0):
            if (self.mask is None or self.mask.is_trivial()) and self.group_repetitions == 1:
                raise ValueError("Interframe delay only makes sense for non-trivial masks or repeated frames!")

        if not isinstance(self.mask, Mask | None):
            raise ValueError("Could not initialize mask; Supply `Mask` or `np.ndarray`")

        # This prevents a nasty bug: Imagine using an IQFrameConfig and calling its frame_id. This will
        # cache the answer in _frame_id. If we now create an IQFrameGroupConfig from that, we would use
        # the same id. Instead, we should reset it and re-evaluate lazily.
        self._frame_id = None


    def frame_id(self) -> FrameId:
        """
        Create a hash for a framegroup for identification.
        """
        if self._frame_id:
            return self._frame_id

        base_id = self.base_frame.frame_id()

        if (self.interframe_delay == 0 or self.interframe_delay == timedelta(0)) and self.group_repetitions == 1 and (not self.mask or self.mask.is_trivial()):
            return base_id

        # NOTE: The following should guarantee that a trivial IQFrameGroupConfig object, i.e.
        # one with no repetitions, no mask, no delay, has the same hash as its underlying IQFrameConfig
        hash_id = hashlib.md5()
        hash_bytes = base_id.encode("utf-8")

        if self.mask:
            hash_bytes += hashlib.sha256(self.mask.precoding_symbols.data).hexdigest().encode("utf-8")

        hash_bytes += str(self.group_repetitions).encode("utf-8")
        hash_bytes += str(self.interframe_delay).encode("utf-8")

        hash_id.update(hash_bytes)
        self._frame_id = hash_id.hexdigest()
        return self._frame_id

    @property
    def bandwidth(self):
        """
        Getter to expose bandwidth from base frame
        """
        return self.base_frame.bandwidth

    @property
    def send_rate_hz(self):
        """
        Getter to expose send rate from base frame
        """
        return self.base_frame.send_rate_hz

@dataclass()
class InterleavedIQFrameGroupConfig:
    """
    Configuration of a group of (possibly masked) frames, one after the other.
    All frames in a group are based off of the same basic frame, and differ only
    through precoding masking.
    """
    base_frames       : list[IQFrameConfig]
    group_repetitions : int           = 1     # Number of times to repeat masked-frame-group for final sample file
    interframe_delay  : FrameDelay    = 0     # Number of interframe zero-padding samples
    mask              : Mask | None   = None  # OFDM symbol mask for precoding
    _groups           : list[IQFrameGroupConfig] | None = None  # Automatically generated
    _frame_id         : str = ""

    def __post_init__(self):
        """
        Automatically generate groups This will leverage IQFrameGroupConfig's validation as well as
        aid usage down the line.
        """
        # NOTE: Should check whether all the frame configs are actually compatible,
        # i.e. bandwidths etc match..

        if not self.base_frames or len(self.base_frames) < 2:
            raise ValueError("Must provide multiple base frames to interleave!")

        self._groups = [IQFrameGroupConfig(base_frame = bf, group_repetitions = self.group_repetitions, interframe_delay = self.interframe_delay, mask = self.mask) for bf in self.base_frames]

    def frame_id(self) -> FrameId:
        """
        Create a hash for a framegroup for identification.
        """
        if self._frame_id:
            return self._frame_id

        hash_bytes = "".encode("utf-8")

        assert self._groups, "Groups must be generated automatically!"
        for group in self._groups:
            hash_bytes += group.frame_id().encode("utf-8")

        hash_id = hashlib.md5()
        hash_id.update(hash_bytes)
        self._frame_id = hash_id.hexdigest()
        return self._frame_id

    @property
    def bandwidth(self):
        """
        Getter to expose bandwidth from base frame
        """
        return self.base_frames[0].bandwidth

    @property
    def send_rate_hz(self):
        """
        Getter to expose send rate from base frame
        """
        return self.base_frames[0].send_rate_hz


USRPFrameConfig = Union[IQFrameConfig, IQFrameGroupConfig, InterleavedIQFrameGroupConfig]

@dataclass(frozen=True)
class GeneratedFrameInfo:
    """
    Struct to maintain information on a generated frame, i.e. one for which a
    file with IQ-samples exist
    """
    frame_config : USRPFrameConfig         # Config from which frame was generated
    frame_file   : Path                    # Full path to generated file
    created_at   : datetime                # Time of creation
    file_size    : int                     # Size of the associated IQ-sample file

TRIVIAL_name: str = "_unmodified"
# fmt: on


# -------------------------------------------------------------------------------------
# Frame Cache
# -------------------------------------------------------------------------------------
def importance_sorted(
    frame_dict: dict[str, GeneratedFrameInfo], schedule: list[str] | None = None
):
    """
    Sort to dictate which frames are removed first
    We sort by a few criteria:
        - Whether the frame is reused according to the schedule
        - (minus) File size, small frames should be kept longer
        - Creation time, to remove old frames first

    Args:
        schedule : A schedule indicating which frame digests will be used
            again in which order in the future.
    """
    if schedule is None:
        schedule = []

    return dict(
        sorted(
            frame_dict.items(),
            key=lambda item: (
                item[0] in schedule,
                -item[1].file_size,
                item[1].created_at,
            ),
        )
    )


def select_trim_frames(
    frame_dict: dict[str, GeneratedFrameInfo], remove_bytes: int
) -> dict[str, GeneratedFrameInfo]:
    """
    Select frames to remove such that at least `remove_bytes` are freed.

    Args:
        frame_dict   : Frame dictionary to select frames to remove from
        remove_bytes : Minimum amount of bytes that need to be removed.
    """
    if remove_bytes <= 0:
        return {}

    bytesizes = np.array([frame.file_size for frame in frame_dict.values()])
    bytesizes_cum = np.cumsum(bytesizes)

    # NOTE: Need to choose index + 1 to also include the last element in removal.
    index = np.searchsorted(bytesizes_cum, remove_bytes) + 1
    return dict(list(frame_dict.items())[0:index])


class FrameCache:
    """
    Frame Cache to manage storing frames in a cache directory for reuse
    """

    def __init__(self, cache_dir: Path):
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = cache_dir
        self._cache_file = cache_dir / "frame_map.shelve"
        self._cached_frames = self._read_from_shelve()

    def get_cached_bytesize(self) -> int:
        """
        Get size of cached frame files in bytes
        """
        return sum(frame.file_size for frame in self._cached_frames.values())

    def trim_cache(self, max_bytesize: int, schedule: list[str] | None = None):
        """
        Delete the oldest frames to clear up to cache

        Args:
            max_bytesize : Maximum number of bytesize before cache should be trimmed
            schedule     : A schedule of upcoming frames. If present, can help optimize
                           by opting not to delete frames referenced again in the future.
                           NOTE: This is not a guarantee that they aren't deleted!
        """
        if schedule is None:
            schedule = []

        current_size = self.get_cached_bytesize()

        # If nothing needs to be removed, stop here.
        if current_size < max_bytesize:
            return

        # Calculate how much to remove
        tbremoved = current_size - max_bytesize
        logger.debug(f"Trimming cache of {tbremoved} bytes ...")

        importance_sorted_frames = importance_sorted(self._cached_frames, schedule)
        trim_frames = select_trim_frames(importance_sorted_frames, max_bytesize)

        for frame_digest in trim_frames:
            self.remove_cached_file(frame_digest)
            logger.debug(f"Cleared cache of frame {frame_digest}")

        # Frame dictionary modified -> write
        self._write_to_shelve()

    def get_cached_frame(self, frame_digest: FrameId) -> GeneratedFrameInfo | None:
        """
        Retrieve cached frame value, if any.

        Args:
            frame_digest : Unique ID of frame

        Returns:
            Generated Frame Info if frame is present in cache, otherwise None.
        """
        if frame_digest in self._cached_frames:
            return self._cached_frames[frame_digest]
        return None

    def get_all_cached(self) -> dict[FrameId, GeneratedFrameInfo]:
        """
        Retrieve all currently cached frames
        """
        return self._cached_frames.copy()

    def is_cached(self, frame_digest: FrameId) -> bool:
        """
        Check whether frame is cached or not.

        Args:
            frame : config of frame group
        """
        return frame_digest in self._cached_frames

    def register_cached_file(self, frame_info: GeneratedFrameInfo):
        """
        Register frame as cached

        Args:
            frame_info : Frame info struct of generated frame
        """
        digest = frame_info.frame_config.frame_id()
        self._cached_frames[digest] = frame_info
        self._write_to_shelve()

    def remove_cached_file(self, frame_digest: FrameId):
        """
        Remove cached file

        Args:
            frame_digest : Unique ID of frame to remove
        """
        frame_info = self._cached_frames[frame_digest]
        frame_info.frame_file.unlink()
        del self._cached_frames[frame_digest]
        self._write_to_shelve()

    def _read_from_shelve(self) -> dict[FrameId, GeneratedFrameInfo]:
        """
        Read dictionary values from a shelve.
        """
        with shelve.open(str(self._cache_file)) as db:
            frames: dict[FrameId, GeneratedFrameInfo] = dict(db)
            return {
                frame_id: gen_info
                for frame_id, gen_info in frames.items()
                if gen_info.frame_file.is_file()
            }

    def _write_to_shelve(self):
        """
        Write info of generated frames to a shelve.

        NOTE: Will overwrite previous entries, if existent.
        """
        if not all(
            frame_info.frame_file.is_file()
            for frame_info in self._cached_frames.values()
        ):
            raise ApiUsageError("configs added to shelf must contain valid filepaths!")

        with shelve.open(str(self._cache_file)) as db:
            db.clear()
            for frame_hash, frame_info in self._cached_frames.items():
                db[frame_hash] = frame_info


def resolve_mask(mask: Mask | None, bandwidth_mhz: int) -> np.ndarray:
    """
    Resolve mask, i.e. return either the masking data or a trivial mask.
    """
    n_scs = int(bandwidth_mhz * 3.2)

    if mask:
        mask_scs = mask.precoding_symbols.shape[0]
        assert mask_scs == n_scs, (
            f"Mask does not match number of subcarriers ({mask_scs} vs {n_scs})"
        )
        return mask.precoding_symbols

    return np.ones((n_scs, 1), dtype=np.complex64)


# -------------------------------------------------------------------------------------
# Frame generation helper functions
# -------------------------------------------------------------------------------------
def generate_waveform(
    eng, config: USRPFrameConfig, frame_file: Path
) -> GeneratedFrameInfo:
    """
    Generate IQ waveform based on one of multiple configs.
    """
    # Easiest to just treat a single frame as a group with one element.
    if isinstance(config, IQFrameConfig):
        config = IQFrameGroupConfig(base_frame=config)

    # temp file to put generated frames in to interface between matlab and python.
    # Also aliases for better readability.
    tmp_file = frame_file.with_suffix(".mat")
    mask = config.mask
    group_reps = config.group_repetitions

    # Treating group vs interleaved frames
    if isinstance(config, IQFrameGroupConfig):
        waveforms = generate_frame_group(
            eng, config.base_frame, tmp_file, mask, group_reps
        )
    elif isinstance(config, InterleavedIQFrameGroupConfig):
        groups = [
            generate_frame_group(eng, cfg, tmp_file, mask, group_reps)
            for cfg in config.base_frames
        ]
        waveforms = np.vstack(list(zip(*groups)))
    else:
        raise ValueError(
            f"Unhandled config type in waveform generation: {type(config)}"
        )

    # -------------------------------------------------------------------------
    # Extract interframe padding from delay
    delay = config.interframe_delay
    bandwidth_mhz = config.bandwidth.in_mhz()

    if isinstance(delay, timedelta):
        # Extract time delta in microsecond
        time_micros = delay / timedelta(microseconds=1)

        # This will give the number of complex samples that make up the desired
        # spacing
        # microsecond: 10e-6 seconds
        # megahertz:   10e6  hertz
        delay = int(time_micros * bandwidth_mhz)
    elif not isinstance(delay, int):
        raise ValueError(
            "interframe delay must be either int (num of samples) or timedelta (time)!"
        )

    # Convert transmission rate to mhz
    rate_mhz = config.send_rate_hz
    if rate_mhz > 1e6:
        rate_mhz //= int(1e6)

    # Concatenate generated frames to a single waveform and save to file
    waveform = concat_frames(waveforms, delay, bandwidth_mhz, rate_mhz)
    waveform.astype(np.int16).tofile(frame_file)

    # Ensure frame file actually exists and contains samples as sanity check
    file_size = frame_file.stat().st_size
    if not frame_file.is_file() or file_size == 0:
        raise RuntimeError(
            f"Frame Generation failed?! - {frame_file} is not populated!"
        )

    return GeneratedFrameInfo(
        frame_config=config,
        frame_file=frame_file,
        created_at=datetime.now(),
        file_size=file_size,
    )


def generate_frame_group(
    eng,
    config: IQFrameConfig,
    tmp_file: Path,
    mask: Mask | None = None,
    group_repetitions: int = 1,
) -> np.ndarray:
    """
    Generate an IQ-sample file for a group of WiFi frames

    Args:
        eng     : matlab engine object
        config  : Configuration of frame group to generate
    """
    # Start with sanity checks
    tmp_file = Path(tmp_file)
    if not tmp_file:
        raise ValueError("Must specify frame file path")

    assert config.receiver_address, "Frame generation requires receiver address"
    assert config.transmitter_address, "Frame generation requires transmitter address"
    assert config.bssid_address, "Frame generation requires bssid address"

    # And some announcements
    msg = (
        "Generating frame with configuration: \n"
        + f" -- base group id     : {config.frame_id()}\n"
        + f" -- num group reps    : {group_repetitions}\n"
        + f" -- base frame config : {get_pretty_config(config)}\n"
    )
    if mask:
        msg = (
            msg
            + f" -- mask name         : {mask.name}\n"
            + f" -- mask shape        : {mask.precoding_symbols.shape}\n"
        )

    logger.trace(msg)

    data_rate_mode = config.data_rate_mode.value
    assert config.data_rate_mode in [
        DataRateMode.NON_HIGH_THROUGHPUT,
        DataRateMode.VERY_HIGH_THROUGHPUT,
        DataRateMode.HIGH_THROUGHPUT,
    ], "Only Non-HT, HT and VHT supported for frame generation currently.."

    bandwidth_mhz = config.bandwidth.in_mhz()
    mask_arr = resolve_mask(mask, bandwidth_mhz)

    # -------------------------------------------------------------------------
    # Generate masked frames
    ret = eng.generate_csimasked_frame(
        str(tmp_file),
        config.rescale_factor,
        config.receiver_address.strip(),
        config.transmitter_address.strip(),
        config.bssid_address.strip(),
        bandwidth_mhz,
        group_repetitions,
        config.enable_sounding,
        mask_arr,
        config.guard_iv_mode.value,
        data_rate_mode,
        nargout=0,
        background=True,
    )
    ret.result()

    waveforms = loadmat(tmp_file)["waveforms"]
    tmp_file.unlink()
    return waveforms


def concat_frames(
    waveforms: np.ndarray, frame_sample_len: int, bandwidth_mhz: int, rate_mhz: int
) -> np.ndarray:
    """
    Concatenate matlab-generated frames into a single frame file
    """
    # First things first: If we want to send at a different rate, resample the
    # waveforms properly. Note that this also upscales the full frame length
    if bandwidth_mhz != rate_mhz:
        logger.trace(
            f"Resampling {bandwidth_mhz} to {rate_mhz} using Fourier method..."
        )
        n_samples = (waveforms.shape[1] * rate_mhz) // bandwidth_mhz
        waveforms = resample(waveforms, n_samples, axis=1)
        frame_sample_len = (frame_sample_len * rate_mhz) // bandwidth_mhz

    num_frames, num_samples = waveforms.shape

    # The least distance between two frames must be the frame length.
    # Also, we store IQ samples, i.e. interleaved real and imaginary part,
    # hence we need to scale by a factor of two.
    frame_sample_len = 2 * max(frame_sample_len, num_samples)
    wf_size = num_samples * 2

    logger.trace(
        "Concatenating frames to single waveform: \n"
        + f" -- Num frames......................: {num_frames}\n"
        + f" -- Num real samples per frame......: {wf_size}\n"
        + f" -- Num real samples (after padding): {frame_sample_len}"
    )

    # Making space for the concatenated waveform (IQ samples, hence times two)
    waveform = np.zeros(num_frames * frame_sample_len, dtype=np.float64)

    for i in range(num_frames):
        # Pad with zeros at the front, i.e. put waveform at the last samples
        start_idx = (i + 1) * frame_sample_len - wf_size
        waveform[start_idx : start_idx + wf_size : 2] = np.real(waveforms[i])
        waveform[start_idx + 1 : start_idx + wf_size + 1 : 2] = np.imag(waveforms[i])

    assert np.max(waveform) < 32767, "Bad waveform scaling (exceeds int16 range)"
    assert np.max(waveform) >= 100, "Bad waveform scaling (too small for int16)"

    return waveform


def unique_list(lst: list, seen: set | None = None):
    """
    Get only unique values from a list while preserving the order.

    Args:
        lst  : The list to get unique values from
        seen : A set of preliminary values to remove
    """
    if seen is not None:
        lst = [x for x in lst if x not in seen]

    return list(dict.fromkeys(lst))


# -------------------------------------------------------------------------------------
# Frame Generator class to create IQ frames
# -------------------------------------------------------------------------------------
class FrameGenerator:
    """
    Manager class to simplify frame generation.

    The generator will take care of caching and IQ file generation, with options such
    as pre-generation.
    """

    def __init__(
        self,
        frame_cache_dir: Path = APP_CONFIG.cache_dir / "frames",
        init_from_disk: bool = True,
    ):
        frame_cache_dir.mkdir(parents=True, exist_ok=True)
        self.frame_cache = FrameCache(frame_cache_dir)
        self.frame_generator = MatlabParallelizer(generate_waveform)

        # Dictionaries of frames. Unifies the used type and allows for some
        # simpler code
        # NOTE: We deal with either single WiFi Frames or so-called frame groups,
        # where multiple frames plus breaks in-between are joined in a single IQ
        # sample file. The former are realized through a trivial group setting.
        self.frames: dict[str, USRPFrameConfig] = {}

        # In case we initialize from disk, we can sync with the frame cache.
        if init_from_disk:
            self.frames = {
                digest: frame.frame_config
                for digest, frame in self.frame_cache.get_all_cached().items()
            }

        # A schedule of frames in the order in which they are to be used. This
        # allows parallel pre-generation of frames for a speed up.
        self._frame_schedule: list[str] = []

    def add_frame(self, config: USRPFrameConfig) -> FrameId:
        """
        Register a frame for management.

        Args:
            config : Either a raw frame or a frame group config

        Returns:
            The hash with which the frame can be addressed in the manager in other calls
        """

        frame_id = config.frame_id()
        self.frames[frame_id] = config
        return frame_id

    def enable_pregen(self, frame_schedule: list[FrameId]):
        """
        Enable pregeneration of frames.

        This will take frames, in the order that they were registered, and
        generate a batch of frame files ahead of time.

        This can make use of better parallelization.

        WARNING: This only makes sense if you use the frames in the order in
        which you have registered them!
        """
        self._frame_schedule = frame_schedule

    def _advance_schedule(self, digest):
        """
        Advance in schedule to keep track of the current place and assert that it matches
        the current digest
        """
        if not self._frame_schedule:
            return

        # Update schedule access and check that it matches what was specified
        # NOTE: This is required so that later generator calls know where in
        # the schedule they need to start
        logger.trace(
            f"Advancing with digest {digest}. Schedule: {self._frame_schedule[:4]}, ..."
        )

        if digest not in self._frame_schedule:
            logger.error(
                f"Advancing to digest {digest} which is not in schedule. Switching to non-pregen mode."
            )
            self._frame_schedule = []
            return

        frame_idx = self._frame_schedule.index(digest)
        self._frame_schedule = self._frame_schedule[frame_idx:]

    def retrieve_frame(self, digest: FrameId) -> GeneratedFrameInfo:
        """
        Retrieve a frame, i.e. the IQ frame file corresponding to the frame specified
        by the `digest`.

        Args:
            digest: The hash as it was returned by `add_frame` upon registration
        """
        logger.trace(f"Retrieving frame with digest {digest} from Generator ...")
        if digest not in self.frames:
            raise ApiUsageError(
                f"Hash {digest} does not belong to a known frame; Did you call `add_frame`?"
            )

        if frame := self.frame_cache.get_cached_frame(digest):
            self._advance_schedule(digest)
            return frame

        # -----------------------------------------------------------------------------
        # The desired frame is not cached at this point.
        # Before we generate, we make sure the cache is not trashed too much.
        self.frame_cache.trim_cache(APP_CONFIG.cache_trim)

        # Now we either generate the single desired frame or a whole batch
        if self._frame_schedule:
            self._lookahead_generate(digest)
            self._advance_schedule(digest)
        else:
            self._generate(digest)

        # Retrieve the desired frame
        frame = self.frame_cache.get_cached_frame(digest)
        assert frame, f"FATAL: No frame with digest {digest} cached, but should be!!!"
        return frame

    def _lookahead_generate(self, start_digest: FrameId):
        """
        Generate a batch of IQ frames as specified in the schedule.

        NOTE: Must have `enable_pregen` called with a schedule beforehand.

        Args:
            start_digest : The frame ID from which to start batch generation.
        """
        if not self._frame_schedule:
            return

        if start_digest not in self._frame_schedule:
            logger.warning(
                f"{start_digest} not found in schedule; Did you not keep up the schedule access order?"
            )
            self._generate(start_digest)
            return

        generated_frames = set(self.frame_cache.get_all_cached().keys())

        # Check which ones are not known in frame cache
        ungenerated = unique_list(self._frame_schedule, seen=generated_frames)

        # Add the whole batch to the generator
        for digest in ungenerated[: APP_CONFIG.matlab_batch_size]:
            self._add_to_generator(digest)

        new_frames = self.frame_generator.process()
        for task in new_frames:
            self.frame_cache.register_cached_file(frame_info=task.retval)

    def _generate(self, digest: FrameId):
        """
        Generate a single IQ frame
        """
        self._add_to_generator(digest)
        new_frames = list(self.frame_generator.process())
        assert len(new_frames) == 1, (
            f"Generated just one frame, but generated {len(new_frames)}!"
        )
        self.frame_cache.register_cached_file(frame_info=new_frames[0].retval)

    def _add_to_generator(self, digest: FrameId):
        """
        Add a frame group config to the frame generator to generate the described
        frame later with it.

        Args:
            digest : Unique ID of frame to add
        """
        config = self.frames[digest]
        frame_file = self.frame_cache.cache_dir / f"frame_digest_{digest}.dat"
        self.frame_generator.add_task(
            Task(
                task_id=digest,
                kwargs={"config": config, "frame_file": frame_file},
            )
        )
