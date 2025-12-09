"""
Rectangles experiments

Test how cards detect rectangles, i.e. choosing a set of neighboring
subcarriers and rescaling their CSI amplitudes by a range of factors.
"""

# pylint: disable=duplicate-code
from datetime import timedelta

import numpy as np
import polars as pl
from common import ExperimentFixture


def to_mask(curves: pl.DataFrame, run_idx: int) -> np.ndarray:
    """
    Create a magnitude-mask from a curve.

    Every point in the curve will be interpreted as global scaling factor
    for one frame. In other words, the curve is a curve in time, with each
    value being a scaling factor applied as precoding to a frame.
    """
    # Introduce phase shift on each subcarrier
    envelope = curves[run_idx, "curve"].to_numpy()

    # NOTE: The curve emulates changes in distance d(t) that
    # introduce a Doppler shift phase factor:
    #
    # D(t) = exp(-j 2 pi f d(t) / c)
    #
    # This Doppler shift, in a simple two-path model, will change
    # the channel according to:
    #
    # H(f, t) = H_s(f) + a(f) D(t)
    #
    # where a(f) is some initial dynamic path value and H_s(f)
    # the static channel. We assume a(f) = H_s(f) * 0.1 to be able
    # to compute at least something here by precoding. Since this
    # allows to e.g. use a precoding factor p(t) of:
    #
    # p(t) = 0.9 + 0.1 * D(t)
    #
    # NOTE: We could also use p(t) = D(t), emulating complete shadowing
    # of the direct path and saying a(f) = H_s(f) ...
    #
    # The curves are relatively smooth, going from one extreme to
    # another in maybe 200 samples. Assuming 200 CSI per second,
    # an average walking speed of 1m/s, it is reasonable to interpret
    # d(t) units being in meters.
    #
    # We use Channel 1:
    # f_c   = 2412 Megahertz, BW 20 MHz
    # f_0   = 2402 Megahertz
    # f_63  = 2422 Megahertz
    # del_f = 312.5 kHz
    # => f(i) = 2402 MHz + i * 312.5 kHz
    # => f(i) / c = 8.012 * 1/m + i * 0.0010424 * 1/m

    # NOTE: For now we ignore the 0.001 factor to introduce a conceptually
    # larger shift that should be better detectable?
    n_subcarrier = 64
    freq_comb = 8.012 + np.arange(0, n_subcarrier, 1) * 1 / 64  # * 0.0010424
    doppler_phase = freq_comb[:, np.newaxis] * envelope
    mask = 0.7 + 0.3 * np.exp(-2j * np.pi * doppler_phase)
    return mask


def linear_mask(
    carrier_freq: int, num: int = 30000, channel_spacing: float = 312.5e3
) -> np.ndarray:
    """
    Linear movement emulation
    """
    n_subcarrier = 64

    # this is f_k / c as an array across subcarriers
    c_speed = 299_792_458  # speed of light in m/s
    indices = np.arange(-n_subcarrier // 2, n_subcarrier // 2)
    freq_comb = (carrier_freq / c_speed) + indices * (channel_spacing / c_speed)

    print(f"f/c check: {carrier_freq / c_speed}")

    # linear distance: moving from 100m distance over the course of `num` packets
    # At x fps, this means v = 100m / (x * num seconds)
    # In other words, this is pretty fast.
    distance = np.linspace(150, 250, num=num)

    # exp(-j 2 pi d(t) f / c)
    doppler_shift = np.exp(-2j * np.pi * freq_comb[:, np.newaxis] * distance)

    # This should emulate a two-path model: times the static (i.e. non-emulated)
    # component, then add our doppler shifted one.
    return 0.7 + 0.3 * doppler_shift


def get_fixture() -> ExperimentFixture:
    """
    Configure experiment fixture
    """
    name = "doppler_emulation"

    # NOTE: Play around with the gain!
    fixture = ExperimentFixture(name, if_delay=timedelta(milliseconds=1))

    carrier_freq = fixture.channel.center_freq_hz

    # ---------- Linear Doppler Emulation
    mask = linear_mask(carrier_freq=carrier_freq)
    fixture.add_schedule_for_mask(
        mask=mask,
        schedule_name=name,
        group_reps=1,
    )

    return fixture


def run():
    """run experiment"""
    fixture = get_fixture()
    fixture.run()


if __name__ == "__main__":
    run()
