"""
Rectangles experiments

Test how cards detect rectangles, i.e. choosing a set of neighboring
subcarriers and rescaling their CSI amplitudes by a range of factors.
"""
# pylint: disable=duplicate-code

from pathlib import Path

import numpy as np
import polars as pl
from common import ExperimentFixture
from loguru import logger


def kernel_function(
    x1: np.ndarray, x2: np.ndarray, length_scale: float = 1.0, amplitude: float = 1.0
):
    """
    Kernel (covariance) function - a squared exponential
    """
    return amplitude**2 * np.exp(-0.5 * (np.subtract.outer(x1, x2) / length_scale) ** 2)


def get_random_func(num_samples: int, lower: float = 0.5, upper: float = 1.0):
    """
    Create a smooth random function between [lower, upper] and with num_samples
    points.
    """
    # Input points (time)
    time_vals = np.linspace(0, num_samples * 0.02, num_samples)

    # Generate the covariance matrix using the kernel function
    kernel = kernel_function(time_vals, time_vals)

    # Sample one random Gaussian process
    mean = np.zeros_like(time_vals)
    sample = np.random.multivariate_normal(mean, kernel)

    # Rescale into [lower, upper]
    sample -= np.min(sample)
    sample = (sample / np.max(sample) * (upper - lower)).astype(np.float64)
    sample += lower
    return sample


def linear_correction(arr: np.ndarray) -> np.ndarray:
    """
    Correction used to fix outermost subcarriers to zero

    Args:
        arr : A (num_subcarrier, num_time_samples) array
    """
    idxs = np.arange(arr.shape[0])
    return (
        arr
        - arr[0, :]
        - idxs[:, np.newaxis] * (arr[-1, :] - arr[0, :]) / (arr.shape[0] - 1)
    )


def generate_smooth_random_2d_func(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    num_time_points: int = 700,
    num_subcarrier: int = 64,
    length_scale_time: float = 400.0,
    length_scale_subcarrier: float = 40.0,
    lower: float = 0.3,
    upper: float = 1.0,
    fix_boundary: bool = False,
):
    """
    Generate a smooth random 2D function on a grid with size
    (num_subcarrier x num_time_points). Instead of directly sampling, we sample
    in frequency domain since the exponential kernel factorizes nicely there.
    """
    # Create a 2D grid of frequencies
    freq_x = np.fft.fftfreq(num_time_points)
    freq_y = np.fft.fftfreq(num_subcarrier)
    freq_x, freq_y = np.meshgrid(freq_x, freq_y)

    # Apply different length scales to each dimension
    freq_squared = (freq_x**2 * (length_scale_time**2)) + (
        freq_y**2 * (length_scale_subcarrier**2)
    )

    # Adjust the power spectrum to achieve the desired smoothness
    power_spectrum = np.exp(-0.5 * freq_squared)

    # Sample random Fourier coefficients
    real_part = np.random.normal(0, 1, (num_subcarrier, num_time_points))
    imag_part = np.random.normal(0, 1, (num_subcarrier, num_time_points))
    fourier_coefficients = real_part + 1j * imag_part

    if fix_boundary:
        # Fixes initial value f(0,k) = 0, making linear correction less
        # problematic for smoothness
        fourier_coefficients[0, :] -= np.sum(fourier_coefficients, axis=0)

    # Apply the power spectrum to the Fourier coefficients
    fourier_coefficients *= np.sqrt(power_spectrum)

    # Perform the inverse FFT to get the 2D function in real space
    sample = np.fft.ifft2(fourier_coefficients).real

    if fix_boundary:
        # NOTE: assuming lower = upper
        sample = linear_correction(sample)
        sample = (sample / np.max(np.abs(sample)) * upper).astype(np.float64)
    else:
        # Rescale into [lower, upper]
        sample -= np.min(sample)
        sample = (sample / np.max(sample) * (upper - lower)).astype(np.float64)
        sample += lower

    return sample


def generate_2d_curves(num_samples: int = 700, num_curves: int = 100) -> pl.DataFrame:
    """
    Generate a set of curves
    """
    data = {
        "curve_abs": [
            generate_smooth_random_2d_func(num_samples) for _ in range(num_curves)
        ],
        "curve_phs": [
            generate_smooth_random_2d_func(
                num_samples, lower=-0.25, upper=0.25, fix_boundary=True
            )
            for _ in range(num_curves)
        ],
    }

    return pl.DataFrame(
        data,
        schema={
            "curve_abs": pl.List(inner=pl.List(inner=pl.Float64)),
            "curve_phs": pl.List(inner=pl.List(inner=pl.Float64)),
        },
    ).with_row_index("num_curve")


def generate_curves(num_samples: int = 700, num_curves: int = 100) -> pl.DataFrame:
    """
    Generate a set of curves
    """
    data = {"curve": [get_random_func(num_samples) for _ in range(num_curves)]}

    return pl.DataFrame(
        data,
        schema={
            "curve": pl.List(inner=pl.Float64),
        },
    ).with_row_index("num_curve")


def save_curves(curves: pl.DataFrame, outfile: Path):
    """
    Store curves
    """
    curves.write_parquet(outfile)


def get_curves(
    outfile: Path, num_samples: int = 700, num_curves: int = 100
) -> pl.DataFrame:
    """
    Get curves, using either stored curves or generating a fresh set
    """
    if outfile.exists() and outfile.is_file():
        logger.info(f"Curves were already generated. Reusing from: {outfile}")
        return pl.read_parquet(outfile)

    # Ensure path exists
    outfile.parent.mkdir(exist_ok=True, parents=True)

    logger.info("No curves found. Generating a random set of new ones.")
    df = generate_2d_curves(num_samples=num_samples, num_curves=num_curves)
    save_curves(df, outfile)
    return df


def to_mask_single(curves: pl.DataFrame, run_idx: int) -> np.ndarray:
    """
    Convert curve to a mask
    """
    # Apply same amplitude across every subcarrier
    envelope = curves[run_idx, "curve"].to_numpy()
    return np.repeat(envelope[np.newaxis, :], repeats=64, axis=0)


def to_mask_multi(curves: pl.DataFrame, run_idx: int) -> np.ndarray:
    """
    Convert multi-subcarrier curve to mask
    """
    envelope = curves[run_idx, "curve_abs"].to_list()
    phs = curves[run_idx, "curve_phs"].to_list()
    mask = np.array(envelope) * np.exp(2j * np.pi * np.array(phs))
    return mask


def get_fixture() -> ExperimentFixture:
    """
    Configure experiment fixture
    """
    expname = "random_curves2"
    outfile = Path.cwd() / "data" / expname / "curves.parquet"

    num_samples = 1500
    num_curves = 200
    experiment_nums = range(1, 4)
    curves = get_curves(outfile, num_samples, num_curves)

    fixture = ExperimentFixture(expname)

    num_curves = curves.select(pl.len()).item()
    for num_exp in experiment_nums:
        for curve_idx in range(num_curves):
            mask = to_mask_multi(curves, curve_idx)
            session_nr = (num_exp * num_curves) + curve_idx
            fixture.add_schedule_for_mask(
                mask=mask,
                training_reps=0,
                schedule_name=f"run_{session_nr}",
                group_reps=1,  # mask already contains repetitions implicitly
                curve_nr=(curve_idx, pl.UInt16),
                session_nr=(session_nr, pl.UInt32),
                rep_nr=(num_exp, pl.UInt16),
            )

    return fixture


def run():
    """run experiment"""
    fixture = get_fixture()
    fixture.run()


if __name__ == "__main__":
    run()
