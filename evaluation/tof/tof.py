"""
Time of Flight computations
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.signal
from scipy.optimize import differential_evolution
from evaluation.tof.pdp import compute_pdp, compute_delays

SPEED_OF_LIGHT = 299_792_458  # m/s


###############################################################################
# TOF computation config
###############################################################################
# fmt: off
@dataclass
class TOFConfig:
    """
    Hyperparameters for ToF calculation.
    """

    min_peak_distance : int = 4        # in sample indices
    min_peak_strength : float = 0.01   # relative threshold
    pad_factor        : int = 1        # zero-padding factor


@dataclass
class ToFStats:
    receiver_name   : str              # name of processed receiver
    seq_numbers     : np.ndarray       # Sequence numbers
    computed_ns     : np.ndarray       # Computed time of flight values in nanosecnds
    ground_truth_ns : np.ndarray       # Ground truth ToF values in nanoseconds
    avg_error_ns    : float            # Average error in nanosecnds
    std_error_ns    : float            # standard deviation of the error in nanoseconds
# fmt: on


###############################################################################
# TOF Processing Classes
###############################################################################
class TOFProcessor:
    """
    Performs ToF estimation and ground truth computation using precomputed PDP and delays.
    """

    def __init__(self, config: TOFConfig):
        self.config = config

    ###############################################################################
    # Estimation Functions
    ###############################################################################
    def estimate_tof_difference(
        self, pdp: np.ndarray, delays: np.ndarray
    ) -> np.ndarray:
        """
        Estimate the Time of Flight (ToF) difference based on a precomputed PDP and delay axis.

        Args:
            pdp: PDP matrix of shape (padded_len, n_samples).
            delays: 1D array of delay values (in seconds) corresponding to the PDP rows.

        Returns:
            tof_diff: Array of ToF differences (in seconds) for each time sample.
        """
        n_samples = pdp.shape[1]
        tof_diff = np.zeros(n_samples)

        for t in range(n_samples):
            peaks, properties = scipy.signal.find_peaks(
                pdp[:, t], height=np.max(pdp[:, t]) * self.config.min_peak_strength
            )
            if len(peaks) < 2:
                continue

            sorted_peaks = peaks[np.argsort(properties["peak_heights"])[::-1]]
            primary_peak = sorted_peaks[0]
            secondary_peak = None
            for peak in sorted_peaks[1:]:
                if (
                    np.abs(peak - primary_peak) >= self.config.min_peak_distance
                    and delays[peak] >= 0
                ):
                    secondary_peak = peak
                    break

            if secondary_peak is not None:
                tof_diff[t] = np.abs(delays[secondary_peak] - delays[primary_peak])

        return tof_diff

    def get_tof_stats(
        self,
        computed_tof_ns: np.ndarray,
        sequence_numbers: np.ndarray,
        receiver_name: str,
        verbose: bool = True,
        carrier_spacing: float = 312.5e3,
    ) -> ToFStats:
        """
        Compute the ground truth ToF and error metrics given the already computed ToF difference.

        Args:
            computed_tof_ns: Array of computed ToF differences (in nanoseconds).
            sequence_numbers: 1D array of sequence numbers.
            receiver_name: Identifier for logging.
            verbose: If True, log the deviation and volatility.

        Returns:
            A dictionary with computed ToF, ground truth ToF (both in ns), sequence numbers,
            average deviation (in ns), and volatility (std dev in ns).
        """
        ground_truth = compute_ground_truth_tof_ns(sequence_numbers)
        ground_truth = quantize_tof_ns(ground_truth, carrier_spacing)
        avg_err, std_err = compute_error_metrics(computed_tof_ns, ground_truth)

        # optional logging
        if verbose:
            print(
                f"Receiver {receiver_name} with config: {self.config}: \n"
                f"Avg ToF deviation = {avg_err:.2f} ns\n"
                f"Volatility (std) = {std_err:.2f} ns\n"
            )
        return ToFStats(
            receiver_name=receiver_name,
            seq_numbers=sequence_numbers,
            computed_ns=computed_tof_ns,
            ground_truth_ns=ground_truth,
            avg_error_ns=avg_err,
            std_error_ns=std_err,
        )

    @staticmethod
    def optimize_hyperparameters(
        csi: np.ndarray,
        sequence_numbers: np.ndarray,
        pad_factor: int = 1,
        carrier_spacing: float = 312.5e3,
    ) -> tuple[TOFConfig, float]:
        """
        Optimize TOF hyperparameters using differential evolution.

        Args:
            csi: CSI matrix of shape (n_subcarriers, n_samples).
            sequence_numbers: 1D array of sequence numbers.

        Returns:
            A tuple of the optimal TOFConfig and the best average deviation in ns.
        """
        pdp = compute_pdp(csi, pad_factor)
        n_subcarriers = csi.shape[0]
        delays = compute_delays(n_subcarriers, pad_factor, carrier_spacing)

        def objective(params):
            candidate_config = TOFConfig(
                min_peak_distance=max(1, min(int(round(params[0])), 9)),
                min_peak_strength=params[1],
                pad_factor=pad_factor,
            )
            candidate_proc = TOFProcessor(candidate_config)
            computed_tof_sec = candidate_proc.estimate_tof_difference(pdp, delays)
            computed_tof_ns = computed_tof_sec * 1e9
            ground_truth_distance = 100 + (sequence_numbers / 9999.0) * 200.0
            ground_truth_tof = ground_truth_distance / SPEED_OF_LIGHT
            ground_truth_tof_ns = ground_truth_tof * 1e9
            ts = 1e9 / (57 * carrier_spacing)
            ground_truth_tof_ns = np.round(ground_truth_tof_ns / ts) * ts
            deviation = np.mean(np.abs(computed_tof_ns - ground_truth_tof_ns))
            return deviation

        bounds = [(1, 9), (0.001, 0.03)]
        result = differential_evolution(
            objective, bounds, strategy="best1bin", disp=False, polish=True
        )
        best_config = TOFConfig(
            min_peak_distance=int(round(result.x[0])),
            min_peak_strength=result.x[1],
            pad_factor=pad_factor,
        )
        return best_config, result.fun


def compute_ground_truth_tof_ns(seq: np.ndarray) -> np.ndarray:
    """From sequence numbers → ground-truth ToF in ns (unquantized)."""
    dist = 100 + (seq / 9999.0) * 200.0  # meters
    tof_s = dist / SPEED_OF_LIGHT  # seconds
    return tof_s * 1e9  # in nanoseconds


def quantize_tof_ns(gt_ns: np.ndarray, carrier_spacing: float) -> np.ndarray:
    """Round ground-truth ToF to the system's time resolution."""
    ts = 1e9 / (57 * carrier_spacing)
    return np.round(gt_ns / ts) * ts


def compute_error_metrics(
    computed_ns: np.ndarray, ground_truth_ns: np.ndarray
) -> tuple[float, float]:
    """Return (mean absolute error, std dev)."""
    err = computed_ns - ground_truth_ns
    return float(np.mean(np.abs(err))), float(np.std(err))
