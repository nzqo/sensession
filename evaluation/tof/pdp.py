"""
Power delay profile computations
"""

import numpy as np


###############################################################################
# Processing Helpers
###############################################################################
def compute_pdp(channel: np.ndarray, pad_factor: int = 2):
    """
    Compute the Channel Impulse Response (CIR) and normalized Power Delay Profile (PDP)
    with symmetric zero-padding.

    Args:
        channel: CSI matrix of shape (n_subcarriers, n_samples).
        pad_factor: Zero-padding factor.

    Returns:
        cir: Complex CIR matrix.
        pdp: Normalized PDP matrix.
    """
    n_subcarriers, n_samples = channel.shape
    padded_len = n_subcarriers * pad_factor
    pad_total = padded_len - n_subcarriers
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left

    # Zero-pad symmetrically
    channel_padded = np.concatenate(
        (
            np.zeros((pad_left, n_samples), dtype=channel.dtype),
            channel,
            np.zeros((pad_right, n_samples), dtype=channel.dtype),
        ),
        axis=0,
    )

    # FFT shifting and IFFT to get CIR, then compute PDP
    channel_shifted = np.fft.ifftshift(channel_padded, axes=0)
    cir = np.fft.ifft(channel_shifted, n=padded_len, axis=0)
    cir = np.fft.fftshift(cir, axes=0)
    pdp = np.abs(cir) ** 2
    pdp /= np.max(pdp)

    return pdp


def compute_delays(
    n_subcarriers: int, pad_factor: int, carrier_spacing: float = 312.5e3
) -> np.ndarray:
    """
    Compute a delay axis in seconds.

    Args:
        n_subcarriers: Number of subcarriers in the original CSI matrix.
        pad_factor: Zero-padding factor.
        carrier_spacing: Frequency spacing between subcarriers.

    Returns:
        delays: 1D numpy array of delays in seconds.
    """
    padded_len = n_subcarriers * pad_factor
    bandwidth = padded_len * carrier_spacing
    delays = np.arange(-padded_len // 2 + 1, padded_len // 2 + 1) / bandwidth
    return delays
