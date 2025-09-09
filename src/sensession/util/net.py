"""
Basic network utils
"""

from sensession.config import Channel, Bandwidth


def get_sideband_value(channel: Channel) -> str:
    """
    Nexmon uses "u" for upper and "l" for lower sideband. This function extracts
    this value.

    NOTE: This is only relevant for 40 MHz Bandwidths.
    """
    # On the router, sideband values are only relevant for 2.4 GHz and 40 MHz channels
    if channel.bandwidth == Bandwidth.FOURTY and channel.number <= 14:
        assert channel.center_freq_hz, (
            "Channel must have a center frequency set in > 20 MHz scenarios"
        )

        sideband = "u" if channel.control_freq_hz > channel.center_freq_hz else "l"
        return sideband

    return ""
