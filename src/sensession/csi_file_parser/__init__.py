"""
CSI file parsers are tool specific methods to take in a file (produced by a tool)
and reformat it into a unified format.
"""


def antenna_idxs_to_bitmask(antenna_idxs: list[int]) -> int:
    """
    Convert antenna indices to a bitmask

    Args:
        antenna_idxs : List of antenna indices
    """
    recv_antenna_bitmask = 0
    for antenna_idx in antenna_idxs:
        recv_antenna_bitmask |= 1 << antenna_idx

    if recv_antenna_bitmask == 0:
        raise RuntimeError(
            "Receive antenna bitmask must not be zero! \n"
            + f" -- Receive antennas specified : {antenna_idxs}"
        )
    return recv_antenna_bitmask
