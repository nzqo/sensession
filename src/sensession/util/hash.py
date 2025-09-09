"""
Experiment hashing/ID generation helpers
"""

import time
import hashlib


def get_time_str() -> str:
    """
    Create a current timestamp string
    """
    return f"{time.time_ns()}"


def get_timed_hash(text: str = "") -> str:
    """
    Create a time-salted hash

    Args:
        text : Text to hash
    """
    curr_time = get_time_str()
    hash_id = hashlib.md5()
    hash_id.update((text + curr_time).encode("utf-8"))
    return hash_id.hexdigest()


def get_hash(text: str) -> str:
    """
    Hash a string (deterministically across processes)

    Args:
        text : Text to hash
    """
    hash_id = hashlib.md5()
    hash_id.update((text).encode("utf-8"))
    return hash_id.hexdigest()
