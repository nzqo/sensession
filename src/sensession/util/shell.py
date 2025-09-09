"""
Little shell abstraction to centralize shell command calls
"""

import subprocess

from loguru import logger

from sensession.config import APP_CONFIG
from sensession.util.exceptions import SubprocessError


def shell_run(
    command: str,
    capture=APP_CONFIG.suppress_subprocs,
    try_count: int = 1,
    ignore_codes: list[int] | None = None,
) -> subprocess.CompletedProcess:
    """
    Run shell command

    Args:
        command : command to run
        capture : Whether to capture or suppress the shell run output
        try_count : Possible number of tries, in case timeout occurs

    Returns:
        The completed shell process info struct
    """
    assert try_count > 0, "Must try at least once (base invocation!)"
    logger.debug(f"Executing shell command as subprocess: \n> {command}")

    if not ignore_codes:
        ignore_codes = []

    count = try_count
    while True:
        try:
            result = subprocess.run(
                command,
                shell=True,
                text=True,
                check=False,  # We check below, allowing to ignore specific codes.
                capture_output=capture,
                timeout=APP_CONFIG.shell_timeout_s,
            )
            break
        except subprocess.TimeoutExpired as e:
            count -= 1
            if count == 0:
                raise SubprocessError(
                    f"Subprocess failed after {try_count} tries"
                ) from e

    if result.returncode not in ignore_codes:
        result.check_returncode()
    logger.trace(f"Executed `{command}` successfully")

    return result
