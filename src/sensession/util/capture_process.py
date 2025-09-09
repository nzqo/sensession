"""
Context Manager class to maintain subprocesses of a single capture phase.

Takes care of starting potentially multiple subprocesses and cleaning them
up upon exiting or destruction of the context manager object.
"""

import time
import queue
import shlex
import atexit
import select
import signal
import subprocess
from typing import Any
from datetime import datetime, timedelta
from dataclasses import dataclass

from loguru import logger

from sensession.config import APP_CONFIG
from sensession.util.shell import shell_run
from sensession.util.exceptions import SubprocessError


@dataclass
class Task:
    """
    Struct to connect a process and a potential cleanup command.
    Named task because upon destroying these are "worked off" in an effort to
    clean up properly.
    """

    process: subprocess.Popen[Any] | None  # Running background process
    cleanup_cmd: str | None  # Task specific cleanup shell command


def await_returncode(process: subprocess.Popen[Any], timeout: timedelta) -> int | None:
    """
    Await returncode
    """
    start = datetime.now()
    retcode = None
    while datetime.now() - start < timeout:
        retcode = process.poll()
        if retcode is not None:
            break
        time.sleep(0.1)
    return retcode


def terminate_process(
    process: subprocess.Popen[Any], ignore_codes: list[int] | None = None
) -> None:
    """
    Terminate a subprocess using a three-step escalation:
        1. Send SIGINT and wait for 500ms.
        2. If still running, call terminate() and wait up to 15 seconds.
        3. If still running, call kill().
    Logs any issues but does not raise exceptions.
    """
    if process.poll() is not None:
        return

    if ignore_codes is None:
        ignore_codes = []

    # Step 1: Send SIGINT and wait
    logger.trace(f"Sending SIGINT to process: {process.args!r}")

    process.send_signal(signal.SIGINT)
    retcode = await_returncode(process, timedelta(milliseconds=500))

    # Step 2: If still running, call terminate() and wait up to 15 seconds.
    if retcode is None:
        logger.trace(
            f"SIGINT insufficient; calling terminate() for process: {process.args!r}"
        )
        process.terminate()
        try:
            process.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            # Step 3: If still running, call kill()
            logger.trace(
                f"terminate() timeout; calling kill() for process: {process.args!r}"
            )
            process.kill()

    # Final check: Log errors if the process still did not terminate or terminated abnormally.
    if process.poll() is None:
        logger.error(f"Process did not terminate: {process.args!r}")
    else:
        retcode = process.returncode
        if retcode not in ignore_codes and retcode != 0:
            logger.error(
                f"Process terminated with non-zero return code {retcode}: {process.args!r}",
            )


def wait_for_keyword(process: subprocess.Popen[Any], keyword: str) -> None:
    """
    Block until the specified keyword appears in the process output, or until the timeout
    defined in APP_CONFIG.shell_timeout_s is reached.

    Args:
        process: The subprocess to monitor.
        keyword: The keyword to wait for.

    Raises:
        TimeoutError: If the keyword is not found within the timeout period.
        SubprocessError: If the process terminates before the keyword is found.
    """
    if process.stdout is None:
        raise ValueError("stdout is not available for waiting.")

    logger.trace(f"Waiting for keyword '{keyword}' in process output...")
    start_time = time.time()
    buffer = ""
    timeout = APP_CONFIG.shell_timeout_s  # Global timeout from config.
    while time.time() - start_time < timeout:
        rlist, _, _ = select.select([process.stdout], [], [], 0.1)
        if rlist:
            line = process.stdout.readline()
            if not line:  # EOF reached.
                break
            decoded_line = line.decode("utf-8", errors="replace")
            buffer += decoded_line
            if keyword in decoded_line:
                logger.trace(f"Found keyword '{keyword}' in output.")
                return
        if process.poll() is not None:
            raise SubprocessError(
                "Process terminated unexpectedly before keyword was found. "
                f"Output so far:\n{buffer}"
            )

    # Ensure the process is terminated, since we exit early with an error now.
    terminate_process(process)
    raise SubprocessError(
        f"Timeout waiting for keyword '{keyword}'. Output so far:\n{buffer}"
    )


class CaptureProcess:
    """
    CaptureProcess is a ContextManager wrapper that ensures that a capture process is
    properly cleaned up (i.e. the subprocess terminated).
    """

    def __init__(self):
        """
        Create a capture process for process lifetime management
        """
        # Context object can manage multiple subprocesses; store them in a LIFO queue
        self.processes: queue.LifoQueue = queue.LifoQueue()

        # NOTE: It is not guaranteed by python that object deleters are called, even when
        # explicitly invoking garbage collection. To make sure that leftover processes here
        # are actually torn down, we use an atexit handler.
        atexit.register(self.teardown)
        self._atexit_registered = True

    def start_process(
        self,
        shell_command: str | None = None,
        cleanup_command: str | None = None,
        wait_for: str | None = None,
    ):
        """
        Start a process to be managed by this object.

        Args:
            shell_command   : Verbatim command (no cmd line expansion supported!).
            cleanup_command : Verbatim command to call for cleanup when closing process.
        """
        process = None

        if wait_for:
            stdout = subprocess.PIPE
        else:
            stdout = (
                subprocess.PIPE
                if not APP_CONFIG.suppress_subprocs
                else subprocess.DEVNULL
            )

        if shell_command:
            fmt_cmd = shell_command.replace(" --", " \\ \n\t--")
            logger.info(f"Starting subprocess: \n{fmt_cmd}")

            # The following is required state, the process will be kept open in the
            # background until the CaptureProcess is deleted. Hence, we can not use
            # a context manager here.
            # Note: We are no longer using os.setsid so that PicoScenes and its children
            # remain in the same process group.
            # pylint: disable=consider-using-with
            process = subprocess.Popen(
                shlex.split(shell_command),
                shell=False,
                stdout=stdout,
                stderr=stdout,
            )
        else:
            assert cleanup_command != "", "Neither shell nor cleanup command provided!!"

        if wait_for and process:
            wait_for_keyword(process, wait_for)

        # Remember started processes via queue
        self.processes.put(
            Task(
                process=process,
                cleanup_cmd=cleanup_command,
            )
        )

    def is_running(self) -> bool:
        """
        Check whether any managed subprocess is still running.
        """
        for task in self.processes.queue:
            if task.process and task.process.poll() is None:
                return True
        return False

    def teardown(self, ignore_codes: list[int] | None = None):
        """
        Stop started capturing processes.

        NOTE: Even on exceptions, teardown will not fail.
        If a returncode given in ignore_codes is given, it won't even be considered an error.
        """
        if self._atexit_registered:
            atexit.unregister(self.teardown)
            self._atexit_registered = False

        # Slight timeout to hope for empty buffers everywhere
        if self.processes.empty():
            return

        if not ignore_codes:
            ignore_codes = []

        logger.trace(
            f"Stopping all registered {self.processes.qsize()} subprocesses ..."
        )

        while not self.processes.empty():
            task: Task = self.processes.get()

            if task.process:
                terminate_process(task.process, ignore_codes)

            if task.cleanup_cmd:
                try:
                    shell_run(task.cleanup_cmd, ignore_codes=ignore_codes)
                except Exception as e:  # pylint: disable=broad-exception-caught
                    logger.error(
                        f"Error running cleanup command: {task.cleanup_cmd}. Error: {e}"
                    )

            self.processes.task_done()

        logger.trace("Capture processes stopped!")

    def __enter__(self):
        """
        Context enter. This is trivial, the interesting part here is automatic cleanup
        on context exit.
        """
        return self

    def __exit__(self, exc_type, exc_value, tb):
        """
        Context exit. Ensure that subprocesses are properly terminated.
        """
        logger.trace("Stopping capture process ...")
        self.teardown()

        if exc_type is not None:
            logger.error("Encountered error in CaptureProcess context!")
            return False

        logger.trace("CSI capture subprocesses should be stopped now.")
        return True

    def __del__(self):
        """
        Destructor
        """
        self.teardown()
