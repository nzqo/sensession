"""
Matlab Parallelization helper

Matlab instances are very expensive to startup (in terms of time). To make the most
out of it, this file contains a parallelizer implementation which allows working on
multiple matlab tasks in parallel.

The parallelizer maintains a pool of active matlab engines and workers on top that
use these engines to perform registered tasks. More specifically, workers fetch an
engine, and pass that engine together with a task to the registered callback. Task
here simply means function arguments for the callback. The callback specifies what
to do with the engine.

For example:
    > callback = lambda engine, x : engine.disp(x)
    > parallelizer = MatlabParallelizer(callback)

Now you could register multiple tasks (in this example things to display) and then
work them off in parallel at once.
"""

import atexit
from queue import Queue
from typing import Any, Iterable
from pathlib import Path
from threading import Thread
from collections import deque
from dataclasses import dataclass

try:
    import matlab.engine as matlabengine  # pylint: disable=import-error # type: ignore

    MATLAB_AVAILABLE = True
except ImportError:
    MATLAB_AVAILABLE = False

from loguru import logger  # pylint: disable=wrong-import-order

from sensession.config import APP_CONFIG
from sensession.util.exceptions import MatlabError


@dataclass
class Task:
    """
    Struct identifying a single task to work off in matlab
    """

    task_id: str  # ID to identify this task upon completion (which may be out of order)
    kwargs: dict  # Keyword arguments to feed to the registered callback


@dataclass
class CompletedTask(Task):
    """
    Class wrapper containing information on a completed task
    """

    retval: Any  # Return value of the callback associated to the task


# --------------------------------------------------------------------------
# Engine wrapper - RAII class to maintain a single matlab engine.
# --------------------------------------------------------------------------
class EngineWrapper:
    """
    Engine wrapper to ensure startup and shutdown of associated matlab processes.
    """

    def __init__(self, processing_callback, start_path: Path = Path.cwd() / "matlab"):
        """
        Constructor

        Args:
            processing_callback : Callback with signature (matlab_engine, args), where
            args can be amassed using the Parallelizer below.

        Example:
            wrapper = EngineWrapper(generate_frame)
        """
        if not MATLAB_AVAILABLE:
            raise ImportError(
                "Cant use matlab engine without matlab engine module installed"
            )
        logger.trace("Starting Matlab Engine ...")
        self.eng = matlabengine.start_matlab()
        self.eng.addpath(str(start_path))
        self.processing_callback = processing_callback
        atexit.register(self.eng.quit)

    def __del__(self):
        logger.trace(
            "Destroying Matlab engine wrapper object -- Closing engine connection ..."
        )
        atexit.unregister(self.eng.quit)
        self.eng.quit()

    def process(self, kwargs):
        """
        Process kwargs by invoking the callbak with the wrapped engine

        Args:
            kwargs : arguments to pass to callback aside from engine
        """
        try:
            return self.processing_callback(self.eng, **kwargs)
        except matlabengine.RejectedExecutionError as e:
            raise MatlabError(f"Encountered matlab error: {e}") from e


class MatlabParallelizer:
    """
    Helper to parallelize multiple matlab instances.

    We keep a pool of open Matlab Engines to work with.
    A user can then register new tasks and have them processed using the `process`
    function. We instrument threads that use the engines in the pool to work of all
    registered tasks until finished.
    """

    def __init__(self, matlab_callback):
        """
        Args:
            matlab_callback : A function callback with signature (matlab_engine, ...kwargs).
        """
        self.task_queue: Queue[Task | object] = Queue()
        self.matlab_callback = matlab_callback

        self.engines: Queue[EngineWrapper] = Queue()

        if not APP_CONFIG.matlab_lazy_init:
            logger.info(
                f"Starting {APP_CONFIG.max_matlab_worker} matlab engine instances ..."
            )
            for _ in range(APP_CONFIG.max_matlab_worker):
                self.engines.put(EngineWrapper(self.matlab_callback))

    def get_num_tasks(self) -> int:
        """
        Get number of (unfinished) tasks
        """
        return self.task_queue.qsize()

    def get_num_engines(self) -> int:
        """
        Get number of workers in pool
        """
        return self.engines.qsize()

    def add_task(self, task: Task):
        """
        Args:
            task : Task to work on
        """
        # Enqueue!
        logger.trace(f"Enqueuing task (id: {task.task_id}) ...")
        self.task_queue.put(task)

    def lazy_init_engines(self):
        """
        Lazily initialize matlab processes. We check how many tasks are registered
        and add new workers up to a configured ceiling of `max_worker`.
        """
        if not APP_CONFIG.matlab_lazy_init:
            return

        num_engines = self.get_num_engines()
        wanted_new = self.get_num_tasks() - num_engines
        pool_full = num_engines >= APP_CONFIG.max_matlab_worker
        if wanted_new > 0 and not pool_full:
            num_new_worker = min(wanted_new, APP_CONFIG.max_matlab_worker - num_engines)

            logger.trace(f"Initializing {num_new_worker} new worker for our pool ...")
            for _ in range(num_new_worker):
                self.engines.put(EngineWrapper(self.matlab_callback))

    def process(self) -> Iterable[CompletedTask]:
        """
        Multiprocessing to parallelize active matlab engines.

        Returns:
            A deque filled with the completed tasks
        """
        if self.get_num_tasks() == 0:
            logger.debug("No tasks registered, nothing to do. Returning ...")
            return []

        # Ensure enough engine instances are open
        self.lazy_init_engines()
        num_engines = self.get_num_engines()
        if num_engines == 0:
            raise RuntimeError("No matlab engines in pool, something went wrong.")

        # Sentinel object to signalize workers to shut down
        sentinel = object()
        finished_tasks: "deque[CompletedTask]" = deque()

        # Define worker thread(s)
        # Workers will work through all enlisted tasks and append them to queue as finished
        def worker():
            nonlocal finished_tasks

            while True:
                task = self.task_queue.get()
                if task is sentinel:
                    logger.trace("Sentinel received -- terminating worker")
                    break
                assert isinstance(task, Task), "Tasks must be of type Task."

                # Take an available wrapped engine and execute task
                logger.trace(
                    "Starting to work on task -- Fetching an active engine connection"
                )
                engine = self.engines.get()
                result = engine.process(task.kwargs)

                # Return wrapped engine for another worker to use
                logger.trace("Worker finished task -- Returning engine to pool")
                self.engines.put(engine)

                finished_tasks.append(CompletedTask(**task.__dict__, retval=result))

        # Start worker threads
        logger.trace("Starting worker threads for parallelized matlab processing ...")
        threads = []
        num_worker = num_engines
        for _ in range(num_worker):
            thread = Thread(target=worker)
            threads.append(thread)
            thread.start()
        logger.trace("Matlab worker threads started.")

        # Insert sentinels to terminate threads and wait for them to finish
        for _ in range(num_worker):
            self.task_queue.put(sentinel)

        logger.trace("Waiting for threads to finish ...")
        for thread in threads:
            thread.join()

        logger.trace("All threads finished; Tasks should be completed now!")

        return finished_tasks
