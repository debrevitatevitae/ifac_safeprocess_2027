"""Run independent jobs, such as one training run per seed, in parallel processes.

The scripts' training is dispatch-bound: a small network on mini-batches of 32
windows keeps one CPU core busy, not the whole machine. Independent seeds therefore
scale close to linearly with the number of processes.

Workers are spawned, not forked, since JAX is multithreaded and not fork-safe. A job's
function and arguments must be picklable: a module-level function, and plain data such
as configurations and metrics. A trained model is not picklable, so a job returns its
metrics, and anything that needs the model (a figure, say) runs in the main process.
"""

from __future__ import annotations

import multiprocessing
import os
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor

WORKER_ENVIRONMENT = {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
"""Set for the workers: on a GPU, each would otherwise reserve 75% of its memory."""


def spawn_pool(jobs: int) -> ProcessPoolExecutor:
    """A pool of `jobs` spawned worker processes.

    Sets `WORKER_ENVIRONMENT` in this process's environment, unless already set, so
    the workers inherit it; JAX, already imported here, is unaffected.

    Args:
        jobs: Number of worker processes; positive.

    Returns:
        The pool; use it as a context manager to shut it down.
    """
    for name, value in WORKER_ENVIRONMENT.items():
        os.environ.setdefault(name, value)
    return ProcessPoolExecutor(
        max_workers=jobs, mp_context=multiprocessing.get_context("spawn")
    )


def process_map[T, R](
    function: Callable[[T], R], items: Iterable[T], *, jobs: int
) -> list[R]:
    """Apply `function` to every item, in up to `jobs` processes.

    Args:
        function: A picklable function of one item.
        items: The picklable items.
        jobs: Number of processes; `1` runs every item in this process, in order.

    Returns:
        The results, in item order.

    Raises:
        ValueError: If `jobs` is not positive.
    """
    if jobs <= 0:
        raise ValueError("jobs must be positive.")
    items = list(items)
    if jobs == 1 or len(items) <= 1:
        return [function(item) for item in items]
    with spawn_pool(min(jobs, len(items))) as pool:
        return list(pool.map(function, items))


def add_jobs_argument(parser) -> None:
    """Add a `--jobs` option, the number of processes, to an argument parser.

    Args:
        parser: The script's `argparse.ArgumentParser`.
    """
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help=(
            "Number of processes to train in, one run per process at a time "
            "(default: 1, no parallelism). Up to the number of CPU cores helps."
        ),
    )
