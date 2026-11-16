"""Tests for `process_map`, the scripts' process-parallel map."""

from __future__ import annotations

import operator
import os

import pytest

from ifac_safeprocess_2027.parallel import process_map


def test_process_map_keeps_item_order_across_workers() -> None:
    assert process_map(operator.neg, range(7), jobs=3) == [0, -1, -2, -3, -4, -5, -6]


def test_process_map_runs_in_other_processes_when_jobs_exceed_one() -> None:
    worker_pids = process_map(_pid_of, range(4), jobs=2)
    assert os.getpid() not in worker_pids


def test_process_map_with_one_job_runs_here_without_pickling() -> None:
    # A lambda cannot be pickled, so it can only run in this process.
    assert process_map(lambda item: (item, os.getpid()), [1, 2], jobs=1) == [
        (1, os.getpid()),
        (2, os.getpid()),
    ]


@pytest.mark.parametrize("jobs", [0, -1])
def test_process_map_rejects_non_positive_jobs(jobs: int) -> None:
    with pytest.raises(ValueError, match="jobs must be positive"):
        process_map(operator.neg, [1], jobs=jobs)


def _pid_of(item: int) -> int:
    """The process id of the process running this, ignoring `item`."""
    del item
    return os.getpid()
