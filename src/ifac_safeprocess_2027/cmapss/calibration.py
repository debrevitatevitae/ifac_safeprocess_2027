"""Calibration curve of a posterior predictive on the CMAPSS test split.

A calibration curve pairs each nominal confidence level with the empirical coverage of
the posterior predictive's central interval at that level: the fraction of targets
inside it. A calibrated model's curve follows the diagonal; one below it is
overconfident, one above it underconfident.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .plots import PosteriorPredictiveDistribution

CALIBRATION_LEVELS: tuple[float, ...] = (
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    0.95,
)
"""The nominal levels of the calibration curve: 10% to 90% in steps of 10%, and the
95% the test coverage metric reports."""


def calibration_curve(
    prediction: PosteriorPredictiveDistribution,
    targets: Any,
    *,
    levels: Sequence[float] = CALIBRATION_LEVELS,
) -> np.ndarray:
    """Return the empirical coverage of the central interval at each nominal level.

    The central interval at level `c` runs from the predictive's
    `(1 - c) / 2` quantile to its `(1 + c) / 2` quantile, both bounds
    included, like the test coverage metric's.

    Args:
        prediction: The posterior predictive over the windows, one per target.
        targets: One target per window, in the prediction's units, any shape
            with `n_windows` entries.
        levels: Nominal confidence levels, each in `(0, 1)`. Defaults to
            `CALIBRATION_LEVELS`.

    Returns:
        The fraction of targets inside each level's central interval, shape
        `(len(levels),)`, in the order of `levels`.

    Raises:
        ValueError: If a level lies outside `(0, 1)`.
    """
    targets = np.asarray(targets, dtype=float).reshape(-1)
    coverage = []
    for level in levels:
        if not 0.0 < level < 1.0:
            raise ValueError(f"levels must lie in (0, 1), got {level}.")
        lower = np.asarray(prediction.quantile((1.0 - level) / 2.0)).reshape(-1)
        upper = np.asarray(prediction.quantile((1.0 + level) / 2.0)).reshape(-1)
        coverage.append(np.mean((lower <= targets) & (targets <= upper)))
    return np.asarray(coverage, dtype=float)
