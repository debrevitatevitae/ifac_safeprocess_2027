"""Tests for the CMAPSS calibration curve."""

from __future__ import annotations

import importlib
from types import ModuleType

import numpy as np
import pytest


@pytest.fixture(scope="module")
def cmapss_calibration() -> ModuleType:
    """The calibration module, skipped without its JAX dependencies."""
    pytest.importorskip("stagereg.jax")
    return importlib.import_module("ifac_safeprocess_2027.cmapss.calibration")


def _standard_normal(n_windows: int) -> object:
    """A one-draw posterior predictive: a standard normal at every window."""
    stagereg_jax = pytest.importorskip("stagereg.jax")
    return stagereg_jax.PosteriorPredictive(
        draws=np.zeros((1, n_windows, 1), dtype=np.float32),
        aleatoric_variance=np.ones((n_windows, 1), dtype=np.float32),
    )


def test_coverage_curve_counts_targets_inside_each_central_interval(
    cmapss_calibration: ModuleType,
) -> None:
    # Central-interval half-widths of a standard normal: 0.126 at 10%,
    # 0.674 at 50%, 1.645 at 90% and 1.960 at 95%.
    targets = np.array([0.0, 0.5, 1.0, 2.0])

    coverage = cmapss_calibration.interval_coverage_curve(
        _standard_normal(targets.size), targets, levels=(0.1, 0.5, 0.9, 0.95)
    )

    np.testing.assert_allclose(coverage, [0.25, 0.5, 0.75, 0.75])


@pytest.mark.parametrize("level", [0.0, 1.0, -0.5, 1.5])
def test_coverage_curve_rejects_levels_outside_the_unit_interval(
    cmapss_calibration: ModuleType, level: float
) -> None:
    with pytest.raises(ValueError, match="levels"):
        cmapss_calibration.interval_coverage_curve(
            _standard_normal(2), np.zeros(2), levels=(0.5, level)
        )
