"""Tests for the FD001 calibration curve script."""

from __future__ import annotations

from types import ModuleType

import numpy as np


def _seed(
    base: ModuleType, seed: int, joint: list[float], staged: list[float]
) -> object:
    """A seed's comparison holding only the models' calibration curves."""
    return base.SeedComparison(
        seed=seed,
        prepared=None,
        models=(
            base.ModelResult(
                name="joint+mcd",
                label="Joint MC-Dropout model",
                model=None,
                metrics={},
                calibration_curve=np.array(joint),
            ),
            base.ModelResult(
                name="staged+mcd",
                label="Staged MC-Dropout model",
                model=None,
                metrics={},
                calibration_curve=np.array(staged),
            ),
        ),
    )


def test_curves_are_grouped_by_rate_coloured_by_method_and_styled_by_rate(
    cmapss_fd001: ModuleType, cmapss_fd001_calibration: ModuleType
) -> None:
    base = cmapss_fd001
    results_by_rate = {
        0.3: [
            _seed(base, 0, [0.1, 0.2], [0.3, 0.4]),
            _seed(base, 1, [0.5, 0.6], [0.7, 0.8]),
        ],
        0.9: [
            _seed(base, 0, [0.9, 1.0], [0.0, 0.1]),
            _seed(base, 1, [0.2, 0.3], [0.4, 0.5]),
        ],
    }

    columns = cmapss_fd001_calibration.collect_calibration_curves(results_by_rate)

    assert [column.name for column in columns] == [
        "joint+mcd p=0.3",
        "staged+mcd p=0.3",
        "joint+mcd p=0.9",
        "staged+mcd p=0.9",
    ]
    curves = [column.curves for column in columns]
    assert [c.label for c in curves] == [
        "Joint MC-Dropout, p=0.3",
        "Staged MC-Dropout, p=0.3",
        "Joint MC-Dropout, p=0.9",
        "Staged MC-Dropout, p=0.9",
    ]
    assert [c.color for c in curves] == ["#2a78d6", "#eb6834"] * 2
    assert [c.linestyle for c in curves] == ["-", "-", "--", "--"]
    # One row per seed, in seed order.
    np.testing.assert_allclose(curves[2].curves, [[0.9, 1.0], [0.2, 0.3]])


def test_table_has_one_column_per_model_and_rate_and_one_row_per_level(
    cmapss_fd001_calibration: ModuleType, cmapss_plots: ModuleType
) -> None:
    script = cmapss_fd001_calibration
    n_levels = len(script.CALIBRATION_LEVELS)
    columns = [
        script.CalibrationColumn(
            name,
            cmapss_plots.LabelledCalibrationCurves(
                name, np.array([np.full(n_levels, 0.2), np.full(n_levels, 0.4)])
            ),
        )
        for name in ("joint+mcd p=0.3", "joint+mcd p=0.9")
    ]

    table = script.format_calibration_table(columns).splitlines()

    assert table[0].split() == ["Level", "joint+mcd", "p=0.3", "joint+mcd", "p=0.9"]
    assert len(table) == 1 + n_levels
    # Mean 0.3, sample std 0.1414 over the two seeds.
    assert table[-1].split()[0] == "95%"
    assert table[-1].count("0.3000 ± 0.1414") == 2
