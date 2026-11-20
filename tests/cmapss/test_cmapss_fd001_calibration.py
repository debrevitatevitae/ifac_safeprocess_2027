"""Tests for the FD001 calibration curve script."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np


def _seed(base: ModuleType, seed: int, joint: list[float], staged: list[float]) -> Any:
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


def test_curves_are_grouped_by_rate_coloured_by_model_and_styled_by_rate(
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

    calibrations = cmapss_fd001_calibration.collect_calibration_curves(results_by_rate)

    assert [calibration.name for calibration in calibrations] == [
        "joint+mcd p=0.3",
        "staged+mcd p=0.3",
        "joint+mcd p=0.9",
        "staged+mcd p=0.9",
    ]
    curves = [calibration.labelled for calibration in calibrations]
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
    calibrations = [
        script.RateCalibration(
            name,
            cmapss_plots.LabelledCalibrationCurves(
                name, np.array([np.full(n_levels, 0.2), np.full(n_levels, 0.4)])
            ),
        )
        for name in ("joint+mcd p=0.3", "joint+mcd p=0.9")
    ]

    table = script.format_calibration_table(calibrations).splitlines()

    assert table[0].split() == ["Level", "joint+mcd", "p=0.3", "joint+mcd", "p=0.9"]
    assert len(table) == 1 + n_levels
    # Mean 0.3, sample std 0.1414 over the two seeds.
    assert table[-1].split()[0] == "95%"
    assert table[-1].count("0.3000 ± 0.1414") == 2


def test_line_style_follows_the_rate_value_not_its_position(
    cmapss_fd001: ModuleType, cmapss_fd001_calibration: ModuleType
) -> None:
    results_by_rate = {
        rate: [_seed(cmapss_fd001, 0, [0.1], [0.2])] for rate in (0.9, 0.5, 0.3)
    }

    styles = {
        calibration.name: calibration.labelled.linestyle
        for calibration in cmapss_fd001_calibration.collect_calibration_curves(
            results_by_rate
        )
    }

    assert styles["joint+mcd p=0.3"] == "-"
    assert styles["joint+mcd p=0.9"] == "--"
    assert styles["joint+mcd p=0.5"] == "-."


def test_every_rate_and_seed_trains_once_from_the_same_seeds_grouped_by_rate(
    cmapss_fd001_calibration: ModuleType, monkeypatch: Any
) -> None:
    script = cmapss_fd001_calibration
    trained: list[tuple[float, int]] = []

    def fake_train(config: Any, prepare: Any, on_run: Any) -> tuple[float, int]:
        run = (config.joint_net_config.dropout_rate, config.seed)
        assert config.mean_net_config.dropout_rate == run[0]
        trained.append(run)
        return run

    monkeypatch.setattr(script, "_train_and_score_seed_in_worker", fake_train)
    config = script.CmapssConfig()

    results_by_rate = script.train_over_rates(
        config, seeds=2, rates=[0.9, 0.3, 0.9], prepare=None, jobs=1
    )

    seeds = [config.seed, config.seed + 1]
    # Repeated rates train once; the mapping keeps the rates' first order.
    assert list(results_by_rate) == [0.9, 0.3]
    assert results_by_rate[0.9] == [(0.9, seed) for seed in seeds]
    assert results_by_rate[0.3] == [(0.3, seed) for seed in seeds]
    assert sorted(trained) == sorted(
        [(0.9, s) for s in seeds] + [(0.3, s) for s in seeds]
    )


def test_a_run_line_names_its_model_rate_and_seed(
    cmapss_fd001_calibration: ModuleType, capsys: Any
) -> None:
    cmapss_fd001_calibration.print_rate_run(
        0.3, "joint+mcd", 2, {"rmse": 1.0, "nll": 2.5}
    )

    assert (
        capsys.readouterr().out == "[joint+mcd p=0.3 seed=2] rmse=1.0000, nll=2.5000\n"
    )
