"""Tests for the CMAPSS MC-Dropout model comparison entry point."""

from __future__ import annotations

import math
import re
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

pytest.importorskip("jax")
pytest.importorskip("flax.nnx")
pytest.importorskip("optax")

_METRIC_NAMES = {"rmse", "coverage", "closed_form_crps", "nll"}


_MODEL_NAMES = ("joint+mcd", "staged+mcd")

_MODEL_LABELS = (
    "Joint MC-Dropout model",
    "Staged MC-Dropout model",
)


@pytest.fixture(scope="module")
def mcd_config(tiny_cmapss_config: Any) -> Any:
    """The tiny configuration with eight-draw MC-Dropout models."""
    import dataclasses

    from ifac_safeprocess_2027.cmapss.pipeline import with_dropout_rate

    return dataclasses.replace(with_dropout_rate(tiny_cmapss_config, 0.2), num_draws=8)


def _always(prepared: Any) -> Callable[[Any], Any]:
    """Return a `prepare` stand-in that gives every seed the same prepared data."""
    return lambda config: prepared


def test_compare_models_prepares_the_data_once_per_seed_from_that_seed(
    cmapss_fd001: ModuleType,
    mcd_config: Any,
    synthetic_prepared_data: Callable[..., Any],
) -> None:
    import numpy as np

    prepared_seeds: list[int] = []

    def prepare(config: Any) -> Any:
        prepared_seeds.append(config.seed)
        return synthetic_prepared_data(seed=config.seed)

    results = cmapss_fd001.compare_models(mcd_config, seeds=2, prepare=prepare)

    # Each seed re-draws the train/validation split, and keeps the data it
    # was trained and scored on.
    assert prepared_seeds == [3, 4]
    for result in results:
        expected = synthetic_prepared_data(seed=result.seed)
        np.testing.assert_array_equal(
            result.prepared.train_windows, expected.train_windows
        )


def test_compare_models_scores_the_two_mcd_models_once_per_seed(
    cmapss_fd001: ModuleType,
    mcd_config: Any,
    synthetic_prepared_data: Callable[..., Any],
) -> None:
    from stagereg.jax import PosteriorPredictive

    prepared = synthetic_prepared_data()

    results = cmapss_fd001.compare_models(
        mcd_config, seeds=2, prepare=_always(prepared)
    )

    assert [result.seed for result in results] == [3, 4]
    windows = prepared.test_windows
    for result in results:
        assert [(model.name, model.label) for model in result.models] == list(
            zip(_MODEL_NAMES, _MODEL_LABELS, strict=True)
        )
        for model in result.models:
            assert set(model.metrics) == _METRIC_NAMES
            predictive = model.model(windows)
            assert isinstance(predictive, PosteriorPredictive)
            assert predictive.num_draws == mcd_config.num_draws
            # Dropout stays active at prediction: the mask sets give distinct draws.
            assert len({float(draw.sum()) for draw in predictive.draws}) > 1
        # The joint MC-Dropout model's draws each keep their own aleatoric
        # variance; the staged ones share the Gamma stage's.
        joint_predictive = result.models[0].model(windows)
        assert joint_predictive.aleatoric_variance.shape == (
            joint_predictive.draws.shape
        )


def test_compare_models_trains_silently_and_reproduces_a_seed(
    cmapss_fd001: ModuleType,
    mcd_config: Any,
    synthetic_prepared_data: Callable[..., Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    import numpy as np

    prepared = synthetic_prepared_data()

    (first,) = cmapss_fd001.compare_models(
        mcd_config, seeds=1, prepare=_always(prepared)
    )
    (second,) = cmapss_fd001.compare_models(
        mcd_config, seeds=1, prepare=_always(prepared)
    )
    _, other_seed = cmapss_fd001.compare_models(
        mcd_config, seeds=2, prepare=_always(prepared)
    )

    assert capsys.readouterr().out == ""
    windows = prepared.test_windows
    for first_model, second_model, other_model in zip(
        first.models, second.models, other_seed.models, strict=True
    ):
        assert first_model.metrics == second_model.metrics
        np.testing.assert_array_equal(
            first_model.model(windows).draws, second_model.model(windows).draws
        )
        assert first_model.metrics != other_model.metrics


@settings(max_examples=4, deadline=None)
@given(
    data_seed=st.integers(min_value=0, max_value=2**16),
    max_rul=st.floats(min_value=2.0, max_value=400.0),
)
def test_compare_models_scores_lie_in_their_ranges(
    cmapss_fd001: ModuleType,
    mcd_config: Any,
    synthetic_prepared_data: Callable[..., Any],
    data_seed: int,
    max_rul: float,
) -> None:
    prepared = synthetic_prepared_data(seed=data_seed, max_rul=max_rul)

    (result,) = cmapss_fd001.compare_models(
        mcd_config, seeds=1, prepare=_always(prepared)
    )

    for metrics in (model.metrics for model in result.models):
        assert 0.0 <= metrics["coverage"] <= 1.0
        assert metrics["rmse"] >= 0.0
        assert metrics["closed_form_crps"] >= 0.0
        assert math.isfinite(metrics["nll"])


def test_compare_models_gives_each_model_a_calibration_curve_matching_its_coverage(
    cmapss_fd001: ModuleType,
    mcd_config: Any,
    synthetic_prepared_data: Callable[..., Any],
) -> None:
    from ifac_safeprocess_2027.cmapss.calibration import CALIBRATION_LEVELS

    prepared = synthetic_prepared_data(seed=0)

    (result,) = cmapss_fd001.compare_models(
        mcd_config, seeds=1, prepare=_always(prepared)
    )

    for model in result.models:
        curve = np.asarray(model.calibration)
        assert curve.shape == (len(CALIBRATION_LEVELS),)
        # Central intervals are nested, so coverage never drops as the level rises.
        assert np.all(np.diff(curve) >= 0.0)
        assert np.all((0.0 <= curve) & (curve <= 1.0))
        # Its 95% point is the 95% coverage the test metrics report.
        assert curve[CALIBRATION_LEVELS.index(0.95)] == pytest.approx(
            model.metrics["coverage"]
        )


def test_the_joint_model_trains_for_joint_epochs(
    cmapss_fd001: ModuleType,
    mcd_config: Any,
    synthetic_prepared_data: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dataclasses

    config = dataclasses.replace(
        mcd_config, mean_epochs=2, variance_epochs=3, joint_epochs=4
    )
    trained_epochs: list[int] = []
    run_supervised = cmapss_fd001.run_supervised

    def record_run_supervised(**kwargs: Any) -> Any:
        trained_epochs.append(kwargs["epochs"])
        return run_supervised(**kwargs)

    monkeypatch.setattr(cmapss_fd001, "run_supervised", record_run_supervised)

    cmapss_fd001.compare_models(
        config, seeds=1, prepare=_always(synthetic_prepared_data())
    )

    # Only joint+mcd trains through run_supervised, independently of the
    # stages' epochs.
    assert trained_epochs == [4]


def test_compare_models_predict_in_cycles(
    cmapss_fd001: ModuleType,
    mcd_config: Any,
    synthetic_prepared_data: Callable[..., Any],
) -> None:
    import numpy as np

    # Targets are uniform in [1, 1000) cycles, so their mean is near 500: a
    # model left predicting in standardized units would put its means near 0.
    prepared = synthetic_prepared_data(max_rul=1000.0)

    (result,) = cmapss_fd001.compare_models(
        mcd_config, seeds=1, prepare=_always(prepared)
    )

    for model in result.models:
        mean = model.model(prepared.test_windows).mean()
        assert float(np.mean(mean)) > 100.0, model.name


def test_mcd_models_without_dropout_collapse_to_a_single_draw(
    cmapss_fd001: ModuleType,
    mcd_config: Any,
    synthetic_prepared_data: Callable[..., Any],
) -> None:
    import numpy as np

    # Without dropout every mask set is the identity, so every draw, and every
    # draw's aleatoric variance, is the same.
    from ifac_safeprocess_2027.cmapss.pipeline import with_dropout_rate

    config = with_dropout_rate(mcd_config, 0.0)
    prepared = synthetic_prepared_data()

    (result,) = cmapss_fd001.compare_models(config, seeds=1, prepare=_always(prepared))

    for model in result.models:
        predictive = model.model(prepared.test_windows)
        variances = np.broadcast_to(
            predictive.aleatoric_variance, predictive.draws.shape
        )
        for draw, variance in zip(predictive.draws, variances, strict=True):
            np.testing.assert_allclose(draw, predictive.draws[0], rtol=1e-5)
            np.testing.assert_allclose(variance, variances[0], rtol=1e-5)


def _fixed_posterior_model(low: float, high: float, scales: tuple[float, float]) -> Any:
    """Return a stand-in model mapping any windows to the same two-draw mixture.

    Draw one centres every unit at `low`, draw two at `high`, each with its own
    aleatoric standard deviation, so the predictive is a bimodal mixture.
    """
    import jax.numpy as jnp
    from flax import nnx
    from stagereg.jax import PosteriorPredictive

    class FixedPosteriorModel(nnx.Module):
        def __call__(self, windows: Any) -> Any:
            shape = (windows.shape[0], 1)
            draws = jnp.stack([jnp.full(shape, low), jnp.full(shape, high)])
            variance = jnp.stack([jnp.full(shape, scale**2) for scale in scales])
            return PosteriorPredictive(draws=draws, aleatoric_variance=variance)

    return FixedPosteriorModel()


def test_score_on_test_split_scores_a_posterior_predictive_on_its_exact_mixture(
    cmapss_fd001: ModuleType,
    synthetic_prepared_data: Callable[..., Any],
) -> None:
    import dataclasses

    import numpy as np

    # Targets at the two modes and at the trough between them.
    test_rul = np.array([0.0, 10.0, 5.0], dtype=np.float32)
    prepared = synthetic_prepared_data()
    prepared = dataclasses.replace(
        prepared, test_windows=prepared.test_windows[:3], test_rul=test_rul
    )
    model = _fixed_posterior_model(low=0.0, high=10.0, scales=(1.0, 2.0))

    metrics = cmapss_fd001.score_on_test_split(model, prepared)

    assert set(metrics) == _METRIC_NAMES

    def normal_pdf(y: float, loc: float, scale: float) -> float:
        return math.exp(-0.5 * ((y - loc) / scale) ** 2) / (
            scale * math.sqrt(2 * math.pi)
        )

    mixture_nll = -np.mean(
        [
            math.log(0.5 * normal_pdf(y, 0.0, 1.0) + 0.5 * normal_pdf(y, 10.0, 2.0))
            for y in test_rul
        ]
    )
    assert metrics["nll"] == pytest.approx(mixture_nll, rel=1e-4)
    # The mixture mean is 5 everywhere: residuals 5, 5 and 0.
    assert metrics["rmse"] == pytest.approx(math.sqrt(50 / 3), rel=1e-5)
    # The central 95% mixture interval holds both modes and the trough; the
    # moment-matched Gaussian's would hold them too, so coverage is 1.
    assert metrics["coverage"] == pytest.approx(1.0)


@settings(max_examples=5, deadline=None)
@given(
    loc=st.floats(min_value=-50.0, max_value=50.0),
    scale=st.floats(min_value=0.5, max_value=20.0),
)
def test_score_on_test_split_scores_a_one_point_posterior_as_its_gaussian(
    cmapss_fd001: ModuleType,
    synthetic_prepared_data: Callable[..., Any],
    loc: float,
    scale: float,
) -> None:
    import numpy as np
    from scipy import stats

    prepared = synthetic_prepared_data()
    model = _fixed_posterior_model(low=loc, high=loc, scales=(scale, scale))

    metrics = cmapss_fd001.score_on_test_split(model, prepared)

    rul = np.asarray(prepared.test_rul, dtype=float)
    z = (rul - loc) / scale
    # The closed-form CRPS of a Gaussian (Gneiting & Raftery, 2007).
    crps = scale * (
        z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / math.sqrt(math.pi)
    )
    assert metrics["rmse"] == pytest.approx(
        math.sqrt(np.mean((rul - loc) ** 2)), rel=1e-4
    )
    assert metrics["nll"] == pytest.approx(
        -np.mean(stats.norm.logpdf(rul, loc, scale)), rel=1e-4
    )
    assert metrics["closed_form_crps"] == pytest.approx(np.mean(crps), rel=1e-3)
    assert metrics["coverage"] == pytest.approx(
        np.mean(np.abs(z) <= stats.norm.ppf(0.975))
    )


_CELL = r"-?\d+\.\d+ ± \d+\.\d+"


def test_format_comparison_table_gives_each_model_a_mean_and_sample_std_column(
    cmapss_fd001: ModuleType,
) -> None:
    def seed(staged_rmse: float, joint_rmse: float) -> Any:
        def model(name: str, rmse: float) -> Any:
            metrics = {"rmse": rmse, "coverage": 1.0, "closed_form_crps": 0.0}
            return cmapss_fd001.ModelResult(
                name=name, label=name, model=None, metrics={**metrics, "nll": 0.0}
            )

        return cmapss_fd001.SeedComparison(
            seed=0,
            prepared=None,
            models=(model("staged+mcd", staged_rmse), model("joint+mcd", joint_rmse)),
        )

    table = cmapss_fd001.format_comparison_table([seed(1.0, 10.0), seed(3.0, 10.0)])

    header, rmse, *_ = table.splitlines()
    assert header.split() == ["staged+mcd", "joint+mcd"]
    # Mean 2 and sample std sqrt(2) for the staged model; no spread for the joint.
    assert rmse.split() == ["RMSE", "2.0000", "±", "1.4142", "10.0000", "±", "0.0000"]


def test_format_comparison_table_gives_one_column_per_model_result_in_order(
    cmapss_fd001: ModuleType,
) -> None:
    names = ("joint+mcd", "staged+mcd")

    def model(name: str, nll: float) -> Any:
        metrics = {"rmse": 0.0, "coverage": 1.0, "closed_form_crps": 0.0, "nll": nll}
        return cmapss_fd001.ModelResult(
            name=name, label=name, model=None, metrics=metrics
        )

    seed = cmapss_fd001.SeedComparison(
        seed=0,
        prepared=None,
        models=tuple(model(name, float(index)) for index, name in enumerate(names)),
    )

    table = cmapss_fd001.format_comparison_table([seed])

    header, *rows = table.splitlines()
    assert tuple(header.split()) == names
    (nll,) = [row for row in rows if row.startswith("NLL")]
    assert nll.split()[1::3] == ["0.0000", "1.0000"]


@pytest.mark.usefixtures("offline_fd001")
@pytest.mark.parametrize(
    ("seeds_argv", "seeds", "plot_name"),
    [([], 5, None), (["--seeds", "2"], 2, "comparison.png")],
)
def test_main_prints_one_line_per_run_and_a_mean_std_table_then_plots(
    cmapss_pipeline: ModuleType,
    cmapss_fd001: ModuleType,
    mcd_config: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    plot_recorder: Any,
    seeds_argv: list[str],
    seeds: int,
    plot_name: str | None,
) -> None:
    plot_path = tmp_path / plot_name if plot_name else None
    plot_argv = ["--plot-path", str(plot_path)] if plot_path else []
    monkeypatch.setattr(sys, "argv", ["fd001_base.py", *seeds_argv, *plot_argv])
    monkeypatch.setattr(cmapss_fd001, "CmapssConfig", lambda: mcd_config)
    prepared_with: list[Any] = []
    prepare = cmapss_fd001.prepare_cmapss_windows

    def record_prepare(config: Any) -> Any:
        prepared_with.append(config)
        return prepare(config)

    monkeypatch.setattr(cmapss_fd001, "prepare_cmapss_windows", record_prepare)
    compared: list[Any] = []
    compare = cmapss_fd001.compare_models

    def record_compare(*args: Any, **kwargs: Any) -> Any:
        results = compare(*args, **kwargs)
        compared.extend(results)
        return results

    monkeypatch.setattr(cmapss_fd001, "compare_models", record_compare)
    monkeypatch.setattr(cmapss_fd001, "plot_validation_rul_curves", plot_recorder)

    cmapss_fd001.main()

    # The data is prepared once per seed, from that seed.
    assert [config.seed for config in prepared_with] == list(
        range(mcd_config.seed, mcd_config.seed + seeds)
    )
    # One figure, drawn from the first seed's models, one row per model.
    (plot,) = plot_recorder.calls
    assert [model.label for model in plot.models] == list(_MODEL_LABELS)
    for plotted, result in zip(plot.models, compared[0].models, strict=True):
        assert plotted.model is result.model
    assert plot.save_path == plot_path
    assert plot.window_length == mcd_config.window_length
    assert plot.feature_columns == cmapss_pipeline._SENSOR_NAMES
    # The lifetime-spanning units of the first seed's train split: in-sample
    # for every model, as the joint model trains on train and validation.
    first_train = compared[0].prepared.train_trajectories
    assert plot.trajectories is first_train
    assert plot.units == cmapss_fd001.select_lifetime_spanning_units(first_train)
    # Every run line and the table are printed before the figure is drawn.
    lines = plot.printed.splitlines()
    expected_seeds = [
        f"seed={seed}]" for seed in range(mcd_config.seed, mcd_config.seed + seeds)
    ]
    run_lines = [line for line in lines if line.startswith("[")]
    assert [tuple(line.split()[:2]) for line in run_lines] == [
        (f"[{name}", seed) for seed in expected_seeds for name in _MODEL_NAMES
    ]
    for line in run_lines:
        for name in _METRIC_NAMES:
            assert f"{name}=" in line
    (header,) = [
        line for line in lines if re.search(r"joint\+mcd\s+staged\+mcd\s*$", line)
    ]
    table = lines[lines.index(header) + 1 :]
    for row in ("RMSE", "95% coverage", "closed-form CRPS", "NLL"):
        (line,) = [line for line in table if line.startswith(row)]
        assert re.fullmatch(rf"{re.escape(row)}(\s+{_CELL}){{2}}", line)


def test_the_staged_gamma_fits_train_on_the_held_out_residuals(
    cmapss_fd001: ModuleType,
    mcd_config: Any,
    synthetic_prepared_data: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = synthetic_prepared_data()
    mean_fitted_on: list[Any] = []
    gamma_fitted_on: list[Any] = []

    def record(name: str, position: int, into: list[Any]) -> None:
        original = getattr(cmapss_fd001, name)

        def recorded(*args: Any, **kwargs: Any) -> Any:
            into.append(args[position])
            return original(*args, **kwargs)

        monkeypatch.setattr(cmapss_fd001, name, recorded)

    record("train_mean_model", 0, mean_fitted_on)
    record("train_gamma_model", 1, gamma_fitted_on)
    record("refit_gamma_to_posterior_mean", 1, gamma_fitted_on)

    cmapss_fd001.compare_models(mcd_config, seeds=1, prepare=_always(prepared))

    # The staged model: the mean on the train split, and both Gamma fits
    # (after the mean stage, and the refit to the posterior mean) on the
    # validation split, whose residuals the mean never trained on.
    assert [windows is prepared.train_windows for windows in mean_fitted_on] == [True]
    assert [windows is prepared.validation_windows for windows in gamma_fitted_on] == [
        True
    ] * 2


def test_the_joint_model_trains_on_train_and_validation_without_monitoring(
    cmapss_fd001: ModuleType,
    mcd_config: Any,
    synthetic_prepared_data: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = synthetic_prepared_data()
    runs: list[dict[str, Any]] = []
    run_supervised = cmapss_fd001.run_supervised

    def record_run_supervised(**kwargs: Any) -> Any:
        runs.append(kwargs)
        return run_supervised(**kwargs)

    monkeypatch.setattr(cmapss_fd001, "run_supervised", record_run_supervised)

    cmapss_fd001.compare_models(mcd_config, seeds=1, prepare=_always(prepared))

    (run,) = runs
    assert run["validation"] is None
    batches = run["train_loader"](split="train", epoch=0)
    # The same windows as the staged models see, train and validation.
    n_windows = len(prepared.train_windows) + len(prepared.validation_windows)
    assert sum(batch.inputs.shape[0] for batch in batches) == n_windows
