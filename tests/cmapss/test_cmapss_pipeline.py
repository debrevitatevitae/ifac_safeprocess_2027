"""Tests for the CMAPSS data preparation, training and evaluation pipeline."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

# The example module itself arrives through the `cmapss_pipeline` fixture; these
# names are used directly by the test bodies, so they stay module-level.
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
nnx = pytest.importorskip("flax.nnx")
pytest.importorskip("optax")


def test_staged_mcd_pipeline_and_evaluate_test_metrics_end_to_end(
    cmapss_pipeline: ModuleType,
    tiny_cmapss_config: Any,
    synthetic_windows: Callable[..., tuple[np.ndarray, np.ndarray]],
) -> None:
    rng = np.random.default_rng(0)
    train_windows, train_targets = synthetic_windows(rng, 64)
    validation_windows, validation_targets = synthetic_windows(rng, 16)
    test_windows, test_rul = synthetic_windows(rng, 10)
    config = dataclasses.replace(tiny_cmapss_config, dropout_rate=0.2, num_draws=8)

    _, state = cmapss_pipeline.train_mean_model(
        train_windows,
        train_targets,
        validation_windows,
        validation_targets,
        config,
        event_sinks=(),
        dropout_rate=config.dropout_rate,
    )
    cmapss_pipeline.train_gamma_model(
        state, train_windows, train_targets, config, event_sinks=()
    )
    posterior = cmapss_pipeline.build_staged_mcd_posterior(
        state, train_windows, train_targets, config, event_sinks=()
    )
    model = cmapss_pipeline.refit_gamma_to_posterior_mean(
        posterior, train_windows, train_targets, config, event_sinks=()
    )
    metrics = cmapss_pipeline.evaluate_test_metrics(model, test_windows, test_rul)

    assert set(metrics) == {"rmse", "coverage", "closed_form_crps"}
    assert math.isfinite(metrics["rmse"])
    assert math.isfinite(metrics["closed_form_crps"])
    assert metrics["rmse"] >= 0.0
    assert 0.0 <= metrics["coverage"] <= 1.0
    assert metrics["closed_form_crps"] >= 0.0


class _RecordingSink:
    """An event sink that keeps every training event it receives."""

    def __init__(self) -> None:
        """Start with no recorded events."""
        self.events: list[Any] = []

    def on_event(self, event: Any) -> None:
        """Record one training event.

        Args:
            event: The training event a stage emitted.
        """
        self.events.append(event)


def test_training_functions_report_only_to_the_event_sinks_they_are_given(
    cmapss_pipeline: ModuleType,
    tiny_cmapss_config: Any,
    synthetic_windows: Callable[..., tuple[np.ndarray, np.ndarray]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    rng = np.random.default_rng(3)
    train_windows, train_targets = synthetic_windows(rng, 16)
    validation_windows, validation_targets = synthetic_windows(rng, 8)
    config = dataclasses.replace(
        tiny_cmapss_config, batch_size=8, mean_epochs=2, variance_epochs=3
    )
    mean_sink, gamma_sink = _RecordingSink(), _RecordingSink()

    _, state = cmapss_pipeline.train_mean_model(
        train_windows,
        train_targets,
        validation_windows,
        validation_targets,
        config,
        event_sinks=[mean_sink],
        dropout_rate=0.0,
    )
    cmapss_pipeline.train_gamma_model(
        state, train_windows, train_targets, config, event_sinks=[gamma_sink]
    )

    assert capsys.readouterr().out == ""
    mean_train_steps = [e.step for e in mean_sink.events if e.split == "train"]
    gamma_train_steps = [e.step for e in gamma_sink.events if e.split == "train"]
    assert len(mean_train_steps) == 2
    assert len(gamma_train_steps) == 3
    assert any(e.split == "validation" for e in mean_sink.events)


class _ConstantPosteriorModel(nnx.Module):
    """A one-draw posterior predictive, the same Gaussian for every window."""

    def __init__(self, loc: float, scale: float) -> None:
        """Fix the predicted distribution.

        Args:
            loc: The predicted mean for every window.
            scale: The predicted standard deviation for every window.
        """
        self.loc = loc
        self.scale = scale

    def __call__(self, inputs: Any) -> Any:
        """Predict the fixed one-draw posterior predictive for every window.

        Args:
            inputs: Windowed sensor readings, shape `(batch, window_length,
                n_sensors)`.

        Returns:
            A `PosteriorPredictive` with one draw at `loc` and aleatoric
            variance `scale**2`, each of shape `(1, batch, 1)`.
        """
        from stagereg.jax import PosteriorPredictive

        shape = (1, inputs.shape[0], 1)
        return PosteriorPredictive(
            draws=jnp.full(shape, self.loc),
            aleatoric_variance=jnp.full(shape, self.scale**2),
        )


def test_evaluate_test_metrics_scores_a_posterior_predictive(
    cmapss_pipeline: ModuleType,
) -> None:
    # A one-draw posterior is a standard normal; scored at its own mean: RMSE 0, the target inside
    # every interval, and the closed-form CRPS of N(0, 1) at 0, which is
    # 2 * phi(0) - 1 / sqrt(pi) = (sqrt(2) - 1) / sqrt(pi).
    test_windows = np.zeros((4, 30, 9))
    test_rul = np.zeros(4)

    metrics = cmapss_pipeline.evaluate_test_metrics(
        _ConstantPosteriorModel(loc=0.0, scale=1.0), test_windows, test_rul
    )

    assert metrics["rmse"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["coverage"] == pytest.approx(1.0)
    assert metrics["closed_form_crps"] == pytest.approx(
        (math.sqrt(2.0) - 1.0) / math.sqrt(math.pi), rel=1e-5
    )


def test_evaluate_test_metrics_accepts_extra_epoch_metrics(
    cmapss_pipeline: ModuleType,
) -> None:
    from stagereg.core import NegativeLogLikelihood

    test_windows = np.zeros((4, 30, 9))
    test_rul = np.zeros(4)

    metrics = cmapss_pipeline.evaluate_test_metrics(
        _ConstantPosteriorModel(loc=0.0, scale=1.0),
        test_windows,
        test_rul,
        epoch_metrics=(*cmapss_pipeline.TEST_METRICS, NegativeLogLikelihood()),
    )

    assert set(metrics) == {"rmse", "coverage", "closed_form_crps", "nll"}
    # NLL of N(0, 1) at its mean: log(2 * pi) / 2.
    assert metrics["nll"] == pytest.approx(0.5 * math.log(2.0 * math.pi), rel=1e-5)


def test_cmapss_config_rejects_invalid_values(cmapss_pipeline: ModuleType) -> None:
    with pytest.raises(ValueError, match="window_length and batch_size"):
        cmapss_pipeline.CmapssConfig(window_length=0)
    with pytest.raises(ValueError, match="validation_fraction"):
        cmapss_pipeline.CmapssConfig(validation_fraction=1.5)
    with pytest.raises(ValueError, match="mean_epochs and variance_epochs"):
        cmapss_pipeline.CmapssConfig(mean_epochs=0)
    with pytest.raises(ValueError, match="mean_epochs and variance_epochs"):
        cmapss_pipeline.CmapssConfig(variance_epochs=0)


@given(joint_epochs=st.integers(max_value=0))
def test_cmapss_config_rejects_non_positive_joint_epochs(
    cmapss_pipeline: ModuleType, joint_epochs: int
) -> None:
    with pytest.raises(ValueError, match="joint_epochs"):
        cmapss_pipeline.CmapssConfig(joint_epochs=joint_epochs)


def test_cmapss_config_gives_the_joint_model_the_same_default_budget_as_each_stage(
    cmapss_pipeline: ModuleType,
) -> None:
    config = cmapss_pipeline.CmapssConfig()

    assert config.mean_epochs == config.variance_epochs == 100
    assert config.joint_epochs == 100


def test_cycles_predictive_maps_a_posterior_predictive_back_to_cycles(
    cmapss_pipeline: ModuleType,
) -> None:
    from stagereg.jax import PosteriorPredictive

    mask = jnp.array([True, False])

    class FixedPosterior(nnx.Module):
        def __call__(self, inputs: Any) -> Any:
            return PosteriorPredictive(
                draws=jnp.array([[[0.0]], [[1.0]]]),
                aleatoric_variance=jnp.array([[0.25]]),
                draw_mask=mask,
            )

    standardization = cmapss_pipeline.TargetStandardization(mean=50.0, scale=10.0)

    output = cmapss_pipeline.CyclesPredictive(FixedPosterior(), standardization)(None)

    assert isinstance(output, PosteriorPredictive)
    np.testing.assert_allclose(output.draws, [[[50.0]], [[60.0]]])
    np.testing.assert_allclose(output.aleatoric_variance, [[25.0]])
    np.testing.assert_array_equal(output.draw_mask, mask)


def test_cmapss_config_accepts_the_mc_dropout_defaults(
    cmapss_pipeline: ModuleType,
) -> None:
    config = cmapss_pipeline.CmapssConfig()

    assert config.dropout_rate == 0.5
    assert config.num_draws == 100


@given(dropout_rate=st.floats(min_value=0.0, max_value=0.999))
def test_cmapss_config_accepts_a_dropout_rate_in_the_unit_interval(
    cmapss_pipeline: ModuleType, dropout_rate: float
) -> None:
    assert cmapss_pipeline.CmapssConfig(dropout_rate=dropout_rate).dropout_rate == (
        dropout_rate
    )


@given(
    dropout_rate=st.one_of(
        st.floats(max_value=0.0, exclude_max=True, allow_nan=False),
        st.floats(min_value=1.0, allow_nan=False),
    )
)
def test_cmapss_config_rejects_a_dropout_rate_outside_the_unit_interval(
    cmapss_pipeline: ModuleType, dropout_rate: float
) -> None:
    with pytest.raises(ValueError, match="dropout_rate"):
        cmapss_pipeline.CmapssConfig(dropout_rate=dropout_rate)


@given(num_draws=st.integers(max_value=0))
def test_cmapss_config_rejects_a_non_positive_num_draws(
    cmapss_pipeline: ModuleType, num_draws: int
) -> None:
    with pytest.raises(ValueError, match="num_draws"):
        cmapss_pipeline.CmapssConfig(num_draws=num_draws)


def test_cmapss_config_gives_the_gamma_model_a_narrow_weight_decayed_default(
    cmapss_pipeline: ModuleType,
) -> None:
    config = cmapss_pipeline.CmapssConfig()

    assert config.gamma_hidden_channels < config.hidden_channels
    assert config.gamma_weight_decay > 0.0


@given(channels=st.integers(max_value=0))
def test_cmapss_config_rejects_non_positive_gamma_hidden_channels(
    cmapss_pipeline: ModuleType, channels: int
) -> None:
    with pytest.raises(ValueError, match="gamma_hidden_channels"):
        cmapss_pipeline.CmapssConfig(gamma_hidden_channels=channels)


@given(weight_decay=st.floats(max_value=0.0, exclude_max=True, allow_nan=False))
def test_cmapss_config_rejects_a_negative_gamma_weight_decay(
    cmapss_pipeline: ModuleType, weight_decay: float
) -> None:
    with pytest.raises(ValueError, match="gamma_weight_decay"):
        cmapss_pipeline.CmapssConfig(gamma_weight_decay=weight_decay)


def _train_gamma(
    cmapss_pipeline: ModuleType,
    config: Any,
    synthetic_windows: Callable[..., tuple[np.ndarray, np.ndarray]],
) -> Any:
    rng = np.random.default_rng(5)
    train_windows, train_targets = synthetic_windows(rng, 32)
    validation_windows, validation_targets = synthetic_windows(rng, 8)
    _, state = cmapss_pipeline.train_mean_model(
        train_windows,
        train_targets,
        validation_windows,
        validation_targets,
        config,
        event_sinks=(),
        dropout_rate=0.0,
    )
    return cmapss_pipeline.train_gamma_model(
        state, train_windows, train_targets, config, event_sinks=()
    )


def test_train_gamma_model_is_gamma_hidden_channels_wide(
    cmapss_pipeline: ModuleType,
    tiny_cmapss_config: Any,
    synthetic_windows: Callable[..., tuple[np.ndarray, np.ndarray]],
) -> None:
    from ifac_safeprocess_2027.cmapss.model import Cnn1DGammaModel

    config = dataclasses.replace(tiny_cmapss_config, gamma_hidden_channels=3)

    trained = _train_gamma(cmapss_pipeline, config, synthetic_windows)

    reference = Cnn1DGammaModel(
        3, hidden_channels=3, kernel_size=config.kernel_size, rngs=nnx.Rngs(0)
    )
    shapes = jax.tree.map(jnp.shape, nnx.state(trained, nnx.Param))
    assert shapes == jax.tree.map(jnp.shape, nnx.state(reference, nnx.Param))


def test_gamma_weight_decay_shrinks_the_gamma_models_weights(
    cmapss_pipeline: ModuleType,
    tiny_cmapss_config: Any,
    synthetic_windows: Callable[..., tuple[np.ndarray, np.ndarray]],
) -> None:
    def squared_norm(weight_decay: float) -> float:
        config = dataclasses.replace(
            tiny_cmapss_config, gamma_weight_decay=weight_decay
        )
        trained = _train_gamma(cmapss_pipeline, config, synthetic_windows)
        leaves = jax.tree.leaves(nnx.state(trained, nnx.Param))
        return float(sum(jnp.sum(jnp.square(leaf)) for leaf in leaves))

    assert squared_norm(100.0) < squared_norm(0.0)


def test_staged_mcd_posterior_stage_does_not_train_the_mean_network(
    cmapss_pipeline: ModuleType,
    tiny_cmapss_config: Any,
    synthetic_windows: Callable[..., tuple[np.ndarray, np.ndarray]],
) -> None:
    config = dataclasses.replace(tiny_cmapss_config, dropout_rate=0.2, num_draws=8)
    rng = np.random.default_rng(5)
    train_windows, train_targets = synthetic_windows(rng, 32)
    validation_windows, validation_targets = synthetic_windows(rng, 8)
    mean_model, state = cmapss_pipeline.train_mean_model(
        train_windows,
        train_targets,
        validation_windows,
        validation_targets,
        config,
        event_sinks=(),
        dropout_rate=config.dropout_rate,
    )
    cmapss_pipeline.train_gamma_model(
        state, train_windows, train_targets, config, event_sinks=()
    )

    predictive = cmapss_pipeline.build_staged_mcd_posterior(
        state, train_windows, train_targets, config, event_sinks=()
    )

    trained = jax.tree.leaves(nnx.state(mean_model, nnx.Param))
    posterior = jax.tree.leaves(state.posterior_state)
    assert len(posterior) == len(trained)
    for posterior_leaf, trained_leaf in zip(posterior, trained, strict=True):
        np.testing.assert_array_equal(posterior_leaf, trained_leaf)
    draws = predictive(jnp.asarray(train_windows, dtype=jnp.float32)).draws
    assert draws.shape[0] == config.num_draws
    # Dropout stays active at prediction: the mask sets give distinct draws.
    assert len({float(draw.sum()) for draw in draws}) > 1


def test_cmapss_config_has_no_predictive_sample_count(
    cmapss_pipeline: ModuleType,
) -> None:
    # The closed-form CRPS draws no predictive samples, so the run has no
    # sample count to configure.
    with pytest.raises(TypeError):
        cmapss_pipeline.CmapssConfig(predictive_sample_count=8)


@pytest.fixture
def short_window_config(cmapss_pipeline: ModuleType) -> Any:
    """Configure a window short enough for the synthetic trajectories."""
    return cmapss_pipeline.CmapssConfig(window_length=10)


@pytest.mark.usefixtures("offline_fd001")
@pytest.mark.parametrize("window_length", [5, 10, 30])
def test_prepared_data_windows_and_targets_are_aligned(
    cmapss_pipeline: ModuleType, window_length: int
) -> None:
    config = cmapss_pipeline.CmapssConfig(window_length=window_length)

    prepared = cmapss_pipeline.prepare_cmapss_windows(config)

    n_sensors = len(cmapss_pipeline._SENSOR_NAMES)
    assert prepared.train_windows.shape[1:] == (config.window_length, n_sensors)
    assert prepared.train_windows.shape[0] == prepared.train_targets.shape[0]
    assert (
        prepared.validation_windows.shape[0] == (prepared.validation_targets.shape[0])
    )
    assert prepared.test_windows.shape[0] == prepared.test_rul.shape[0]


@pytest.mark.usefixtures("offline_fd001")
def test_prepared_data_standardizes_the_targets_with_training_statistics(
    cmapss_pipeline: ModuleType, short_window_config: Any
) -> None:
    prepared = cmapss_pipeline.prepare_cmapss_windows(short_window_config)

    stats = prepared.target_standardization
    np.testing.assert_allclose(prepared.train_targets.mean(), 0.0, atol=1e-8)
    np.testing.assert_allclose(prepared.train_targets.std(ddof=0), 1.0)
    # Validation targets carry the training statistics: back in cycles they
    # are a linear RUL, non-negative integers.
    validation_cycles = prepared.validation_targets * stats.scale + stats.mean
    np.testing.assert_allclose(validation_cycles, np.round(validation_cycles))
    assert validation_cycles.min() == pytest.approx(0.0, abs=1e-8)
    # The test RUL stays in cycles.
    np.testing.assert_array_equal(
        prepared.test_rul, cmapss_pipeline.load_fd001_test_rul()
    )


@pytest.mark.usefixtures("offline_fd001")
def test_prepared_data_is_frozen(
    cmapss_pipeline: ModuleType, short_window_config: Any
) -> None:
    prepared = cmapss_pipeline.prepare_cmapss_windows(short_window_config)

    with pytest.raises(dataclasses.FrozenInstanceError):
        prepared.train_targets = prepared.train_targets  # type: ignore[misc]


@pytest.mark.usefixtures("offline_fd001")
def test_prepared_data_standardizes_every_subset_with_training_statistics(
    cmapss_pipeline: ModuleType,
    short_window_config: Any,
) -> None:
    prepared = cmapss_pipeline.prepare_cmapss_windows(short_window_config)

    stats = prepared.standardization
    assert stats.feature_columns == cmapss_pipeline._SENSOR_NAMES
    # Statistics fitted on the training subset only: standardizing that
    # subset with its own statistics leaves it at zero mean and unit scale.
    standardized_train = prepared.train_trajectories[stats.feature_columns].to_numpy()
    assert np.allclose(standardized_train.mean(axis=0), 0.0, atol=1e-8)
    assert np.allclose(standardized_train.std(axis=0, ddof=0), 1.0, atol=1e-8)
    # The validation trajectories carry those same statistics, not their own.
    raw = cmapss_pipeline.load_fd001_data()
    raw_validation = raw[
        raw["unit_id"].isin(prepared.validation_trajectories["unit_id"].unique())
    ]
    expected = cmapss_pipeline.apply_standardization(raw_validation, stats)
    assert np.allclose(
        prepared.validation_trajectories[stats.feature_columns].to_numpy(),
        expected[stats.feature_columns].to_numpy(),
    )


def test_proxy_variance_nll_scores_predictions_under_the_schedule_scale(
    cmapss_pipeline: ModuleType, cmapss_preprocessing: ModuleType
) -> None:
    from stagereg.core.types import Batch

    standardization = cmapss_preprocessing.TargetStandardization(mean=50.0, scale=25.0)
    schedule = cmapss_preprocessing.ProxyVarianceSchedule(
        initial_scale=100.0,
        reference_rul=200.0,
        floor=5.0,
        target_standardization=standardization,
    )
    # RUL 100 cycles: target 2.0, scale 50/25 = 2.0, residual 1.0.
    # RUL 0 cycles: target -2.0, scale floor 5/25 = 0.2, residual 0.2.
    targets = jnp.array([[2.0], [-2.0]])
    prediction = jnp.array([[3.0], [-1.8]])

    nll = cmapss_pipeline.ProxyVarianceNLL(schedule).per_example(
        prediction, Batch(inputs=prediction, targets=targets)
    )

    half_log_two_pi = 0.5 * math.log(2.0 * math.pi)
    np.testing.assert_allclose(
        nll,
        [
            [0.5 * 0.25 + math.log(2.0) + half_log_two_pi],
            [0.5 * 1.0 + math.log(0.2) + half_log_two_pi],
        ],
        rtol=1e-5,
    )
