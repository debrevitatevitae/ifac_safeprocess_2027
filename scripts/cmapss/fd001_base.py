"""Compare the joint and staged MC-Dropout models on CMAPSS FD001 over several seeds.

For each seed, prepares the FD001 data with `prepare_cmapss_windows`,
drawing that seed's train/validation split, then trains three MC-Dropout
models on it. Each trains once with dropout active, its only regularizer,
and predicts with dropout active, as a posterior predictive over fixed
dropout-mask sets:

* the **joint MC-Dropout model** (`joint+mcd`): one 1D-CNN backbone followed
  by a Gaussian head, trained with `run_supervised` on the Gaussian negative
  log-likelihood over the train and validation splits together, with no
  validation monitoring; each draw keeps its own aleatoric variance;
* the **staged MC-Dropout model** (`staged+mcd`): a 1D-CNN mean stage,
  trained on the train split and monitored on the validation split,
  followed by a narrower 1D-CNN Gamma variance stage, trained with weight
  decay on the mean's held-out squared residuals on the validation split;
  then a posterior stage with the `MCDropout` inference method that does
  not train, its posterior being the trained mean network under dropout
  masks. A fresh Gamma network is then re-fitted to the validation split's
  squared residuals of the posterior predictive's mean, and the draws share
  that variance;
* the **staged MC-Dropout proxy-variance model** (`staged+mcd+proxy`): the
  staged MC-Dropout model with its mean stage trained and validated on
  `ProxyVarianceNLL` instead of the squared error. That is a Gaussian NLL
  under a fictitious scale that shrinks linearly with the true RUL, from
  the spread of the training units' lifetimes to
  `proxy_variance_floor_cycles` at failure, so end-of-life windows weigh
  most and the dropout spread is pushed down there.

All three train on the standardized linear-RUL targets, and their
predictions are mapped back to RUL cycles to be scored against the official
FD001 test split on RMSE, 95% interval coverage, closed-form CRPS and NLL,
each on the exact mixture. Every seed re-draws the train/validation split,
re-fits the standardization on its train split, and re-initializes the
models and their dropout masks.

It then draws the first seed's RUL curves, in cycles, one row per model,
for the lifetime-spanning units of that seed's train split. Every model
trained on those units, so the figure is for visualization only.

Run it with:

    uv run python scripts/cmapss/fd001_base.py --seeds 5

Pass `--plot-path` to save the figure instead of displaying it.
"""

from __future__ import annotations

import argparse
import dataclasses
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from flax import nnx
from stagereg.core import NegativeLogLikelihood, NegativeLogLikelihoodLoss
from stagereg.core.distributions import PredictionLoss
from stagereg.core.metric_registry import EpochMetric
from stagereg.core.tracking import EventSink
from stagereg.jax import (
    initialize_training_state,
    make_supervised_loss,
    run_supervised,
)

from ifac_safeprocess_2027.cmapss.model import Cnn1DGaussianModel
from ifac_safeprocess_2027.cmapss.pipeline import (
    TEST_METRICS,
    CmapssConfig,
    CyclesPredictive,
    PreparedCmapssData,
    ProxyVarianceNLL,
    build_mc_dropout_predictive,
    build_staged_mcd_posterior,
    build_supervised_training_setup,
    evaluate_test_metrics,
    prepare_cmapss_windows,
    refit_gamma_to_posterior_mean,
    train_gamma_model,
    train_mean_model,
)
from ifac_safeprocess_2027.cmapss.plots import LabelledModel, plot_validation_rul_curves
from ifac_safeprocess_2027.cmapss.preprocessing import (
    fit_proxy_variance_schedule,
    select_lifetime_spanning_units,
    unit_lifetimes,
)

COMPARISON_METRICS: tuple[EpochMetric, ...] = (*TEST_METRICS, NegativeLogLikelihood())
"""The comparison's test metrics: RMSE, 95% coverage, closed-form CRPS and NLL."""

JOINT_STAGE = "joint"
"""The stage segment the joint model's training is reported under."""


def train_joint_model(
    prepared: PreparedCmapssData,
    config: CmapssConfig,
    *,
    epochs: int,
    event_sinks: Sequence[EventSink],
    dropout_rate: float,
) -> Cnn1DGaussianModel:
    """Train a joint model with `run_supervised` on the Gaussian NLL.

    The model trains on the train and validation windows together, the same
    data budget as the staged models, whose Gamma stage fits the validation
    split's held-out residuals. Nothing is held out to monitor it. It trains
    on the standardized RUL targets.

    Args:
        prepared: The prepared FD001 windows whose train and validation
            splits it trains on.
        config: Example configuration controlling model size and training;
            `config.seed` sets the model's initialization.
        epochs: Number of training epochs.
        event_sinks: Sinks notified of the training events; pass none to
            train silently.
        dropout_rate: The network's dropout rate, active during training.

    Returns:
        The trained `Cnn1DGaussianModel`.
    """
    windows = np.concatenate([prepared.train_windows, prepared.validation_windows])
    targets = np.concatenate([prepared.train_targets, prepared.validation_targets])
    # The setup's validation loader is left unused.
    setup = build_supervised_training_setup(
        Cnn1DGaussianModel,
        windows,
        targets,
        windows,
        targets,
        config,
        dropout_rate=dropout_rate,
    )
    run_supervised(
        model=setup.model,
        optimizer=setup.optimizer,
        train_loader=setup.train_loader,
        loss=make_supervised_loss(NegativeLogLikelihoodLoss()),
        state=initialize_training_state(
            setup.model, setup.optimizer, rng_key=setup.rng_key
        ),
        epochs=epochs,
        validation=None,
        event_sinks=event_sinks,
        stage=JOINT_STAGE,
    )
    return setup.model


def score_on_test_split(
    model: nnx.Module, prepared: PreparedCmapssData
) -> dict[str, float]:
    """Score a model on the FD001 test split with `COMPARISON_METRICS`.

    The model must predict in RUL cycles, as `CyclesPredictive` does, since
    the test RUL is in cycles.

    The model is scored with `PosteriorPredictivePredictor`, on its exact
    mixture: mixture log-density, mixture-quantile coverage and the
    mixture's closed-form CRPS.

    Args:
        model: A model mapping windows to a `PosteriorPredictive`.
        prepared: The prepared FD001 windows whose test split is scored.

    Returns:
        The model's `COMPARISON_METRICS` on the test split, by metric name.
    """
    return evaluate_test_metrics(
        model,
        prepared.test_windows,
        prepared.test_rul,
        epoch_metrics=COMPARISON_METRICS,
    )


@dataclass(frozen=True)
class ModelResult:
    """One trained model of a comparison and its test-split scores.

    Attributes:
        name: The model's name in the per-run lines and the table's column
            header, e.g. `"staged+mcd"`.
        label: The model's row label in the figure, e.g. `"Staged
            MC-Dropout model"`.
        model: The trained model, returning a `PosteriorPredictive` in RUL
            cycles.
        metrics: The model's `COMPARISON_METRICS` on the test split, by
            metric name.
    """

    name: str
    label: str
    model: nnx.Module
    metrics: dict[str, float]


@dataclass(frozen=True)
class SeedComparison:
    """The models trained from one seed, and their test-split scores.

    Attributes:
        seed: The seed every model was initialized from, and the split drawn.
        prepared: The FD001 windows prepared from `seed`: its train/validation
            split, and the standardization fitted on its train split.
        models: One result per model, in the table's column order:
            `joint+mcd`, `staged+mcd`, then `staged+mcd+proxy`.
    """

    seed: int
    prepared: PreparedCmapssData
    models: tuple[ModelResult, ...]


RunReporter = Callable[[str, int, dict[str, float]], None]
"""Called as `report(model, seed, metrics)` once each (seed, model) run is scored."""


def _ignore_run(model: str, seed: int, metrics: dict[str, float]) -> None:
    """Report nothing about a finished run.

    Args:
        model: The model's name, ignored.
        seed: The seed the model was initialized from, ignored.
        metrics: The model's test metrics by name, ignored.
    """
    del model, seed, metrics


def _train_joint_mcd(prepared: PreparedCmapssData, config: CmapssConfig) -> nnx.Module:
    """Silently train the joint MC-Dropout model and return its predictive.

    Args:
        prepared: The prepared FD001 windows to train and validate on.
        config: Example configuration; its `seed` initializes the network and
            sets its training masks and the predictive's draws.

    Returns:
        The joint model trained at `config.dropout_rate`, wrapped by
        `build_mc_dropout_predictive`.
    """
    network = train_joint_model(
        prepared,
        config,
        epochs=config.joint_epochs,
        event_sinks=(),
        dropout_rate=config.dropout_rate,
    )
    return build_mc_dropout_predictive(network, config)


def _train_staged_mcd(
    prepared: PreparedCmapssData,
    config: CmapssConfig,
    *,
    objective: PredictionLoss | None = None,
) -> nnx.Module:
    """Silently train the staged MC-Dropout model and return its predictive.

    Trains the mean stage at `config.dropout_rate` on the train split, the
    validation split monitoring it, and the Gamma stage on its held-out
    residuals on the validation split. It then runs the inference-only
    posterior stage and re-fits the Gamma network to the posterior
    predictive's mean, again on its held-out residuals.

    Args:
        prepared: The prepared FD001 windows to train and validate on.
        config: Example configuration; its `seed` initializes the networks and
            sets the mean network's training masks and the predictive's draws.
        objective: The mean stage's per-window objective; `None`, the
            default, is the squared error.

    Returns:
        The posterior's draws under the re-fitted Gamma variance.
    """
    _, state = train_mean_model(
        prepared.train_windows,
        prepared.train_targets,
        prepared.validation_windows,
        prepared.validation_targets,
        config,
        event_sinks=(),
        dropout_rate=config.dropout_rate,
        objective=objective,
    )
    train_gamma_model(
        state,
        prepared.validation_windows,
        prepared.validation_targets,
        config,
        event_sinks=(),
    )
    predictive = build_staged_mcd_posterior(
        state,
        prepared.train_windows,
        prepared.train_targets,
        config,
        event_sinks=(),
    )
    return refit_gamma_to_posterior_mean(
        predictive,
        prepared.validation_windows,
        prepared.validation_targets,
        config,
        event_sinks=(),
    )


def _train_and_score_seed(
    prepared: PreparedCmapssData, config: CmapssConfig, *, on_run: RunReporter
) -> SeedComparison:
    """Train and score every model of the comparison from `config.seed`.

    The models are the joint MC-Dropout, the staged MC-Dropout and the staged
    MC-Dropout proxy-variance model. Each is trained from scratch on the
    standardized targets, wrapped in `CyclesPredictive`, and scored in that
    order.

    Args:
        prepared: The prepared FD001 windows to train, validate and score on.
        config: Example configuration; its `seed` initializes every model.
        on_run: Called with each model's name, the seed and its test metrics
            as soon as that model is scored.

    Returns:
        The seed's `SeedComparison`.
    """

    def score(name: str, label: str, model: nnx.Module) -> ModelResult:
        model = CyclesPredictive(model, prepared.target_standardization)
        metrics = score_on_test_split(model, prepared)
        on_run(name, config.seed, metrics)
        return ModelResult(name=name, label=label, model=model, metrics=metrics)

    joint_mcd_result = score(
        "joint+mcd", "Joint MC-Dropout model", _train_joint_mcd(prepared, config)
    )
    staged_mcd_result = score(
        "staged+mcd", "Staged MC-Dropout model", _train_staged_mcd(prepared, config)
    )
    schedule = fit_proxy_variance_schedule(
        unit_lifetimes(prepared.train_trajectories),
        prepared.target_standardization,
        floor_cycles=config.proxy_variance_floor_cycles,
    )
    proxy_model = _train_staged_mcd(
        prepared, config, objective=ProxyVarianceNLL(schedule)
    )
    proxy_result = score(
        "staged+mcd+proxy", "Staged MC-Dropout proxy-variance model", proxy_model
    )
    return SeedComparison(
        seed=config.seed,
        prepared=prepared,
        models=(joint_mcd_result, staged_mcd_result, proxy_result),
    )


def compare_models(
    config: CmapssConfig,
    *,
    seeds: int,
    prepare: Callable[[CmapssConfig], PreparedCmapssData] = prepare_cmapss_windows,
    on_run: RunReporter = _ignore_run,
) -> list[SeedComparison]:
    """Prepare the data, then train and score the comparison's models, once per seed.

    Seed index `i` runs under `config.seed + i`: that seed draws the
    train/validation split, so the standardization is re-fitted on each
    seed's train split, and it initializes every model and its dropout
    masks. Training is silent. Each `SeedComparison` keeps its prepared data
    and trained models, so a caller can plot, say, the first seed's models
    without retraining.

    The models are the joint MC-Dropout (`"joint+mcd"`), the staged
    MC-Dropout (`"staged+mcd"`) and the staged MC-Dropout proxy-variance
    (`"staged+mcd+proxy"`) model, in that order. The last one's mean stage
    trains on `ProxyVarianceNLL`, with a schedule fitted to the training
    units' lifetimes.

    Args:
        config: Example configuration; its `seed` is the first seed.
        seeds: Number of seeds to prepare the data and train the models from.
        prepare: Prepares the FD001 windows from a seed's configuration;
            `prepare_cmapss_windows` by default.
        on_run: Called with the model's name, the seed and the test metrics
            as soon as each run is scored. Does nothing by default.

    Returns:
        One `SeedComparison` per seed, in seed order.
    """
    results = []
    for index in range(seeds):
        seed_config = dataclasses.replace(config, seed=config.seed + index)
        results.append(
            _train_and_score_seed(prepare(seed_config), seed_config, on_run=on_run)
        )
    return results


_TABLE_ROWS = (
    ("RMSE", "rmse"),
    ("95% coverage", "coverage"),
    ("closed-form CRPS", "closed_form_crps"),
    ("NLL", "nll"),
)


def _mean_std(values: Sequence[float]) -> str:
    """Format values as `mean ± std`: the sample std, or 0 for a single value.

    Args:
        values: One metric's values, one per seed; at least one.

    Returns:
        The mean and the sample standard deviation (`ddof=1`), both to four
        decimals, joined by `±`.
    """
    array = np.asarray(values, dtype=float)
    std = float(array.std(ddof=1)) if array.size > 1 else 0.0
    return f"{float(array.mean()):.4f} ± {std:.4f}"


def format_comparison_table(results: Sequence[SeedComparison]) -> str:
    """Tabulate every model's test metrics as `mean ± std` over seeds.

    Args:
        results: The per-seed comparisons from `compare_models`; at least
            one.

    Returns:
        A table with one row per metric (RMSE, 95% coverage, closed-form
        CRPS, NLL) and one column per model result, headed by its name, in
        the comparisons' model order.
    """
    names = [model.name for model in results[0].models]
    rows = [
        (
            label,
            [
                _mean_std([result.models[index].metrics[metric] for result in results])
                for index in range(len(names))
            ],
        )
        for label, metric in _TABLE_ROWS
    ]
    label_width = max(len(label) for label, _ in rows)
    cell_width = max(len(cell) for _, cells in rows for cell in cells)

    def line(label: str, cells: Sequence[str]) -> str:
        return "  ".join(
            [f"{label:<{label_width}}", *(f"{cell:>{cell_width}}" for cell in cells)]
        )

    return "\n".join([line("", names), *(line(label, cells) for label, cells in rows)])


def print_run(model: str, seed: int, metrics: dict[str, float]) -> None:
    """Print one finished (seed, model) run's test metrics on one line.

    Args:
        model: The model's name, e.g. `"joint+mcd"`.
        seed: The seed the model was initialized from.
        metrics: The model's test metrics by name.
    """
    values = ", ".join(f"{name}={value:.4f}" for name, value in metrics.items())
    print(f"[{model} seed={seed}] {values}")


def main() -> None:
    """Compare the CMAPSS MC-Dropout models over seeds, print and plot the results.

    Compares `joint+mcd`, `staged+mcd` and `staged+mcd+proxy`, in that order, and draws
    every model as one row of the figure.

    Pass `--seeds` to set the number of seeds (default 5) and `--plot-path` to save the
    first seed's RUL-curves figure to that path instead of displaying it.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="Number of seeds to train the models from (default: 5).",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=None,
        help=(
            "Save the first seed's train-split RUL-curves figure, one row per "
            "model, to this path instead of displaying it."
        ),
    )
    args = parser.parse_args()
    if args.seeds <= 0:
        parser.error("--seeds must be positive.")

    config = CmapssConfig()
    results = compare_models(
        config, seeds=args.seeds, prepare=prepare_cmapss_windows, on_run=print_run
    )
    print(f"\nFD001 test metrics over {args.seeds} seeds (mean ± std):")
    print(format_comparison_table(results))

    first = results[0]
    train_trajectories = first.prepared.train_trajectories
    plot_validation_rul_curves(
        train_trajectories,
        first.prepared.standardization.feature_columns,
        [LabelledModel(result.label, result.model) for result in first.models],
        units=select_lifetime_spanning_units(train_trajectories),
        window_length=config.window_length,
        save_path=args.plot_path,
    )


if __name__ == "__main__":
    main()
