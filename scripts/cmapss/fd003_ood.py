"""Out-of-distribution performance on FD003/test of models trained on FD001/train.

Models compared, trained as in `fd001_base.py`:

* `joint+mcd` ("joint"): the joint-trained baseline.
* `staged+mcd+proxy` ("staged"): the stage-based model with the proxy-variance
  mean stage.

Metrics scored on FD003/test, over seeds: RMSE, 95% interval coverage,
closed-form CRPS and the ratio of the 95% interval width on FD003/test to that
on FD001/test.

Output: per-seed metrics printed to the console, a box plot of each metric
over seeds, one box per model, and a figure of the first seed's RUL curve for
FD003/train unit 94, one row per model.

Run it with:

    uv run python scripts/cmapss/fd003_ood.py [--seeds 3] [--jobs 3] [--plot-path fd003.png]

`--plot-path` saves the RUL-curve figure to that path and the metrics figure
next to it as `<stem>_metrics<suffix>` (e.g. `fd003_metrics.png`), instead of
displaying both. `--jobs` trains the seeds in that many processes: the first seed in
this one, the others in workers.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import zipfile
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from fd001_base import (
    COMPARISON_METRICS,
    ModelResult,
    _train_joint_mcd,
    _train_staged_mcd,
)
from flax import nnx
from stagereg.jax import PosteriorPredictive

from ifac_safeprocess_2027.cmapss.data import (
    _ensure_cached_archive,
    _load_member,
)
from ifac_safeprocess_2027.cmapss.pipeline import (
    _SENSOR_NAMES,
    CmapssConfig,
    CyclesPredictive,
    PreparedCmapssData,
    ProxyVarianceNLL,
    evaluate_test_metrics,
    prepare_cmapss_windows,
)
from ifac_safeprocess_2027.cmapss.plots import _draw_unit_rul_curve
from ifac_safeprocess_2027.cmapss.preprocessing import (
    apply_standardization,
    build_last_windows,
    fit_proxy_variance_schedule,
    unit_lifetimes,
)
from ifac_safeprocess_2027.parallel import add_jobs_argument, spawn_pool

PLOT_UNIT = 94

OOD_DROPPED_METRICS = ("nll",)
"""Metrics left out on FD003: the NLL is dominated by a few units."""

PLOTTED_METRICS = (
    ("RMSE (cycles)", "rmse"),
    ("95% coverage", "coverage"),
    ("closed-form CRPS (cycles)", "closed_form_crps"),
    ("95% width FD003/FD001", "width_ratio"),
)
"""The FD003 metrics box-plotted over seeds: (axis title, metric name)."""


class ClippedVariancePredictive(nnx.Module, pytree=False):
    """A predictive whose aleatoric variance is clipped to `[low, high]`."""

    def __init__(self, model: nnx.Module, low: float, high: float) -> None:
        self.model = model
        self.low = low
        self.high = high

    def __call__(self, inputs) -> PosteriorPredictive:
        output = self.model(inputs)
        return PosteriorPredictive(
            draws=output.draws,
            aleatoric_variance=jnp.clip(output.aleatoric_variance, self.low, self.high),
            draw_mask=output.draw_mask,
        )


@nnx.jit
def _aleatoric_variance_extremes(
    model, batch: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Min and max aleatoric variance over every draw and window of a batch."""
    variance = model(batch).aleatoric_variance
    return variance.min(), variance.max()


def aleatoric_variance_range(
    model, windows: np.ndarray, chunk: int = 64
) -> tuple[float, float]:
    """Min and max aleatoric variance over every draw and window given.

    Each chunk is reduced in one compiled pass. The last chunk is padded with copies of
    the last window, so every chunk has the same shape and compiles once; repeated
    windows leave the min and max unchanged.
    """
    windows = np.asarray(windows, dtype=np.float32)
    padding = -len(windows) % chunk
    windows = np.concatenate([windows, np.repeat(windows[-1:], padding, axis=0)])
    lows, highs = zip(
        *(
            _aleatoric_variance_extremes(model, windows[start : start + chunk])
            for start in range(0, len(windows), chunk)
        ),
        strict=True,
    )
    return float(jnp.min(jnp.stack(lows))), float(jnp.max(jnp.stack(highs)))


def mean_interval_width(model, windows: np.ndarray, level: float = 0.95) -> float:
    """Mean width of the central `level` mixture interval over the windows."""
    prediction = model(np.asarray(windows, dtype=np.float32))
    tail = (1.0 - level) / 2.0
    width = prediction.quantile(1.0 - tail) - prediction.quantile(tail)
    return float(np.mean(np.asarray(width, dtype=float)))


def load_fd003() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Load FD003 train trajectories, test trajectories and test RUL."""
    with zipfile.ZipFile(_ensure_cached_archive()) as archive:
        train = _load_member(archive, "train_FD003.txt")
        test = _load_member(archive, "test_FD003.txt")
        with archive.open("RUL_FD003.txt") as rul_data:
            rul = pd.read_csv(rul_data, sep=r"\s+", header=None)[0].to_numpy(float)
    return train, test, rul


def plot_metric_boxes(
    scores: dict[str, dict[str, list[float]]], seeds: int
) -> plt.Figure:
    """One box plot per metric of `PLOTTED_METRICS`, one box per model."""
    figure, grid = plt.subplots(
        1, len(PLOTTED_METRICS), figsize=(3.2 * len(PLOTTED_METRICS), 3.4)
    )
    names = list(scores)
    for axes, (title, metric) in zip(grid, PLOTTED_METRICS, strict=True):
        axes.boxplot([scores[name][metric] for name in names], tick_labels=names)
        axes.set_title(title)
        if metric == "coverage":
            axes.axhline(0.95, color="grey", linestyle="--", linewidth=1)
        if metric == "width_ratio":
            axes.axhline(1.0, color="grey", linestyle="--", linewidth=1)
    figure.suptitle(f"FD003 test metrics over {seeds} seeds — FD001-trained")
    figure.tight_layout()
    return figure


def train_and_score_seed(
    config: CmapssConfig, fd003_test_raw: pd.DataFrame, fd003_rul: np.ndarray
) -> tuple[PreparedCmapssData, list[ModelResult]]:
    """Train both models from `config.seed` on FD001 and score them on FD003/test.

    Prints each model's train aleatoric variance range and FD003 metrics.

    Returns:
        The seed's prepared FD001 data and one clipped `ModelResult` per model,
        `"joint"` then `"staged"`.
    """
    prepared = prepare_cmapss_windows(config)
    fd003_test_windows = build_last_windows(
        apply_standardization(fd003_test_raw, prepared.standardization),
        _SENSOR_NAMES,
        window_length=config.window_length,
    )
    schedule = fit_proxy_variance_schedule(
        unit_lifetimes(prepared.train_trajectories),
        prepared.target_standardization,
        prognosability=config.proxy_variance_prognosability,
        epsilon_cycles=config.proxy_variance_epsilon_cycles,
    )
    models = {
        "joint": ("Joint MC-Dropout", _train_joint_mcd(prepared, config)),
        "staged": (
            "Staged MC-Dropout proxy",
            _train_staged_mcd(prepared, config, objective=ProxyVarianceNLL(schedule)),
        ),
    }
    results: list[ModelResult] = []
    for name, (label, raw) in models.items():
        model = CyclesPredictive(raw, prepared.target_standardization)
        low, high = aleatoric_variance_range(model, prepared.train_windows)
        print(
            f"[seed={config.seed} {name}] train aleatoric variance range "
            f"[{low:.4g}, {high:.4g}] cycles^2",
            flush=True,
        )
        model = ClippedVariancePredictive(model, low, high)
        metrics = evaluate_test_metrics(
            model, fd003_test_windows, fd003_rul, epoch_metrics=COMPARISON_METRICS
        )
        for metric in OOD_DROPPED_METRICS:
            metrics.pop(metric)
        metrics["width_ratio"] = mean_interval_width(
            model, fd003_test_windows
        ) / mean_interval_width(model, prepared.test_windows)
        print(f"[FD003 {name} seed={config.seed}] {metrics}", flush=True)
        results.append(ModelResult(name, f"{label} clipped", model, metrics))
    return prepared, results


def _metrics_by_model(results: list[ModelResult]) -> dict[str, dict[str, float]]:
    """Each model's FD003 metrics, by model name."""
    return {result.name: result.metrics for result in results}


def _seed_metrics_in_worker(
    config: CmapssConfig, fd003_test_raw: pd.DataFrame, fd003_rul: np.ndarray
) -> dict[str, dict[str, float]]:
    """`train_and_score_seed` for a worker process: only the metrics are sent back."""
    _, results = train_and_score_seed(config, fd003_test_raw, fd003_rul)
    return _metrics_by_model(results)


def run_seeds(
    configs: list[CmapssConfig],
    fd003_test_raw: pd.DataFrame,
    fd003_rul: np.ndarray,
    *,
    jobs: int,
) -> tuple[PreparedCmapssData, list[ModelResult], list[dict[str, dict[str, float]]]]:
    """Train and score every seed, in up to `jobs` processes.

    The first seed always trains in this process, so its models can be plotted; with
    `jobs > 1`, the others train meanwhile in `jobs - 1` spawned workers.

    Returns:
        The first seed's prepared data and model results, and every seed's metrics by
        model name, in seed order.
    """
    if jobs == 1 or len(configs) == 1:
        first_prepared, first_results = train_and_score_seed(
            configs[0], fd003_test_raw, fd003_rul
        )
        rest = [
            _seed_metrics_in_worker(config, fd003_test_raw, fd003_rul)
            for config in configs[1:]
        ]
        return first_prepared, first_results, [_metrics_by_model(first_results), *rest]
    with spawn_pool(min(jobs - 1, len(configs) - 1)) as pool:
        futures = [
            pool.submit(_seed_metrics_in_worker, config, fd003_test_raw, fd003_rul)
            for config in configs[1:]
        ]
        first_prepared, first_results = train_and_score_seed(
            configs[0], fd003_test_raw, fd003_rul
        )
        rest = [future.result() for future in futures]
    return first_prepared, first_results, [_metrics_by_model(first_results), *rest]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=None,
        help=(
            "Save the RUL-curve figure to this path and the metrics figure to "
            "<stem>_metrics<suffix> next to it, instead of displaying both."
        ),
    )
    add_jobs_argument(parser)
    args = parser.parse_args()

    fd003_train_raw, fd003_test_raw, fd003_rul = load_fd003()
    base = CmapssConfig()
    configs = [
        dataclasses.replace(base, seed=base.seed + index) for index in range(args.seeds)
    ]
    first_prepared, first_models, seed_metrics = run_seeds(
        configs, fd003_test_raw, fd003_rul, jobs=args.jobs
    )
    scores = {
        name: {
            metric: [metrics[name][metric] for metrics in seed_metrics]
            for _, metric in PLOTTED_METRICS
        }
        for name in ("joint", "staged")
    }
    first_fd003_train = apply_standardization(
        fd003_train_raw, first_prepared.standardization
    )

    metrics_figure = plot_metric_boxes(scores, args.seeds)

    figure, grid = plt.subplots(
        len(first_models), 1, figsize=(9, 3.2 * len(first_models)), sharex=True
    )
    for axes, result in zip(grid, first_models, strict=True):
        lifetime = _draw_unit_rul_curve(
            axes,
            first_fd003_train,
            _SENSOR_NAMES,
            result.model,
            unit_id=PLOT_UNIT,
            role="OOD",
            window_length=base.window_length,
        )
        axes.set_ylabel(f"{result.label}\nRUL (cycles)")
    grid[0].set_title(
        f"FD003 train unit {PLOT_UNIT} ({lifetime:g} cycles) — FD001-trained, "
        f"seed {base.seed}  [PROTOTYPE]"
    )
    grid[-1].set_xlabel("Time cycles")
    grid[0].legend(loc="upper right")
    figure.tight_layout()
    if args.plot_path:
        figure.savefig(args.plot_path, dpi=150)
        metrics_path = args.plot_path.with_name(
            f"{args.plot_path.stem}_metrics{args.plot_path.suffix}"
        )
        metrics_figure.savefig(metrics_path, dpi=150)
        print(f"\nSaved figures to {args.plot_path} and {metrics_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
