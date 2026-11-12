"""PROTOTYPE (throwaway): out-of-distribution coverage on FD003 of FD001-trained models.

Question: how well calibrated are `joint+mcd` and `staged+mcd+proxy`, trained
on FD001 exactly as in `compare.py`, on CMAPSS FD003 (same single operating
condition, an extra fan-degradation fault mode)?

* Scores both models on the FD003 test split (one last window per unit,
  `RUL_FD003.txt`) and box-plots each metric over seeds, one box per model.
  The FD001 test windows are only used for the 95% width ratio's denominator.
* Plots the RUL curve of FD003 train unit 94 (run-to-failure), one row per
  model, for the first seed.

FD003 is standardized with FD001's training statistics, as a deployed model
would see it.

    uv run python scripts/cmapss/prototype_fd003_ood.py --seeds 3 \
        --plot-path fd003_unit94.png --metrics-plot-path fd003_metrics.png
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import zipfile
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from compare import (
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


def aleatoric_variance_range(
    model, windows: np.ndarray, chunk: int = 256
) -> tuple[float, float]:
    """Min and max aleatoric variance over every draw and window given."""
    low, high = np.inf, -np.inf
    for start in range(0, len(windows), chunk):
        batch = np.asarray(windows[start : start + chunk], dtype=np.float32)
        variance = np.asarray(model(batch).aleatoric_variance, dtype=float)
        low, high = min(low, variance.min()), max(high, variance.max())
    return float(low), float(high)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--plot-path", type=Path, default=None)
    parser.add_argument("--metrics-plot-path", type=Path, default=None)
    args = parser.parse_args()

    fd003_train_raw, fd003_test_raw, fd003_rul = load_fd003()
    base = CmapssConfig()
    scores: dict[str, dict[str, list[float]]] = {
        "joint": {metric: [] for _, metric in PLOTTED_METRICS},
        "staged": {metric: [] for _, metric in PLOTTED_METRICS},
    }
    first_models = None
    first_fd003_train = None

    for index in range(args.seeds):
        config = dataclasses.replace(base, seed=base.seed + index)
        prepared = prepare_cmapss_windows(config)
        stats = prepared.standardization
        fd003_test_windows = build_last_windows(
            apply_standardization(fd003_test_raw, stats),
            _SENSOR_NAMES,
            window_length=config.window_length,
        )
        schedule = fit_proxy_variance_schedule(
            unit_lifetimes(prepared.train_trajectories),
            prepared.target_standardization,
            floor_cycles=config.proxy_variance_floor_cycles,
        )
        models = {
            "joint": ("Joint MC-Dropout", _train_joint_mcd(prepared, config)),
            "staged": (
                "Staged MC-Dropout proxy",
                _train_staged_mcd(
                    prepared, config, objective=ProxyVarianceNLL(schedule)
                ),
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
            for _, metric in PLOTTED_METRICS:
                scores[name][metric].append(metrics[metric])
            results.append(ModelResult(name, f"{label} clipped", model, metrics))
        if first_models is None:
            first_models = results
            first_fd003_train = apply_standardization(fd003_train_raw, stats)

    metrics_figure = plot_metric_boxes(scores, args.seeds)
    if args.metrics_plot_path:
        metrics_figure.savefig(args.metrics_plot_path, dpi=150)

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
    else:
        plt.show()


if __name__ == "__main__":
    main()
