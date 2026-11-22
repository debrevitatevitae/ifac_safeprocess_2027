"""Sensitivity of the FD001 test metrics to the MC-Dropout rate.

Models compared, trained as in `fd001_base.py` for every dropout rate:

* `joint+mcd` ("joint"): the joint-trained baseline.
* `staged+mcd` ("staged"): the stage-based model.

Metrics scored on FD001/test, mean ± std over seeds: RMSE, 95% interval
coverage, NLL and the epistemic fraction, the epistemic share of the predictive
variance averaged over the test units. At p=0 each model takes a single draw, so
its epistemic fraction is zero by construction and is left out of the table and
the figure.

Output: per-run metrics (closed-form CRPS included) and one mean ± std table
per metric printed to the console, plus a 1x4 figure sized for the full text
width of a two-column paper, with one subplot per metric, the dropout rate on
the x-axis and one line per model.

Run it with:

    uv run python scripts/cmapss/fd001_dropout_sensitivity.py \
        [--seeds 5] [--rates 0 0.1 0.5] [--jobs 8] [--plot-path dropout.png]

`--jobs` trains the (seed, rate) runs in that many processes.

Smoke test: `--seeds 1 --rates 0 0.5 --epochs 1 --num-draws 10`.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from fd001_base import (
    _train_joint_mcd,
    _train_staged_mcd,
    epistemic_fraction,
    score_on_test_split,
)
from matplotlib.ticker import MultipleLocator

from ifac_safeprocess_2027.cmapss.pipeline import (
    CmapssConfig,
    CyclesPredictive,
    prepare_cmapss_windows,
    with_dropout_rate,
)
from ifac_safeprocess_2027.parallel import add_jobs_argument, process_map

RATES = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9)
MODELS = ("joint", "staged")
METRICS = (
    ("rmse", "RMSE [cycles]"),
    ("coverage", "95% coverage"),
    ("nll", "NLL"),
    ("epistemic_fraction", "Epi. fraction"),
)
EPISTEMIC_FRACTION = "epistemic_fraction"
"""The metric undefined at p=0, where each model takes a single draw."""
COLORS = {"joint": "#2a78d6", "staged": "#eb6834"}


def run_one(seed_config: CmapssConfig, p: float) -> dict[str, dict[str, float]]:
    """Train and score both models at dropout rate `p`: metrics by model name."""
    prepared = prepare_cmapss_windows(seed_config)
    p_config = dataclasses.replace(
        with_dropout_rate(seed_config, p),
        num_draws=1 if p == 0.0 else seed_config.num_draws,
    )
    trained = {
        "joint": _train_joint_mcd(prepared, p_config),
        "staged": _train_staged_mcd(prepared, p_config),
    }
    scores = {}
    for name, model in trained.items():
        cycles = CyclesPredictive(model, prepared.target_standardization)
        scores[name] = metrics = {
            **score_on_test_split(cycles, prepared),
            EPISTEMIC_FRACTION: epistemic_fraction(cycles, prepared),
        }
        values = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"[{name} p={p} seed={seed_config.seed}] {values}", flush=True)
    return scores


def _run_task(task: tuple[CmapssConfig, float]) -> dict[str, dict[str, float]]:
    """`run_one` on a `(seed_config, p)` pair, for `process_map`."""
    return run_one(*task)


def run(config: CmapssConfig, seeds: int, rates: list[float], jobs: int = 1) -> dict:
    """Return results[model][p] -> list of metric dicts, one per seed.

    The (seed, rate) runs are independent, and train in up to `jobs` processes.
    """
    tasks = [
        (dataclasses.replace(config, seed=config.seed + index), p)
        for index in range(seeds)
        for p in rates
    ]
    results = {m: {p: [] for p in rates} for m in MODELS}
    for (_, p), scores in zip(
        tasks, process_map(_run_task, tasks, jobs=jobs), strict=True
    ):
        for name, metrics in scores.items():
            results[name][p].append(metrics)
    return results


def _scored_rates(key: str, rates: list[float]) -> list[float]:
    """The rates `key` is reported at: all but p=0 for the epistemic fraction."""
    return [p for p in rates if p != 0.0] if key == EPISTEMIC_FRACTION else rates


def summarize(results: dict, rates: list[float]) -> None:
    """Print mean ± std over seeds per (metric, model, p)."""
    for key, label in METRICS:
        print(f"\n{label}")
        print("p      " + "  ".join(f"{m:>18}" for m in MODELS))
        for p in rates:
            if p not in _scored_rates(key, rates):
                print(f"{p:<5}  " + "  ".join(f"{'—':>18}" for _ in MODELS))
                continue
            cells = []
            for m in MODELS:
                v = np.array([r[key] for r in results[m][p]])
                std = v.std(ddof=1) if v.size > 1 else 0.0
                cells.append(f"{v.mean():9.4f} ± {std:6.4f}")
            print(f"{p:<5}  " + "  ".join(cells))


def plot(results: dict, rates: list[float], save_path: Path | None) -> None:
    """1x4 line subplots: metric vs p, one line per model, ± std band."""
    fig, axes = plt.subplots(
        1, 4, figsize=(7.0, 1.9), sharex=True, constrained_layout=True
    )
    for ax, (key, label) in zip(axes, METRICS, strict=True):
        scored = _scored_rates(key, rates)
        for m in MODELS if scored else ():
            v = np.array([[r[key] for r in results[m][p]] for p in scored])
            mean = v.mean(axis=1)
            std = v.std(axis=1, ddof=1) if v.shape[1] > 1 else np.zeros_like(mean)
            ax.plot(scored, mean, marker="o", ms=5, lw=2, color=COLORS[m], label=m)
            ax.fill_between(scored, mean - std, mean + std, color=COLORS[m], alpha=0.15)
        if key == "coverage":
            ax.axhline(0.95, color="0.5", lw=1, ls="--")
        if key == EPISTEMIC_FRACTION:
            ax.set_ylim(0.0, 0.5)
        ax.set_title(label)
        ax.set_xlabel("dropout rate p")
        ax.xaxis.set_major_locator(MultipleLocator(0.3))
        ax.xaxis.set_minor_locator(MultipleLocator(0.1))
        ax.grid(alpha=0.3)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(frameon=False)
    if save_path is None:
        plt.show()
    else:
        fig.savefig(save_path, dpi=150)
        print(f"\nSaved figure to {save_path}")


def main() -> None:
    """Parse args, run the sweep, print the summary and plot it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--rates", type=float, nargs="+", default=list(RATES))
    parser.add_argument("--epochs", type=int, default=None, help="Override all epochs.")
    parser.add_argument("--num-draws", type=int, default=None)
    parser.add_argument("--plot-path", type=Path, default=None)
    add_jobs_argument(parser)
    args = parser.parse_args()

    config = CmapssConfig()
    if args.epochs is not None:
        config = dataclasses.replace(
            config,
            mean_epochs=args.epochs,
            variance_epochs=args.epochs,
            joint_epochs=args.epochs,
        )
    if args.num_draws is not None:
        config = dataclasses.replace(config, num_draws=args.num_draws)

    results = run(config, args.seeds, args.rates, jobs=args.jobs)
    summarize(results, args.rates)
    plot(results, args.rates, args.plot_path)


if __name__ == "__main__":
    main()
