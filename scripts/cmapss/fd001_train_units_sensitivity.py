"""Sensitivity of the FD001 test metrics to the number of training units.

Models compared, trained as in `fd001_base.py` on nested random subsets of the
FD001/train units:

* `joint+mcd` ("joint"): the joint-trained baseline.
* `staged+mcd+proxy` ("staged"): the stage-based model with the proxy-variance
  mean stage.

Metrics scored on FD001/test, mean ± std over seeds: RMSE, 95% interval
coverage, closed-form CRPS and NLL.

Output: per-run metrics and one mean ± std table per metric printed to the
console, plus a 1x4 figure with one subplot per metric, the fraction of
training units on the x-axis and one line per model.

Run it with:

    uv run python scripts/cmapss/fd001_train_units_sensitivity.py \
        [--seeds 5] [--fractions 0.1 0.5 1.0] [--plot-path units.png]

Smoke test: `--seeds 1 --fractions 0.1 1.0 --epochs 1 --num-draws 10`.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from fd001_base import _train_joint_mcd, _train_staged_mcd, score_on_test_split

from ifac_safeprocess_2027.cmapss.data import load_fd001_data
from ifac_safeprocess_2027.cmapss.pipeline import (
    CmapssConfig,
    CyclesPredictive,
    ProxyVarianceNLL,
    prepare_cmapss_windows,
)
from ifac_safeprocess_2027.cmapss.preprocessing import (
    fit_proxy_variance_schedule,
    unit_lifetimes,
)

FRACTIONS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
MODELS = ("joint", "staged")
METRICS = (
    ("rmse", "RMSE [cycles]"),
    ("coverage", "95% coverage"),
    ("closed_form_crps", "CRPS [cycles]"),
    ("nll", "NLL"),
)
COLORS = {"joint": "#2a78d6", "staged": "#eb6834"}


def run(config: CmapssConfig, seeds: int, fractions: list[float]) -> dict:
    """Return results[model][f] -> list of metric dicts, one per seed."""
    all_units = np.unique(load_fd001_data()["unit_id"])
    results = {m: {f: [] for f in fractions} for m in MODELS}
    for index in range(seeds):
        seed_config = dataclasses.replace(config, seed=config.seed + index)
        order = np.random.default_rng(seed_config.seed).permutation(all_units)
        for f in fractions:
            n_units = max(2, round(f * all_units.size))
            prepared = prepare_cmapss_windows(
                seed_config, train_units=order[:n_units].tolist()
            )
            schedule = fit_proxy_variance_schedule(
                unit_lifetimes(prepared.train_trajectories),
                prepared.target_standardization,
                floor_cycles=seed_config.proxy_variance_floor_cycles,
            )
            trained = {
                "joint": _train_joint_mcd(prepared, seed_config),
                "staged": _train_staged_mcd(
                    prepared, seed_config, objective=ProxyVarianceNLL(schedule)
                ),
            }
            for name, model in trained.items():
                cycles = CyclesPredictive(model, prepared.target_standardization)
                metrics = score_on_test_split(cycles, prepared)
                results[name][f].append(metrics)
                values = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                print(
                    f"[{name} f={f} units={n_units} seed={seed_config.seed}] {values}",
                    flush=True,
                )
    return results


def summarize(results: dict, fractions: list[float]) -> None:
    """Print mean ± std over seeds per (metric, model, f)."""
    for key, label in METRICS:
        print(f"\n{label}")
        print("f      " + "  ".join(f"{m:>18}" for m in MODELS))
        for f in fractions:
            cells = []
            for m in MODELS:
                v = np.array([r[key] for r in results[m][f]])
                std = v.std(ddof=1) if v.size > 1 else 0.0
                cells.append(f"{v.mean():9.4f} ± {std:6.4f}")
            print(f"{f:<5}  " + "  ".join(cells))


def plot(results: dict, fractions: list[float], save_path: Path | None) -> None:
    """1x4 line subplots: metric vs f, one line per model, ± std band."""
    percent = [100 * f for f in fractions]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), constrained_layout=True)
    for ax, (key, label) in zip(axes, METRICS, strict=True):
        for m in MODELS:
            v = np.array([[r[key] for r in results[m][f]] for f in fractions])
            mean = v.mean(axis=1)
            std = v.std(axis=1, ddof=1) if v.shape[1] > 1 else np.zeros_like(mean)
            ax.plot(percent, mean, marker="o", ms=5, lw=2, color=COLORS[m], label=m)
            ax.fill_between(
                percent, mean - std, mean + std, color=COLORS[m], alpha=0.15
            )
        if key == "coverage":
            ax.axhline(0.95, color="0.5", lw=1, ls="--")
        ax.set_title(label)
        ax.set_xlabel("training units [%]")
        ax.set_xticks(percent)
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
    parser.add_argument("--fractions", type=float, nargs="+", default=list(FRACTIONS))
    parser.add_argument("--epochs", type=int, default=None, help="Override all epochs.")
    parser.add_argument("--num-draws", type=int, default=None)
    parser.add_argument("--plot-path", type=Path, default=None)
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

    results = run(config, args.seeds, args.fractions)
    summarize(results, args.fractions)
    plot(results, args.fractions, args.plot_path)


if __name__ == "__main__":
    main()
