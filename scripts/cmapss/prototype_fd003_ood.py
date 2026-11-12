"""PROTOTYPE (throwaway): out-of-distribution coverage on FD003 of FD001-trained models.

Question: how well calibrated are `joint+mcd` and `staged+mcd+proxy`, trained
on FD001 exactly as in `compare.py`, on CMAPSS FD003 (same single operating
condition, an extra fan-degradation fault mode)?

* Scores both models on the FD003 test split (one last window per unit,
  `RUL_FD003.txt`) next to their in-distribution FD001 test scores.
* Plots the RUL curve of FD003 train unit 94 (run-to-failure), one row per
  model, for the first seed.

FD003 is standardized with FD001's training statistics, as a deployed model
would see it.

    uv run python scripts/cmapss/prototype_fd003_ood.py --seeds 3 --plot-path fd003_unit94.png
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import compare as compare_module
from compare import (
    COMPARISON_METRICS,
    ModelResult,
    SeedComparison,
    _train_joint_mcd,
    _train_staged_mcd,
    format_comparison_table,
)

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

compare_module._TABLE_ROWS = (
    *compare_module._TABLE_ROWS,
    ("sigma_epi/sigma_tot", "epistemic_ratio"),
)


def epistemic_ratio(model, windows: np.ndarray) -> float:
    """Mean over windows of sigma_epistemic / sigma_total.

    sigma_epi^2 is the variance of the draws' means; sigma_tot^2 adds the draws' mean
    aleatoric variance (law of total variance).
    """
    prediction = model(np.asarray(windows, dtype=np.float32))
    draws = np.asarray(prediction.draws, dtype=float).reshape(
        prediction.draws.shape[0], -1
    )
    aleatoric = np.broadcast_to(
        np.asarray(prediction.aleatoric_variance, dtype=float),
        prediction.draws.shape,
    ).reshape(draws.shape)
    if prediction.draw_mask is not None:
        keep = np.asarray(prediction.draw_mask, dtype=bool).reshape(draws.shape[0], -1)
        keep = keep.all(axis=1)
        draws, aleatoric = draws[keep], aleatoric[keep]
    epistemic_var = draws.var(axis=0)
    total_var = epistemic_var + aleatoric.mean(axis=0)
    return float(np.mean(np.sqrt(epistemic_var / total_var)))


def load_fd003() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Load FD003 train trajectories, test trajectories and test RUL."""
    with zipfile.ZipFile(_ensure_cached_archive()) as archive:
        train = _load_member(archive, "train_FD003.txt")
        test = _load_member(archive, "test_FD003.txt")
        with archive.open("RUL_FD003.txt") as rul_data:
            rul = pd.read_csv(rul_data, sep=r"\s+", header=None)[0].to_numpy(float)
    return train, test, rul


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--plot-path", type=Path, default=None)
    args = parser.parse_args()

    fd003_train_raw, fd003_test_raw, fd003_rul = load_fd003()
    base = CmapssConfig()
    per_split: dict[str, list[SeedComparison]] = {"FD001": [], "FD003": []}
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
            "joint+mcd": ("Joint MC-Dropout", _train_joint_mcd(prepared, config)),
            "staged+mcd+proxy": (
                "Staged MC-Dropout proxy",
                _train_staged_mcd(
                    prepared, config, objective=ProxyVarianceNLL(schedule)
                ),
            ),
        }
        results: dict[str, list[ModelResult]] = {"FD001": [], "FD003": []}
        for name, (label, raw) in models.items():
            model = CyclesPredictive(raw, prepared.target_standardization)
            for split, windows, rul in (
                ("FD001", prepared.test_windows, prepared.test_rul),
                ("FD003", fd003_test_windows, fd003_rul),
            ):
                metrics = evaluate_test_metrics(
                    model, windows, rul, epoch_metrics=COMPARISON_METRICS
                )
                metrics["epistemic_ratio"] = epistemic_ratio(model, windows)
                print(f"[{split} {name} seed={config.seed}] {metrics}", flush=True)
                results[split].append(ModelResult(name, label, model, metrics))
        for split in per_split:
            per_split[split].append(
                SeedComparison(config.seed, prepared, tuple(results[split]))
            )
        if first_models is None:
            first_models = results["FD003"]
            first_fd003_train = apply_standardization(fd003_train_raw, stats)

    for split, comparisons in per_split.items():
        print(f"\n{split} test metrics over {args.seeds} seeds (mean ± std):")
        print(format_comparison_table(comparisons))

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
