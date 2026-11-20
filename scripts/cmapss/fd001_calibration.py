"""Calibration curves on FD001/test of the joint and staged MC-Dropout models.

Models compared, trained as in `fd001_base.py` at each dropout rate in `--rates`
(default 0.3 and 0.9):

* `joint+mcd`: the joint-trained baseline.
* `staged+mcd`: the stage-based model.

For each nominal confidence level in `CALIBRATION_LEVELS` (10% to 90% in steps
of 10%, and 95%), each model's central interval is scored by its empirical
coverage on FD001/test: the fraction of the 100 test units whose ground-truth
RUL lies inside it.

Output: a table of each (model, rate)'s coverage per level, mean ± std over
seeds, printed to the console, and a figure of every calibration curve on one
axes, the mean over seeds with a ± std band, against the diagonal. The colour
marks the model, the line style the dropout rate.

Run it with:

    uv run python scripts/cmapss/fd001_calibration.py --seeds 5 [--rates 0.3 0.9] \
        [--jobs 10] [--plot-path calibration.png]

`--plot-path` saves the figure instead of displaying it. `--jobs` trains the
(rate, seed) runs in that many processes, so up to `len(rates) * seeds` of them help.
Every rate trains from the same seeds.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from fd001_base import (
    SeedComparison,
    _mean_std,
    _train_and_score_seed_in_worker,
)

from ifac_safeprocess_2027.cmapss.calibration import CALIBRATION_LEVELS
from ifac_safeprocess_2027.cmapss.pipeline import (
    CmapssConfig,
    PreparedCmapssData,
    prepare_cmapss_windows,
    with_dropout_rate,
)
from ifac_safeprocess_2027.cmapss.plots import (
    LabelledCalibrationCurves,
    plot_calibration_curves,
)
from ifac_safeprocess_2027.parallel import add_jobs_argument, process_map

RATES = (0.3, 0.9)
"""The default dropout rates: the configured 0.3, and 0.9, where the joint model's
95% coverage comes closest to nominal."""

COLORS = {"joint+mcd": "#2a78d6", "staged+mcd": "#eb6834"}
"""Each model's colour, as in the dropout sensitivity figure."""

LINESTYLES = {0.3: "-", 0.9: "--"}
"""The line style of each default dropout rate."""

OTHER_LINESTYLE = "-."
"""The line style of any dropout rate outside `LINESTYLES`."""


@dataclass(frozen=True)
class RateCalibration:
    """One (model, dropout rate)'s calibration curves over seeds.

    Attributes:
        name: The table's column header, e.g. `"joint+mcd p=0.3"`.
        labelled: The curves over seeds, labelled and styled for the figure.
    """

    name: str
    labelled: LabelledCalibrationCurves


def collect_calibration_curves(
    results_by_rate: Mapping[float, Sequence[SeedComparison]],
) -> list[RateCalibration]:
    """Stack each (model, dropout rate)'s calibration curve over seeds.

    Args:
        results_by_rate: The per-seed comparisons from `compare_models` at each
            dropout rate, at least one seed per rate.

    Returns:
        One `RateCalibration` per model per rate, grouped by rate in the
        mapping's order and by model in the comparisons' order. Each holds
        curves with shape `(n_seeds, len(CALIBRATION_LEVELS))`, coloured by
        model and styled by the rate's value: `LINESTYLES`, else
        `OTHER_LINESTYLE`.
    """
    calibrations = []
    for rate, results in results_by_rate.items():
        for model_index, model in enumerate(results[0].models):
            calibrations.append(
                RateCalibration(
                    name=f"{model.name} p={rate:g}",
                    labelled=LabelledCalibrationCurves(
                        label=f"{model.label.removesuffix(' model')}, p={rate:g}",
                        curves=np.stack(
                            [
                                np.asarray(result.models[model_index].calibration_curve)
                                for result in results
                            ]
                        ),
                        color=COLORS.get(model.name),
                        linestyle=LINESTYLES.get(rate, OTHER_LINESTYLE),
                    ),
                )
            )
    return calibrations


def format_calibration_table(calibrations: Sequence[RateCalibration]) -> str:
    """Tabulate each (model, rate)'s coverage per nominal level as `mean ± std`.

    Args:
        calibrations: The (model, rate)s' calibration curves over seeds, from
            `collect_calibration_curves`, in column order.

    Returns:
        A table with one row per level in `CALIBRATION_LEVELS` and one
        column per (model, rate).
    """
    names = [calibration.name for calibration in calibrations]
    rows = [
        (
            f"{level:.0%}",
            [
                _mean_std(calibration.labelled.curves[:, level_index])
                for calibration in calibrations
            ],
        )
        for level_index, level in enumerate(CALIBRATION_LEVELS)
    ]
    label_width = max(len("Level"), *(len(label) for label, _ in rows))
    cell_width = max(
        *(len(name) for name in names), *(len(c) for _, r in rows for c in r)
    )

    def line(label: str, cells: Sequence[str]) -> str:
        return "  ".join(
            [f"{label:<{label_width}}", *(f"{cell:>{cell_width}}" for cell in cells)]
        )

    return "\n".join(
        [line("Level", names), *(line(label, cells) for label, cells in rows)]
    )


def config_at_rate(config: CmapssConfig, rate: float) -> CmapssConfig:
    """Return `config` with both MC-Dropout networks at dropout rate `rate`.

    Without dropout every MC-Dropout draw is the same, so a rate of 0 takes a
    single draw.

    Args:
        config: The configuration to copy.
        rate: The dropout rate, in `[0, 1)`.

    Returns:
        A copy of `config` at `rate`.
    """
    return dataclasses.replace(
        with_dropout_rate(config, rate),
        num_draws=1 if rate == 0.0 else config.num_draws,
    )


def print_rate_run(
    rate: float, model: str, seed: int, metrics: dict[str, float]
) -> None:
    """Print one finished (rate, seed, model) run's test metrics on one line.

    Args:
        rate: The dropout rate the model was trained at.
        model: The model's name, e.g. `"joint+mcd"`.
        seed: The seed the model was initialized from.
        metrics: The model's test metrics by name.
    """
    values = ", ".join(f"{name}={value:.4f}" for name, value in metrics.items())
    print(f"[{model} p={rate:g} seed={seed}] {values}", flush=True)


type _RateTask = tuple[
    float, CmapssConfig, Callable[[CmapssConfig], PreparedCmapssData]
]
"""One run: the dropout rate, the seed's configuration and the data preparation."""


def _train_and_score_rate_task(task: _RateTask) -> SeedComparison:
    """Train and score one (rate, seed) run's models, for `process_map`.

    Args:
        task: The dropout rate, the seed's configuration and the data preparation.

    Returns:
        The run's `SeedComparison`, without its trained models.
    """
    rate, seed_config, prepare = task
    return _train_and_score_seed_in_worker(
        config_at_rate(seed_config, rate),
        prepare,
        functools.partial(print_rate_run, rate),
    )


def train_over_rates(
    config: CmapssConfig,
    *,
    seeds: int,
    rates: Sequence[float],
    prepare: Callable[[CmapssConfig], PreparedCmapssData] = prepare_cmapss_windows,
    jobs: int = 1,
) -> dict[float, list[SeedComparison]]:
    """Train and score both MC-Dropout models at every (dropout rate, seed).

    Seed index `i` runs under `config.seed + i` at every rate, as in
    `compare_models`. The runs are independent and train in up to `jobs`
    processes, so up to `len(rates) * seeds` of them help; each prints its
    models' test metrics as they are scored. No run keeps its trained models.

    Args:
        config: Example configuration; its `seed` is the first seed.
        seeds: Number of seeds to train from at each rate.
        rates: Dropout rates, each in `[0, 1)`; a repeated rate trains once.
        prepare: Prepares the FD001 windows from a run's configuration;
            `prepare_cmapss_windows` by default. Must be picklable if `jobs > 1`.
        jobs: Number of processes; `1` trains every run in this one, in order.

    Returns:
        One `SeedComparison` per seed, in seed order, by rate, in the order the
        rates first appear.
    """
    rates = list(dict.fromkeys(rates))
    seed_configs = [
        dataclasses.replace(config, seed=config.seed + index) for index in range(seeds)
    ]
    tasks = [
        (rate, seed_config, prepare) for rate in rates for seed_config in seed_configs
    ]
    results = process_map(_train_and_score_rate_task, tasks, jobs=jobs)
    return {
        rate: results[index * seeds : (index + 1) * seeds]
        for index, rate in enumerate(rates)
    }


def main() -> None:
    """Compare the FD001 MC-Dropout models' calibration over seeds and rates.

    Pass `--seeds` to set the number of seeds (default 5), `--rates` the dropout rates
    (default 0.3 and 0.9), `--jobs` to train the (rate, seed) runs in that many
    processes (default 1) and `--plot-path` to save the calibration figure to that path
    instead of displaying it.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="Number of seeds to train the models from (default: 5).",
    )
    parser.add_argument(
        "--rates",
        type=float,
        nargs="+",
        default=list(RATES),
        help="Dropout rates, each in [0, 1), to train the models at (default: 0.3 0.9).",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=None,
        help="Save the calibration figure to this path instead of displaying it.",
    )
    add_jobs_argument(parser)
    args = parser.parse_args()
    if args.seeds <= 0:
        parser.error("--seeds must be positive.")
    if args.jobs <= 0:
        parser.error("--jobs must be positive.")
    if not all(0.0 <= rate < 1.0 for rate in args.rates):
        parser.error("--rates must each lie in [0, 1).")

    results_by_rate = train_over_rates(
        CmapssConfig(), seeds=args.seeds, rates=args.rates, jobs=args.jobs
    )
    calibrations = collect_calibration_curves(results_by_rate)
    print(
        f"\nFD001 test coverage per nominal level over {args.seeds} seeds (mean ± std):"
    )
    print(format_calibration_table(calibrations))
    plot_calibration_curves(
        [calibration.labelled for calibration in calibrations],
        CALIBRATION_LEVELS,
        title="CMAPSS FD001 test calibration",
        save_path=args.plot_path,
    )


if __name__ == "__main__":
    main()
