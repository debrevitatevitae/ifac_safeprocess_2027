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
        [--jobs 5] [--plot-path calibration.png]

`--plot-path` saves the figure instead of displaying it. `--jobs` trains each rate's
seeds in that many processes: the first seed in this one, the others in workers.
Every rate trains from the same seeds.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from fd001_base import SeedComparison, _mean_std, compare_models, print_run

from ifac_safeprocess_2027.cmapss.calibration import CALIBRATION_LEVELS
from ifac_safeprocess_2027.cmapss.pipeline import (
    CmapssConfig,
    prepare_cmapss_windows,
    with_dropout_rate,
)
from ifac_safeprocess_2027.cmapss.plots import (
    LabelledCalibrationCurves,
    plot_calibration_curves,
)
from ifac_safeprocess_2027.parallel import add_jobs_argument

RATES = (0.3, 0.9)
"""The default dropout rates: the configured 0.3, and 0.9, where the joint model's
95% coverage comes closest to nominal."""

COLORS = {"joint+mcd": "#2a78d6", "staged+mcd": "#eb6834"}
"""Each model's colour, as in the dropout sensitivity figure."""

LINESTYLES = ("-", "--", "-.")
"""The line style of each dropout rate, in the order given, cycling past the last."""


@dataclass(frozen=True)
class CalibrationColumn:
    """One (model, dropout rate)'s calibration curves over seeds.

    Attributes:
        name: The table's column header, e.g. `"joint+mcd p=0.3"`.
        curves: The curves over seeds, labelled and styled for the figure.
    """

    name: str
    curves: LabelledCalibrationCurves


def collect_calibration_curves(
    results_by_rate: Mapping[float, Sequence[SeedComparison]],
) -> list[CalibrationColumn]:
    """Stack each (model, dropout rate)'s calibration curve over seeds.

    Args:
        results_by_rate: The per-seed comparisons from `compare_models` at each
            dropout rate, at least one seed per rate.

    Returns:
        One `CalibrationColumn` per model per rate, grouped by rate in the
        mapping's order and by model in the comparisons' order. Each holds
        curves with shape `(n_seeds, len(CALIBRATION_LEVELS))`, coloured by
        model and styled by rate.
    """
    columns = []
    for rate_index, (rate, results) in enumerate(results_by_rate.items()):
        for index, model in enumerate(results[0].models):
            columns.append(
                CalibrationColumn(
                    name=f"{model.name} p={rate:g}",
                    curves=LabelledCalibrationCurves(
                        label=f"{model.label.removesuffix(' model')}, p={rate:g}",
                        curves=np.stack(
                            [
                                np.asarray(result.models[index].calibration_curve)
                                for result in results
                            ]
                        ),
                        color=COLORS.get(model.name),
                        linestyle=LINESTYLES[rate_index % len(LINESTYLES)],
                    ),
                )
            )
    return columns


def format_calibration_table(columns: Sequence[CalibrationColumn]) -> str:
    """Tabulate each column's coverage per nominal level as `mean ± std` over seeds.

    Args:
        columns: The (model, rate)s' calibration curves over seeds, from
            `collect_calibration_curves`, in column order.

    Returns:
        A table with one row per level in `CALIBRATION_LEVELS` and one
        column per (model, rate).
    """
    names = [column.name for column in columns]
    rows = [
        (
            f"{level:.0%}",
            [_mean_std(column.curves.curves[:, index]) for column in columns],
        )
        for index, level in enumerate(CALIBRATION_LEVELS)
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


def main() -> None:
    """Compare the FD001 MC-Dropout models' calibration over seeds and rates.

    Pass `--seeds` to set the number of seeds (default 5), `--rates` the dropout rates
    (default 0.3 and 0.9), `--jobs` to train each rate's seeds in that many processes
    (default 1) and `--plot-path` to save the calibration figure to that path instead of
    displaying it.
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

    config = CmapssConfig()
    results_by_rate = {}
    for rate in dict.fromkeys(args.rates):
        print(f"\nDropout rate p={rate:g}:", flush=True)
        results_by_rate[rate] = compare_models(
            config_at_rate(config, rate),
            seeds=args.seeds,
            prepare=prepare_cmapss_windows,
            on_run=print_run,
            jobs=args.jobs,
        )
    columns = collect_calibration_curves(results_by_rate)
    print(
        f"\nFD001 test coverage per nominal level over {args.seeds} seeds (mean ± std):"
    )
    print(format_calibration_table(columns))
    plot_calibration_curves(
        [column.curves for column in columns],
        CALIBRATION_LEVELS,
        title="CMAPSS FD001 test calibration",
        save_path=args.plot_path,
    )


if __name__ == "__main__":
    main()
