"""Calibration curves on FD001/test of the joint and staged MC-Dropout models.

Models compared, trained as in `fd001_base.py`:

* `joint+mcd`: the joint-trained baseline.
* `staged+mcd`: the stage-based model.

For each nominal confidence level in `CALIBRATION_LEVELS` (10% to 90% in steps
of 10%, and 95%), each model's central interval is scored by its
empirical coverage on FD001/test: the fraction of the 100 test units whose
ground-truth RUL lies inside it.

Output: a table of each model's coverage per level, mean ± std over seeds,
printed to the console, and a figure of both models' calibration curves on one
axes, the mean over seeds with a ± std band, against the diagonal.

Run it with:

    uv run python scripts/cmapss/fd001_calibration.py --seeds 5 [--jobs 5] [--plot-path calibration.png]

`--plot-path` saves the figure instead of displaying it. `--jobs` trains the seeds in
that many processes: the first seed in this one, the others in workers.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from fd001_base import SeedComparison, _mean_std, compare_models, print_run

from ifac_safeprocess_2027.cmapss.calibration import CALIBRATION_LEVELS
from ifac_safeprocess_2027.cmapss.pipeline import CmapssConfig, prepare_cmapss_windows
from ifac_safeprocess_2027.cmapss.plots import (
    LabelledCalibrationCurves,
    plot_calibration_curves,
)
from ifac_safeprocess_2027.parallel import add_jobs_argument


def collect_calibration_curves(
    results: Sequence[SeedComparison],
) -> list[LabelledCalibrationCurves]:
    """Stack each model's calibration curve over seeds.

    Args:
        results: The per-seed comparisons from `compare_models`; at least
            one.

    Returns:
        One `LabelledCalibrationCurves` per model, in the comparisons' model order,
        labelled with the model's figure label and holding its curves
        with shape `(n_seeds, len(CALIBRATION_LEVELS))`.
    """
    return [
        LabelledCalibrationCurves(
            label=model.label,
            curves=np.stack(
                [
                    np.asarray(result.models[index].calibration_curve)
                    for result in results
                ]
            ),
        )
        for index, model in enumerate(results[0].models)
    ]


def format_calibration_table(
    names: Sequence[str], curves: Sequence[LabelledCalibrationCurves]
) -> str:
    """Tabulate each model's coverage per nominal level as `mean ± std` over seeds.

    Args:
        names: The models' names, the table's column headers, in the order
            of `curves`.
        curves: Each model's calibration curves over seeds, from
            `collect_calibration_curves`.

    Returns:
        A table with one row per level in `CALIBRATION_LEVELS` and one
        column per model.
    """
    rows = [
        (
            f"{level:.0%}",
            [_mean_std(model.curves[:, column]) for model in curves],
        )
        for column, level in enumerate(CALIBRATION_LEVELS)
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


def main() -> None:
    """Compare the FD001 MC-Dropout models' calibration over seeds.

    Pass `--seeds` to set the number of seeds (default 5), `--jobs` to train them in
    that many processes (default 1) and `--plot-path` to save the calibration figure to
    that path instead of displaying it.
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
        help="Save the calibration figure to this path instead of displaying it.",
    )
    add_jobs_argument(parser)
    args = parser.parse_args()
    if args.seeds <= 0:
        parser.error("--seeds must be positive.")
    if args.jobs <= 0:
        parser.error("--jobs must be positive.")

    results = compare_models(
        CmapssConfig(),
        seeds=args.seeds,
        prepare=prepare_cmapss_windows,
        on_run=print_run,
        jobs=args.jobs,
    )
    curves = collect_calibration_curves(results)
    names = [model.name for model in results[0].models]
    print(
        f"\nFD001 test coverage per nominal level over {args.seeds} seeds (mean ± std):"
    )
    print(format_calibration_table(names, curves))
    plot_calibration_curves(
        curves,
        CALIBRATION_LEVELS,
        title="CMAPSS FD001 test calibration",
        save_path=args.plot_path,
    )


if __name__ == "__main__":
    main()
