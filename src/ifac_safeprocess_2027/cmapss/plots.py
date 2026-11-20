"""RUL curves and calibration curves for the CMAPSS example.

Turns one or more trained models, each predicting a posterior predictive,
and the standardized trajectories they were trained under into the
example's headline figure: one row per model, and in each row three
held-out validation units side by side — the shortest-, median-, and
longest-lifetime one — each showing its true linear RUL across the whole
trajectory, the model's predicted mean, and two 95% bands around that
mean: a total band and a narrower epistemic band. Spanning the lifetime
range that way shows whether the model's accuracy and its stated
uncertainty behave consistently across units that fail early and units
that survive several times as long.

The predicted curve begins at each unit's first full window, because no
prediction exists before `window_length` cycles of sensor history have
accumulated. The true-RUL line spans the whole trajectory regardless, so
the length of that warm-up stays visible rather than hidden.

This module knows nothing about training, about the archive, or about
which units are worth looking at: it takes already-standardized
trajectories, already-trained labelled models, and the three units to
draw.

It also draws the calibration curves of several models on one axes, each
already computed over seeds by `calibration.calibration_curve`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .preprocessing import LifetimeSpanningUnits, build_unit_rul_curve

_TRUE_RUL_LABEL = "True linear RUL"
_PREDICTED_MEAN_LABEL = "Predicted mean"
_TOTAL_INTERVAL_LABEL = "95% total interval"
_EPISTEMIC_INTERVAL_LABEL = "95% epistemic interval"
# Central 95% interval of a posterior predictive and of its draws' means.
_LOWER_PROBABILITY = 0.025
_UPPER_PROBABILITY = 0.975
# The role each column plays in the figure, in column order.
_ROLE_LABELS: tuple[str, ...] = (
    "shortest lifetime",
    "median lifetime",
    "longest lifetime",
)


class PosteriorPredictiveDistribution(Protocol):
    """A posterior predictive over a batch of windows: a mixture over draws.

    The library's `PosteriorPredictive` is one.
    """

    @property
    def draws(self) -> Any:
        """Each draw's mean, shape `(n_draws, n_windows, 1)`."""
        ...

    @property
    def draw_mask(self) -> Any | None:
        """Which draw slots the mixture includes, shape `(n_draws,)`.

        `None` includes every slot.
        """
        ...

    def mean(self) -> Any:
        """Return the mixture mean, shape `(n_windows, 1)`."""
        ...

    def quantile(self, probability: float) -> Any:
        """Return the mixture quantile at `probability`, shape `(n_windows, 1)`."""
        ...


class PredictiveModel(Protocol):
    """A model mapping windowed sensor readings to a predictive distribution."""

    def __call__(self, inputs: Any) -> PosteriorPredictiveDistribution:
        """Predict one posterior predictive over the windowed examples.

        Args:
            inputs: Windowed sensor readings, shape `(n_windows,
                window_length, n_features)`.

        Returns:
            The posterior predictive over those windows, each of whose draws
            is one function applied to the whole batch.
        """
        ...


@dataclass(frozen=True)
class LabelledModel:
    """A trained predictive model with the label naming its row in the figure.

    Attributes:
        label: The name of the model, shown as its row's label, e.g.
            `"Staged MC-Dropout model"`.
        model: The trained model whose posterior-predictive output the row
            draws.
    """

    label: str
    model: PredictiveModel


def _per_window_array(values: Any) -> np.ndarray:
    """Return per-window values as a flat float array.

    Args:
        values: One value per window, in any array-like shape.

    Returns:
        The values as a 1-D float array, one entry per window.
    """
    return np.asarray(values, dtype=float).reshape(-1)


def _draw_posterior_bands(
    axes: Axes, window_cycles: Any, prediction: PosteriorPredictiveDistribution
) -> None:
    """Draw a posterior predictive's mean, total band and epistemic band.

    The total band is the mixture's central 95% interval, from its
    quantiles; the epistemic band is the 2.5% and 97.5% quantiles of the
    draws' means at each window, ignoring the aleatoric variance. Draw
    slots a `draw_mask` excludes are left out of the epistemic band, as the
    mixture leaves them out of its quantiles.
    """
    draws = np.asarray(prediction.draws, dtype=float)
    if prediction.draw_mask is not None:
        draws = draws[np.asarray(prediction.draw_mask, dtype=bool)]
    draws = draws.reshape(draws.shape[0], -1)
    epistemic_lower, epistemic_upper = np.quantile(
        draws, (_LOWER_PROBABILITY, _UPPER_PROBABILITY), axis=0
    )
    axes.plot(
        window_cycles,
        _per_window_array(prediction.mean()),
        label=_PREDICTED_MEAN_LABEL,
        color="tab:blue",
    )
    axes.fill_between(
        window_cycles,
        _per_window_array(prediction.quantile(_LOWER_PROBABILITY)),
        _per_window_array(prediction.quantile(_UPPER_PROBABILITY)),
        alpha=0.2,
        color="tab:blue",
        label=_TOTAL_INTERVAL_LABEL,
    )
    axes.fill_between(
        window_cycles,
        epistemic_lower,
        epistemic_upper,
        alpha=0.4,
        color="tab:orange",
        label=_EPISTEMIC_INTERVAL_LABEL,
    )


def _draw_unit_rul_curve(
    axes: Axes,
    trajectories: pd.DataFrame,
    feature_columns: Sequence[str],
    model: PredictiveModel,
    *,
    unit_id: object,
    role: str,
    window_length: int,
) -> float:
    """Draw one unit's RUL curve and predictive band onto one column.

    Args:
        axes: The column to draw onto. Its limits are left to autoscale, so
            each column is scaled to its own unit and the band stays
            readable in the short-lived unit.
        trajectories: The standardized trajectories holding `unit_id`.
        feature_columns: Names of the columns to use as window channels, in
            the order the model was trained with.
        model: The trained predictive model.
        unit_id: Identifier of the unit to draw.
        role: The role this unit plays in the figure, named in the title.
        window_length: Number of cycles per window, as trained with.

    Returns:
        The unit's lifetime, in cycles.
    """
    curve = build_unit_rul_curve(
        trajectories,
        feature_columns,
        unit_id=unit_id,
        window_length=window_length,
    )
    # One evaluation over all of the unit's windows, so that each draw of a
    # posterior predictive is the same function along the whole curve.
    prediction = model(np.asarray(curve.windows, dtype=np.float32))

    axes.plot(
        curve.trajectory_cycles,
        curve.trajectory_linear_rul,
        label=_TRUE_RUL_LABEL,
        color="black",
    )
    _draw_posterior_bands(axes, curve.window_cycles, prediction)
    return curve.lifetime


def plot_validation_rul_curves(
    trajectories: pd.DataFrame,
    feature_columns: Sequence[str],
    models: Sequence[LabelledModel],
    *,
    units: LifetimeSpanningUnits,
    window_length: int = 30,
    save_path: Path | None = None,
) -> Figure:
    """Plot RUL curves for three units spanning a lifetime range, per model.

    The figure is an N×3 grid: one row per labelled model, in the order
    given, and one column per unit. Each column shares its x-axis across
    rows, so every model's curve for a unit sits on the same time axis,
    and each row's leftmost axis is labelled with its model's name. A
    single model yields a single row.

    The units to draw are chosen by the caller, so this module decides how
    a RUL curve looks and never which subjects it looks at. Passing the
    lifetime-spanning units of a held-out validation subset gives the
    example's headline figure: those units are never trained on, so the
    curves are evidence of generalization rather than of memorized fit,
    and a unit's lifetime is only observable at all for the
    run-to-failure train-split trajectories the validation subset is
    carved out of. Each row draws the mixture mean, a total band (the mixture's central 95%
    interval, from its quantiles) and a narrower epistemic band (the 2.5%
    and 97.5% quantiles of the draws' means at each window). Each curve
    comes from one evaluation over the unit's windows, so every draw is
    the same function along it. No band is clipped at zero, so that a band
    dipping below zero stays visible as evidence of the distributional
    assumption breaking down near end of life. One figure legend lists the
    elements drawn.

    Args:
        trajectories: The standardized trajectories holding `units`,
            already scaled with the statistics fitted on the training
            subset only.
        feature_columns: Names of the columns to use as window channels, in
            the order the model was trained with.
        models: The trained models to draw, one row each, top to bottom.
        units: The three units to draw, one per column. Each must have at
            least `window_length` cycles in `trajectories`.
        window_length: Number of cycles per window, as trained with.
        save_path: Where to save the figure. When omitted, the figure is
            displayed instead.

    Returns:
        The figure holding the grid, rows in `models` order and columns in
        shortest / median / longest lifetime order.

    Raises:
        ValueError: If `models` is empty, or if any of `units` is absent
            from `trajectories` or has fewer than `window_length` cycles.
    """
    if not models:
        raise ValueError("At least one labelled model is required.")
    # x-axes shared per column only, so a unit is compared on the same time
    # axis across models while each column keeps its own range; y-axes are
    # left unshared, so each panel's band stays readable on its own scale.
    figure, grid = plt.subplots(
        len(models),
        len(_ROLE_LABELS),
        figsize=(15.0, 4.5 * len(models)),
        sharex="col",
        squeeze=False,
    )
    plotted_units = (units.shortest, units.median, units.longest)
    last_row = len(models) - 1
    for row, labelled in enumerate(models):
        for column, (unit_id, role) in enumerate(
            zip(plotted_units, _ROLE_LABELS, strict=True)
        ):
            axes = grid[row, column]
            lifetime = _draw_unit_rul_curve(
                axes,
                trajectories,
                feature_columns,
                labelled.model,
                unit_id=unit_id,
                role=role,
                window_length=window_length,
            )
            # Unit titles head the top row and time labels sit under the
            # bottom row; the rows in between share both with them.
            if row == 0:
                axes.set_title(f"Unit {unit_id} — {role} ({lifetime:g} cycles)")
            if row == last_row:
                axes.set_xlabel("Time cycles")
            y_label = "RUL (cycles)"
            axes.set_ylabel(f"{labelled.label}\n{y_label}" if column == 0 else y_label)

    # One legend for the whole figure, listing each element drawn once, in
    # first-drawn order: every panel draws the same elements, so a legend
    # per panel would say nothing new.
    legend_entries: dict[str, Any] = {}
    for axes in grid.flat:
        for handle, label in zip(*axes.get_legend_handles_labels(), strict=True):
            legend_entries.setdefault(label, handle)
    figure.legend(
        list(legend_entries.values()),
        list(legend_entries),
        loc="lower center",
        ncol=len(legend_entries),
    )
    figure.suptitle("CMAPSS FD001 held-out validation RUL curves")
    figure.tight_layout(rect=(0.0, 0.08 / len(models), 1.0, 1.0))

    if save_path is not None:
        figure.savefig(Path(save_path))
    else:
        plt.show()
    return figure


_IDEAL_CALIBRATION_LABEL = "Ideal calibration"


@dataclass(frozen=True)
class LabelledCalibrationCurves:
    """A model's calibration curves over seeds, with the label naming it.

    Attributes:
        label: The name of the model, shown in the legend, e.g. `"Staged
            MC-Dropout model"`.
        curves: The model's calibration curve on each seed: its empirical
            coverage at each nominal level, one row per seed, shape
            `(n_seeds, n_levels)`.
        color: The colour of the model's curve and band, any matplotlib
            colour; `None`, the default, takes the next colour of the cycle.
        linestyle: The matplotlib line style of the model's curve; solid by
            default.
    """

    label: str
    curves: np.ndarray
    color: str | None = None
    linestyle: str = "-"


def plot_calibration_curves(
    models: Sequence[LabelledCalibrationCurves],
    levels: Sequence[float],
    *,
    title: str,
    save_path: Path | None = None,
) -> Figure:
    """Plot each model's calibration curve, mean ± std over seeds, on one axes.

    Each model draws its mean empirical coverage against the nominal
    confidence level, with a band of ± the sample standard deviation over
    seeds (none for a single seed), in the model's colour and line style. The
    dotted diagonal marks ideal calibration: a curve below it is overconfident,
    one above it underconfident.

    Args:
        models: The models' calibration curves over seeds, drawn in the
            order given.
        levels: The nominal confidence levels the curves' columns belong to.
        title: The axes' title, naming the data the curves were scored on.
        save_path: Where to save the figure. When omitted, the figure is
            displayed instead.

    Returns:
        The figure holding the single calibration axes.

    Raises:
        ValueError: If `models` is empty.
    """
    if not models:
        raise ValueError("At least one labelled model is required.")
    figure, axes = plt.subplots(figsize=(5.5, 5.0))
    axes.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle=":",
        color="0.5",
        label=_IDEAL_CALIBRATION_LABEL,
    )
    x = np.asarray(levels, dtype=float)
    for labelled in models:
        curves = np.atleast_2d(np.asarray(labelled.curves, dtype=float))
        mean = curves.mean(axis=0)
        std = curves.std(axis=0, ddof=1) if curves.shape[0] > 1 else np.zeros_like(mean)
        (line,) = axes.plot(
            x,
            mean,
            marker="o",
            color=labelled.color,
            linestyle=labelled.linestyle,
            label=labelled.label,
        )
        axes.fill_between(
            x, mean - std, mean + std, color=line.get_color(), alpha=0.12, linewidth=0
        )
    axes.set_xlim(0.0, 1.0)
    axes.set_ylim(0.0, 1.0)
    axes.set_aspect("equal")
    axes.set_xlabel("Nominal confidence level")
    axes.set_ylabel("Empirical coverage")
    axes.set_title(title)
    axes.legend(loc="lower right")
    figure.tight_layout()

    if save_path is not None:
        figure.savefig(Path(save_path))
    else:
        plt.show()
    return figure
