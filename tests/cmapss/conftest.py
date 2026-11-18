"""Shared fixtures for the CMAPSS tests.

The library modules are imported from `ifac_safeprocess_2027.cmapss`; the
`fd001_base.py` script, which lives under `scripts/cmapss/` rather than in the
installed package, is loaded from its file path. Each is provided once per
session behind one fixture per module. Session scope also keeps Hypothesis happy: a
function-scoped fixture in a `@given` test trips its
`function_scoped_fixture` health check.

The tests share a tiny configuration (`tiny_cmapss_config`), synthetic
windows and prepared data (`synthetic_windows`, `synthetic_prepared_data`),
and a recording stand-in for the entry points' plotting (`plot_recorder`).

Each fixture guards its own optional dependencies, so a test module is
collected even when the extras it needs are absent and only the tests that
actually touch that module skip.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

_SCRIPTS_DIR = Path(__file__).parents[2] / "scripts" / "cmapss"


def _load_script_module(module_name: str, file_name: str) -> ModuleType:
    """Load one of the CMAPSS scripts from its file path.

    Args:
        module_name: Name to register the loaded module under, so that
            anything importing it a second time gets the same object.
        file_name: The script's file name within `scripts/cmapss/`.

    Returns:
        The executed module.

    Raises:
        RuntimeError: If the module could not be loaded from its path.
    """
    spec = importlib.util.spec_from_file_location(module_name, _SCRIPTS_DIR / file_name)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load the script module {file_name!r}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def cmapss_preprocessing() -> ModuleType:
    """The windowing and standardization module."""
    for dependency in ("pandas", "sklearn"):
        pytest.importorskip(dependency)
    return importlib.import_module("ifac_safeprocess_2027.cmapss.preprocessing")


@pytest.fixture(scope="session")
def cmapss_plots() -> ModuleType:
    """The RUL-curve plotting module."""
    for dependency in ("matplotlib", "pandas", "sklearn"):
        pytest.importorskip(dependency)
    return importlib.import_module("ifac_safeprocess_2027.cmapss.plots")


@pytest.fixture(scope="session")
def cmapss_pipeline() -> ModuleType:
    """The data preparation, training and evaluation pipeline module."""
    for dependency in (
        "jax",
        "jax.numpy",
        "flax.nnx",
        "optax",
        "pandas",
        "matplotlib",
        "sklearn",
    ):
        pytest.importorskip(dependency)
    return importlib.import_module("ifac_safeprocess_2027.cmapss.pipeline")


@pytest.fixture(scope="session")
def cmapss_fd001(cmapss_pipeline: ModuleType) -> ModuleType:
    """The MC-Dropout model comparison script."""
    return _load_script_module("cmapss_fd001", "fd001_base.py")


@pytest.fixture
def offline_fd001(cmapss_pipeline: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve synthetic FD001 splits in place of the cached NASA archive."""
    from cmapss_trajectories import build_random_trajectories

    rng = np.random.default_rng(7)
    sensor_names = cmapss_pipeline._SENSOR_NAMES
    train_data = build_random_trajectories(
        rng, unit_ids=range(1, 11), n_cycles=60, feature_columns=sensor_names
    )
    test_data = build_random_trajectories(
        rng, unit_ids=range(1, 6), n_cycles=45, feature_columns=sensor_names
    )
    test_rul = rng.uniform(low=1.0, high=100.0, size=5)
    monkeypatch.setattr(cmapss_pipeline, "load_fd001_data", lambda: train_data)
    monkeypatch.setattr(cmapss_pipeline, "load_fd001_test_data", lambda: test_data)
    monkeypatch.setattr(cmapss_pipeline, "load_fd001_test_rul", lambda: test_rul)


@pytest.fixture(scope="session")
def tiny_cmapss_config(cmapss_pipeline: ModuleType) -> Any:
    """A CMAPSS configuration small enough to train every model in a second or two."""
    return cmapss_pipeline.CmapssConfig(
        window_length=10,
        batch_size=16,
        joint_net_config=cmapss_pipeline.MeanCNNConfig(kernel_size=3, hidden_units=2),
        mean_net_config=cmapss_pipeline.MeanLSTMConfig(2, 2, 2, 2),
        gamma_net_config=cmapss_pipeline.GammaCNNConfig(kernel_size=3, hidden_units=2),
        mean_epochs=2,
        variance_epochs=2,
        joint_epochs=2,
        seed=3,
    )


def _synthetic_windows(
    rng: np.random.Generator,
    n_windows: int,
    *,
    window_length: int = 10,
    n_sensors: int = 3,
    max_rul: float = 100.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw standard-normal windows and uniform RUL targets in `[1, max_rul)`.

    Args:
        rng: The generator to draw from.
        n_windows: Number of windows to draw.
        window_length: Number of cycles per window.
        n_sensors: Number of sensor channels per window.
        max_rul: Upper bound of the uniform RUL targets.

    Returns:
        The windows, shape `(n_windows, window_length, n_sensors)`, and
        their targets, shape `(n_windows,)`.
    """
    windows = rng.normal(size=(n_windows, window_length, n_sensors))
    targets = rng.uniform(low=1.0, high=max_rul, size=n_windows)
    return windows, targets


@pytest.fixture(scope="session")
def synthetic_windows() -> Callable[..., tuple[np.ndarray, np.ndarray]]:
    """Return a factory of random CMAPSS-shaped windows and RUL targets.

    It is called as `synthetic_windows(rng, n_windows, *, window_length=10, n_sensors=3,
    max_rul=100.0)`.
    """
    return _synthetic_windows


@pytest.fixture(scope="session")
def synthetic_prepared_data(cmapss_pipeline: ModuleType) -> Callable[..., Any]:
    """Return a factory of small synthetic `PreparedCmapssData`.

    It is called as `synthetic_prepared_data(*, seed=0, max_rul=100.0)`. Only the
    windowed arrays (32 train, 8 validation and 6 test windows of length 10 over 3
    sensors) are meant to be trained on. The training trajectories hold four units with
    random lifetimes in `[1, max_rul]` and zero sensors, only so that a proxy-variance
    schedule can be fitted to their lifetimes; the validation trajectories are empty. As
    in the real data, the train and validation targets are standardized with statistics
    fitted on the train targets, and the test RUL stays in cycles.
    """
    import pandas as pd

    def make(*, seed: int = 0, max_rul: float = 100.0) -> Any:
        rng = np.random.default_rng(seed)
        train_windows, train_targets = _synthetic_windows(rng, 32, max_rul=max_rul)
        validation_windows, validation_targets = _synthetic_windows(
            rng, 8, max_rul=max_rul
        )
        test_windows, test_rul = _synthetic_windows(rng, 6, max_rul=max_rul)
        target_stats = cmapss_pipeline.fit_target_standardization(train_targets)
        n_sensors = train_windows.shape[-1]
        sensors = [f"sensor_{i}" for i in range(n_sensors)]
        columns = ["unit_id", "time_cycles", *sensors]
        lifetimes = rng.integers(1, int(max_rul), endpoint=True, size=4)
        train_trajectories = pd.DataFrame(
            [
                {"unit_id": unit, "time_cycles": cycle, **dict.fromkeys(sensors, 0.0)}
                for unit, lifetime in enumerate(lifetimes, start=1)
                for cycle in range(1, int(lifetime) + 1)
            ],
            columns=columns,
        )
        return cmapss_pipeline.PreparedCmapssData(
            train_windows=train_windows,
            train_targets=target_stats.standardize(train_targets),
            validation_windows=validation_windows,
            validation_targets=target_stats.standardize(validation_targets),
            test_windows=test_windows,
            test_rul=test_rul,
            train_trajectories=train_trajectories,
            validation_trajectories=pd.DataFrame(columns=columns),
            standardization=cmapss_pipeline.SensorStandardization(
                feature_columns=sensors,
                mean=np.zeros(n_sensors),
                scale=np.ones(n_sensors),
            ),
            target_standardization=target_stats,
        )

    return make


@dataclass(frozen=True)
class RecordedPlot:
    """One call of a CMAPSS entry point's `plot_validation_rul_curves`.

    Attributes:
        trajectories: The validation trajectories handed over.
        feature_columns: The feature columns handed over.
        models: The labelled models, one per figure row.
        units: The three units to draw.
        window_length: The window length handed over.
        save_path: Where the figure would be saved, or `None` to display it.
        printed: Everything printed to stdout before the call.
    """

    trajectories: Any
    feature_columns: list[str]
    models: list[Any]
    units: Any
    window_length: int
    save_path: Path | None
    printed: str


class PlotRecorder:
    """A stand-in for `plot_validation_rul_curves` that records its calls.

    Attributes:
        calls: The calls recorded so far, in order.
    """

    def __init__(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Start with no recorded calls.

        Args:
            capsys: The capture whose output so far each call records.
        """
        self._capsys = capsys
        self.calls: list[RecordedPlot] = []

    def __call__(
        self,
        trajectories: Any,
        feature_columns: list[str],
        models: Sequence[Any],
        *,
        units: Any,
        window_length: int,
        save_path: Path | None,
    ) -> None:
        """Record one plotting call and whatever was printed before it.

        Args:
            trajectories: The validation trajectories to draw from.
            feature_columns: The models' feature columns.
            models: The labelled models, one per figure row.
            units: The three units to draw.
            window_length: Number of cycles per window.
            save_path: Where to save the figure, or `None` to display it.
        """
        self.calls.append(
            RecordedPlot(
                trajectories=trajectories,
                feature_columns=feature_columns,
                models=list(models),
                units=units,
                window_length=window_length,
                save_path=save_path,
                printed=self._capsys.readouterr().out,
            )
        )


@pytest.fixture
def plot_recorder(capsys: pytest.CaptureFixture[str]) -> PlotRecorder:
    """Return a fresh `PlotRecorder` to monkeypatch over an entry point's plotting."""
    return PlotRecorder(capsys)


@pytest.fixture(scope="session")
def spanning_units(
    cmapss_preprocessing: ModuleType,
) -> Callable[[int, int, int], Any]:
    """Return a factory naming three units in figure column order."""

    def make(shortest: int, median: int, longest: int) -> Any:
        return cmapss_preprocessing.LifetimeSpanningUnits(
            shortest=shortest, median=median, longest=longest
        )

    return make
