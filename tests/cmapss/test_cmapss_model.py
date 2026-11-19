"""Tests for the CMAPSS networks and their backbones."""

from __future__ import annotations

import importlib
from collections.abc import Callable

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

jax = pytest.importorskip("jax")
nnx = pytest.importorskip("flax.nnx")

_MODEL = importlib.import_module("ifac_safeprocess_2027.cmapss.model")

_DROPOUT_BACKBONES = ("Cnn1DBackbone", "LstmBackbone")
"""The backbones that accept a dropout rate, for the mean and joint models."""

_DROPOUT_MODELS = ("MeanModel", "GaussianModel")
"""The CMAPSS models whose backbone may use dropout."""


def _build(
    model_name: str, backbone_name: str, n_sensors: int, seed: int = 0, **kwargs: object
) -> nnx.Module:
    """Build `model_name` on a fresh `backbone_name`, both from one `Rngs`."""
    rngs = nnx.Rngs(seed)
    backbone = getattr(_MODEL, backbone_name)(n_sensors, rngs=rngs, **kwargs)
    return getattr(_MODEL, model_name)(backbone, rngs=rngs)


@pytest.mark.parametrize("backbone_name", _DROPOUT_BACKBONES)
@pytest.mark.parametrize(("batch_size", "n_sensors"), [(4, 9), (1, 3)])
def test_mean_model_outputs_one_scalar_per_example(
    backbone_name: str, batch_size: int, n_sensors: int
) -> None:
    model = _build("MeanModel", backbone_name, n_sensors)
    inputs = jax.random.normal(jax.random.key(1), (batch_size, 30, n_sensors))

    assert model(inputs).shape == (batch_size, 1)


@pytest.mark.parametrize(
    ("backbone_name", "kwargs", "features"),
    [
        ("Cnn1DBackbone", {"hidden_channels": 8, "kernel_size": 3}, 8),
        (
            "LstmBackbone",
            {
                "hidden_units_layer_1": 3,
                "hidden_units_layer_2": 5,
                "hidden_units_fc_1": 4,
                "hidden_units_fc_2": 6,
            },
            6,
        ),
        ("GammaCnnBackbone", {"hidden_channels": 2, "kernel_size": 3}, 2),
    ],
)
def test_backbones_output_their_configured_features(
    backbone_name: str, kwargs: dict[str, int], features: int
) -> None:
    backbone = getattr(_MODEL, backbone_name)(5, rngs=nnx.Rngs(0), **kwargs)
    inputs = jax.random.normal(jax.random.key(3), (6, 30, 5))

    assert backbone.features == features
    assert backbone(inputs).shape == (6, features)


def test_gamma_model_outputs_one_gamma_distribution_per_example() -> None:
    batch_size, window_length, n_sensors = 4, 30, 9
    model = _build("GammaModel", "GammaCnnBackbone", n_sensors)
    inputs = jax.random.normal(
        jax.random.key(1), (batch_size, window_length, n_sensors)
    )

    prediction = model(inputs)

    assert prediction.concentration.shape == (batch_size, 1)
    assert prediction.rate.shape == (batch_size, 1)
    assert bool(jax.numpy.all(prediction.concentration > 0))
    assert bool(jax.numpy.all(prediction.rate > 0))


def _outputs(model: nnx.Module, inputs: jax.Array, **kwargs: object) -> jax.Array:
    """Return a model's outputs as one array, whatever its output type."""
    prediction = model(inputs, **kwargs)
    if isinstance(prediction, jax.Array):
        return prediction
    return jax.numpy.concatenate([prediction.loc, prediction.scale], axis=-1)


_dropout_cases = pytest.mark.parametrize(
    ("model_name", "backbone_name"),
    [(model, backbone) for model in _DROPOUT_MODELS for backbone in _DROPOUT_BACKBONES],
)


@_dropout_cases
def test_dropout_rate_zero_leaves_the_model_unchanged(
    model_name: str, backbone_name: str
) -> None:
    default = _build(model_name, backbone_name, 4)
    zero_rate = _build(model_name, backbone_name, 4, dropout_rate=0.0)
    inputs = jax.random.normal(jax.random.key(5), (3, 30, 4))

    default_state = nnx.state(default)
    zero_rate_state = nnx.state(zero_rate)

    assert jax.tree.structure(default_state) == jax.tree.structure(zero_rate_state)
    for default_leaf, zero_rate_leaf in zip(
        jax.tree.leaves(default_state), jax.tree.leaves(zero_rate_state), strict=True
    ):
        assert bool(jax.numpy.array_equal(default_leaf, zero_rate_leaf))
    assert bool(
        jax.numpy.array_equal(
            _outputs(default, inputs),
            _outputs(zero_rate, inputs, dropout_key=jax.random.key(6)),
        )
    )


@_dropout_cases
def test_active_dropout_shares_one_mask_across_the_batch(
    model_name: str, backbone_name: str
) -> None:
    model = _build(model_name, backbone_name, 4, dropout_rate=0.5)
    row = jax.random.normal(jax.random.key(7), (1, 30, 4))
    inputs = jax.numpy.repeat(row, 8, axis=0)

    outputs = _outputs(model, inputs, dropout_key=jax.random.key(8))
    model.eval()
    without_dropout = _outputs(model, inputs)

    assert bool(jax.numpy.all(outputs == outputs[:1]))
    assert not bool(jax.numpy.allclose(outputs, without_dropout))


@_dropout_cases
def test_active_dropout_differs_between_keys(
    model_name: str, backbone_name: str
) -> None:
    model = _build(model_name, backbone_name, 4, dropout_rate=0.5)
    inputs = jax.random.normal(jax.random.key(9), (3, 30, 4))

    first = _outputs(model, inputs, dropout_key=jax.random.key(10))
    same = _outputs(model, inputs, dropout_key=jax.random.key(10))
    other = _outputs(model, inputs, dropout_key=jax.random.key(11))

    assert bool(jax.numpy.array_equal(first, same))
    assert not bool(jax.numpy.allclose(first, other))


@_dropout_cases
def test_active_dropout_draws_fresh_masks_from_its_own_streams(
    model_name: str, backbone_name: str
) -> None:
    model = _build(model_name, backbone_name, 4, dropout_rate=0.5)
    inputs = jax.random.normal(jax.random.key(12), (3, 30, 4))

    assert not bool(
        jax.numpy.allclose(_outputs(model, inputs), _outputs(model, inputs))
    )


@settings(max_examples=10, deadline=None)
@given(dropout_rate=st.floats(min_value=0.0, max_value=0.99))
@_dropout_cases
def test_dropout_rate_leaves_the_parameter_tree_unchanged(
    model_name: str, backbone_name: str, dropout_rate: float
) -> None:
    without_dropout = _build(model_name, backbone_name, 4)
    with_dropout = _build(model_name, backbone_name, 4, dropout_rate=dropout_rate)

    without_parameters = nnx.state(without_dropout, nnx.Param)
    with_parameters = nnx.state(with_dropout, nnx.Param)

    assert jax.tree.structure(with_parameters) == jax.tree.structure(without_parameters)
    assert [leaf.shape for leaf in jax.tree.leaves(with_parameters)] == [
        leaf.shape for leaf in jax.tree.leaves(without_parameters)
    ]


def test_gamma_backbone_takes_no_dropout_rate() -> None:
    build: Callable[..., object] = _MODEL.GammaCnnBackbone
    with pytest.raises(TypeError, match="dropout_rate"):
        build(4, dropout_rate=0.1, rngs=nnx.Rngs(0))
