"""Staged and joint RUL models for windowed CMAPSS data.

PROTOTYPE (branch `prototype/lstm-backbone`): the mean and joint networks can
use either a 1D-CNN (`Cnn1DBackbone`) or an LSTM (`LstmBackbone`) backbone,
and the Gamma network a single-convolution backbone (`GammaCnnBackbone`).
Each model is a backbone-agnostic output head over an already-built
backbone; `pipeline.build_backbone` builds one from its network configuration.
"""

from __future__ import annotations

from typing import Protocol

import jax
import jax.numpy as jnp
from flax import nnx
from stagereg.jax import Gamma, GammaHead, Gaussian, GaussianHead


class Backbone(Protocol):
    """A feature extractor mapping windows to `features`-wide vectors."""

    features: int

    def __call__(
        self, inputs: jax.Array, *, dropout_key: jax.Array | None = None
    ) -> jax.Array:
        """Map windows ``(batch, window_length, n_sensors)`` to ``(batch,
        features)``."""
        ...


def _batch_shared_dropouts(dropout_rate: float, count: int, rngs: nnx.Rngs) -> nnx.List:
    """Build `count` dropout layers sharing one mask across the batch axis.

    At rate 0 they draw nothing from `rngs`, so the backbone's state is that
    of a backbone without dropout.
    """
    dropout_rngs = rngs if dropout_rate > 0.0 else None
    return nnx.List(
        [
            nnx.Dropout(dropout_rate, broadcast_dims=(0,), rngs=dropout_rngs)
            for _ in range(count)
        ]
    )


def _dropout_key(dropout_key: jax.Array | None, index: int) -> jax.Array | None:
    """Return the key of dropout layer `index`, or `None` for its own stream."""
    return None if dropout_key is None else jax.random.fold_in(dropout_key, index)


class Cnn1DBackbone(nnx.Module):
    """Two 1D-convolutional layers over the time axis, sensors as channels.

    Dropout follows every hidden layer, i.e. each convolutional layer, with
    one mask per call shared across the batch axis (`broadcast_dims=(0,)`),
    as [`MCDropout`][stagereg.jax.MCDropout] requires: a draw is then the
    same function at every window. Dropout adds no parameters, so the
    parameter tree does not depend on the rate. At rate 0 the dropout layers
    are the identity and hold no random state, so the backbone, its state
    and its outputs are those of a backbone without dropout.
    """

    def __init__(
        self,
        n_sensors: int,
        *,
        hidden_channels: int = 16,
        kernel_size: int = 5,
        dropout_rate: float = 0.0,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize the stacked 1D-convolutional feature extractor.

        Args:
            n_sensors: Number of sensor channels in the windowed input.
            hidden_channels: Number of output channels for both
                convolutional layers.
            kernel_size: Convolution kernel width along the time axis.
            dropout_rate: Dropout rate after every convolutional layer; 0
                disables dropout.
            rngs: NNX random-number generators used to initialize parameters
                and, at a positive rate, the dropout masks' streams.
        """
        self.features = hidden_channels
        self.conv1 = nnx.Conv(
            in_features=n_sensors,
            out_features=hidden_channels,
            kernel_size=kernel_size,
            padding="SAME",
            rngs=rngs,
        )
        self.conv2 = nnx.Conv(
            in_features=hidden_channels,
            out_features=hidden_channels,
            kernel_size=kernel_size,
            padding="SAME",
            rngs=rngs,
        )
        # Built after every parameter, so that a rate of 0 draws nothing from
        # `rngs` and leaves the state unchanged.
        self.dropouts = _batch_shared_dropouts(dropout_rate, 2, rngs)

    def __call__(
        self, inputs: jax.Array, *, dropout_key: jax.Array | None = None
    ) -> jax.Array:
        """Extract a fixed-size feature vector per windowed example.

        Args:
            inputs: Windowed sensor readings, shape ``(batch, window_length,
                n_sensors)``.
            dropout_key: Key the dropout masks are drawn from; `None` draws
                them from the backbone's own streams.

        Returns:
            Time-averaged convolutional features, shape ``(batch,
            hidden_channels)``.
        """
        hidden = inputs
        for index, (conv, dropout) in enumerate(
            zip((self.conv1, self.conv2), self.dropouts, strict=True)
        ):
            hidden = nnx.relu(conv(hidden))
            hidden = dropout(hidden, rngs=_dropout_key(dropout_key, index))
        return jnp.mean(hidden, axis=1)


class LstmBackbone(nnx.Module):
    """PROTOTYPE: LSTM(seq) -> dropout -> LSTM(last) -> dropout -> FC -> FC.

    Dropout masks are shared across the batch axis (`broadcast_dims=(0,)`), as
    `MCDropout` requires, and, as in `Cnn1DBackbone`, a rate of 0 leaves the
    state unchanged.
    """

    def __init__(
        self,
        n_sensors: int,
        *,
        hidden_units_layer_1: int = 32,
        hidden_units_layer_2: int = 64,
        hidden_units_fc_1: int = 8,
        hidden_units_fc_2: int = 8,
        dropout_rate: float = 0.0,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize the two LSTM layers and the two FC layers.

        Args:
            n_sensors: Number of sensor channels in the windowed input.
            hidden_units_layer_1: Width of the first LSTM, which returns its
                whole sequence.
            hidden_units_layer_2: Width of the second LSTM, whose last step
                feeds the FC layers.
            hidden_units_fc_1: Width of the first FC layer.
            hidden_units_fc_2: Width of the second FC layer, the backbone's
                output.
            dropout_rate: Dropout rate after each LSTM; 0 disables dropout.
            rngs: NNX random-number generators used to initialize parameters
                and, at a positive rate, the dropout masks' streams.
        """
        self.features = hidden_units_fc_2
        self.lstm1 = nnx.OptimizedLSTMCell(n_sensors, hidden_units_layer_1, rngs=rngs)
        self.lstm2 = nnx.OptimizedLSTMCell(
            hidden_units_layer_1, hidden_units_layer_2, rngs=rngs
        )
        self.fc1 = nnx.Linear(hidden_units_layer_2, hidden_units_fc_1, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_units_fc_1, hidden_units_fc_2, rngs=rngs)
        self.dropouts = _batch_shared_dropouts(dropout_rate, 2, rngs)

    def __call__(
        self, inputs: jax.Array, *, dropout_key: jax.Array | None = None
    ) -> jax.Array:
        """Extract a fixed-size feature vector per windowed example.

        Args:
            inputs: Windowed sensor readings, shape ``(batch, window_length,
                n_sensors)``.
            dropout_key: Key the dropout masks are drawn from; `None` draws
                them from the backbone's own streams.

        Returns:
            The second FC layer's features, shape ``(batch,
            hidden_units_fc_2)``.
        """

        def run(cell: nnx.OptimizedLSTMCell, sequence: jax.Array) -> jax.Array:
            # Plain `lax.scan` from a zero (c, h): `nnx.RNN` (nnx.scan, and a
            # carry init drawing from stored rngs) fails under the pipeline's
            # MC-Dropout `jax.vmap`.
            zero = jnp.zeros((sequence.shape[0], cell.hidden_features))
            _, outputs = jax.lax.scan(
                lambda carry, x: cell(carry, x),
                (zero, zero),
                jnp.swapaxes(sequence, 0, 1),
            )
            return jnp.swapaxes(outputs, 0, 1)  # (batch, window_length, width)

        hidden = run(self.lstm1, inputs)
        hidden = self.dropouts[0](hidden, rngs=_dropout_key(dropout_key, 0))
        hidden = run(self.lstm2, hidden)[:, -1, :]  # last step only
        hidden = self.dropouts[1](hidden, rngs=_dropout_key(dropout_key, 1))
        hidden = nnx.relu(self.fc1(hidden))
        return nnx.relu(self.fc2(hidden))


class GammaCnnBackbone(nnx.Module):
    """PROTOTYPE: Conv1D -> ReLU -> global average pooling, without dropout."""

    def __init__(
        self,
        n_sensors: int,
        *,
        hidden_channels: int = 4,
        kernel_size: int = 5,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize the single convolutional layer.

        Args:
            n_sensors: Number of sensor channels in the windowed input.
            hidden_channels: Number of output channels of the convolution.
            kernel_size: Convolution kernel width along the time axis.
            rngs: NNX random-number generators used to initialize parameters.
        """
        self.features = hidden_channels
        self.conv = nnx.Conv(
            in_features=n_sensors,
            out_features=hidden_channels,
            kernel_size=kernel_size,
            padding="SAME",
            rngs=rngs,
        )

    def __call__(
        self, inputs: jax.Array, *, dropout_key: jax.Array | None = None
    ) -> jax.Array:
        """Extract time-averaged convolutional features, shape ``(batch, features)``.

        Args:
            inputs: Windowed sensor readings, shape ``(batch, window_length,
                n_sensors)``.
            dropout_key: Ignored: the backbone has no dropout.

        Returns:
            Time-averaged convolutional features, shape ``(batch,
            hidden_channels)``.
        """
        del dropout_key
        return jnp.mean(nnx.relu(self.conv(inputs)), axis=1)


class MeanModel(nnx.Module):
    """Deterministic point-RUL regressor over windowed CMAPSS sensor data."""

    def __init__(self, backbone: Backbone, *, rngs: nnx.Rngs) -> None:
        """Wrap a backbone with the mean output head.

        Args:
            backbone: The feature extractor, e.g. a `Cnn1DBackbone` or an
                `LstmBackbone`; its dropout rate is the model's.
            rngs: NNX random-number generators used to initialize the head.
        """
        self.backbone = backbone
        self.output = nnx.Linear(backbone.features, 1, rngs=rngs)

    def __call__(
        self, inputs: jax.Array, *, dropout_key: jax.Array | None = None
    ) -> jax.Array:
        """Predict a scalar RUL for each windowed example in the batch.

        Args:
            inputs: Windowed sensor readings, shape ``(batch, window_length,
                n_sensors)``.
            dropout_key: Key the backbone's dropout masks are drawn from;
                `None` draws them from the backbone's own streams.

        Returns:
            Predicted RUL, shape ``(batch, 1)``.
        """
        return self.output(self.backbone(inputs, dropout_key=dropout_key))


class GammaModel(nnx.Module):
    """Gamma residual regressor over windowed CMAPSS sensor data.

    Independently initialized from ``MeanModel`` and trained on
    Stage-1's squared residuals through
    [`GammaVarianceStage`][stagereg.jax.GammaVarianceStage].
    """

    def __init__(self, backbone: Backbone, *, rngs: nnx.Rngs) -> None:
        """Wrap a backbone with the Gamma output head.

        Args:
            backbone: The feature extractor, e.g. a `GammaCnnBackbone`.
            rngs: NNX random-number generators used to initialize the head.
        """
        self.backbone = backbone
        self.head = GammaHead(backbone.features, 1, rngs=rngs)

    def __call__(self, inputs: jax.Array) -> Gamma:
        """Predict a Gamma distribution over the squared RUL residual.

        Args:
            inputs: Windowed sensor readings, shape ``(batch, window_length,
                n_sensors)``.

        Returns:
            A [`Gamma`][stagereg.jax.Gamma] with positive concentration and
            rate, each shaped ``(batch, 1)``.
        """
        return self.head(self.backbone(inputs))


class GaussianModel(nnx.Module):
    """Joint model: one backbone predicting a Gaussian RUL distribution.

    The network of the joint MC-Dropout model: a single backbone followed by
    a [`GaussianHead`][stagereg.jax.GaussianHead] that predicts mean and
    scale together.
    """

    def __init__(self, backbone: Backbone, *, rngs: nnx.Rngs) -> None:
        """Wrap a backbone with the Gaussian output head.

        Args:
            backbone: The feature extractor, e.g. a `Cnn1DBackbone` or an
                `LstmBackbone`; its dropout rate is the model's.
            rngs: NNX random-number generators used to initialize the head.
        """
        self.backbone = backbone
        self.head = GaussianHead(backbone.features, 1, rngs=rngs)

    def __call__(
        self, inputs: jax.Array, *, dropout_key: jax.Array | None = None
    ) -> Gaussian:
        """Predict a Gaussian RUL distribution for each windowed example.

        Args:
            inputs: Windowed sensor readings, shape ``(batch, window_length,
                n_sensors)``.
            dropout_key: Key the backbone's dropout masks are drawn from;
                `None` draws them from the backbone's own streams.

        Returns:
            A [`Gaussian`][stagereg.jax.Gaussian] with ``loc`` and a strictly
            positive ``scale``, each shaped ``(batch, 1)``.
        """
        return self.head(self.backbone(inputs, dropout_key=dropout_key))
