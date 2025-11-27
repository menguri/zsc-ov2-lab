import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from typing import Any, Callable, List, Sequence
import jax.numpy as jnp
from flax import struct


class CNN(nn.Module):
    output_size: int = 64
    activation: Callable[..., Any] = nn.relu

    @nn.compact
    def __call__(self, x, train=False):
        x = nn.Conv(
            features=128,
            kernel_size=(1, 1),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)
        x = nn.Conv(
            features=128,
            kernel_size=(1, 1),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)
        x = nn.Conv(
            features=8,
            kernel_size=(1, 1),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        x = nn.Conv(
            features=16,
            kernel_size=(3, 3),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        x = x.reshape((x.shape[0], -1))
        # x = x.flatten()
        print("CNN shapes", x.shape)

        x = nn.Dense(
            features=self.output_size,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        return x


class CNNSimple(nn.Module):
    output_size: int = 64
    activation: Callable[..., Any] = nn.relu

    @nn.compact
    def __call__(self, x, train=False):
        x = nn.Conv(
            features=32,
            kernel_size=(5, 5),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        x = x.reshape((x.shape[0], -1))
        # x = x.flatten()
        print("CNN shapes", x.shape)

        x = nn.Dense(
            features=self.output_size,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        return x


class CNNNew(nn.Module):
    features: int = 32
    output_size: int = 64
    activation: Callable[..., Any] = nn.relu

    @nn.compact
    def __call__(self, x, train=False):
        x = nn.Conv(
            features=128,
            kernel_size=(1, 1),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)
        x = nn.Conv(
            features=128,
            kernel_size=(1, 1),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)
        x = nn.Conv(
            features=8,
            kernel_size=(1, 1),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        x = nn.Conv(
            features=16,
            kernel_size=(3, 3),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)
        residual = x
        x = nn.Conv(
            features=16,
            kernel_size=(3, 3),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)
        x += residual
        x = nn.Conv(
            features=16,
            kernel_size=(3, 3),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Conv(
            features=16,
            kernel_size=(3, 3),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)
        residual = x
        x = nn.Conv(
            features=16,
            kernel_size=(3, 3),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)
        x += residual
        x = nn.Conv(
            features=16,
            kernel_size=(3, 3),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = x.reshape((x.shape[0], -1))

        # print("test1", x.shape)

        # x = x.reshape((x.shape[0], -1))

        # print("test2", x.shape)

        x = nn.Dense(
            features=self.output_size,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        # x = nn.BatchNorm(use_running_average=not train)(x)
        x = self.activation(x)

        return x


class MLP(nn.Module):
    """A simple Multilayer perceptron with 1 hidden layer.

    Attributes:
      hidden_size: The size of the hidden layer.
      output_size: The size of the output.
      activation: The activation function to apply to the hidden layer.
      dropout_rate: The dropout rate applied to the hidden layer.
      output_bias: If False, do not use a bias term in the last layer.
      deterministic: Disables dropout if set to True.
    """

    hidden_size: int
    output_size: int
    activation: Callable[..., Any] = nn.relu
    dropout_rate: float = 0.0
    output_bias: bool = False
    # deterministic: bool | None = None

    def setup(self):
        self.intermediate_layer = nn.Dense(self.hidden_size)
        self.output_layer = nn.Dense(self.output_size, use_bias=self.output_bias)
        self.dropout_layer = nn.Dropout(rate=self.dropout_rate)

    def __call__(
        self,
        inputs: jnp.ndarray,
        #   deterministic: bool | None = None,
        train: bool = False,
    ):
        """Applies the MLP to the last dimension of the inputs.

        Args:
          inputs: <float32>[batch_size, ..., input_features].
          deterministic: Disables dropout when set to True.

        Returns:
          The MLP output <float32>[batch_size, ..., output_size]
        """
        # deterministic = nn.module.merge_param(
        #     "deterministic", self.deterministic, deterministic
        # )
        hidden = self.intermediate_layer(inputs)
        hidden = self.activation(hidden)
        # hidden = self.dropout_layer(hidden, deterministic=deterministic)
        output = self.output_layer(hidden)
        return output


class ResNetBlock(nn.Module):
    features: int
    stride: int = 1
    activation: Callable[..., Any] = nn.relu

    @nn.compact
    def __call__(self, x, train: bool = False):
        residual = x

        x = nn.Conv(
            features=self.features,
            kernel_size=(3, 3),
            strides=(self.stride, self.stride),
            padding="SAME",
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        # x = nn.BatchNorm(use_running_average=not train)(x)
        x = self.activation(x)

        x = nn.Conv(
            features=self.features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        # x = nn.BatchNorm(use_running_average=not train)(x)

        if residual.shape != x.shape:
            residual = nn.Conv(
                features=self.features,
                kernel_size=(1, 1),
                strides=(self.stride, self.stride),
                padding="SAME",
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0),
            )(residual)
            # residual = nn.BatchNorm(use_running_average=not train)(residual)

        x += residual
        x = self.activation(x)
        return x


class ResNet(nn.Module):
    # block_sizes: List[int] = [2, 2, 2, 2]
    block_sizes: Sequence[int] = struct.field(default_factory=lambda: [2, 2, 2, 2])
    output_size: int = 10
    activation: Callable[..., Any] = nn.relu

    @nn.compact
    def __call__(self, x, train: bool = False):
        # x = nn.Conv(
        #     features=64,
        #     kernel_size=(7, 7),
        #     strides=(2, 2),
        #     padding="SAME",
        #     kernel_init=orthogonal(jnp.sqrt(2)),
        #     bias_init=constant(0.0),
        # )(x)
        # # x = nn.BatchNorm(use_running_average=not train)(x)
        # x = self.activation(x)
        # x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")

        features = 64
        for i, block_size in enumerate(self.block_sizes):
            for j in range(block_size):
                stride = 2 if i != 0 and j == 0 else 1
                x = ResNetBlock(features, stride=stride, activation=self.activation)(
                    x, train
                )
            features *= 2

        # print("test1", x.shape)
        x = nn.avg_pool(x, window_shape=(x.shape[1], x.shape[2]))
        # print("test2", x.shape)
        x = x.reshape((x.shape[0], -1))
        # x = x.reshape((x.shape[0], x.shape[1], -1))
        # print("test3", x.shape)

        x = nn.Dense(
            features=self.output_size,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        # print("test4", x.shape)
        return x