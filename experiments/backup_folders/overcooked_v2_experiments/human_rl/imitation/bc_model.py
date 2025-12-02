import distrax
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
from flax import struct

from overcooked_v2_experiments.human_rl.human.process_dataframes import (
    get_human_human_trajectories,
)
from .utils import store_bc_checkpoint
from clu import metrics
from flax.linen.initializers import constant, orthogonal
from .data import load_data, split_data
from jaxmarl.environments.overcooked_v2.common import Actions


@struct.dataclass
class Metrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output("loss")


class TrainState(train_state.TrainState):
    metrics: Metrics


# Define the model architecture
class BCModel(nn.Module):
    action_dim: int
    hidden_dims: tuple

    @nn.compact
    def __call__(self, x):
        for dim in self.hidden_dims:
            x = nn.Dense(
                dim,
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0),
            )(x)
            x = nn.relu(x)
        x = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(x)
        return x


@jax.jit
def train_step(state, batch):
    """Train for a single step."""

    def _loss_fn(params):
        logits = state.apply_fn({"params": params}, batch["input"])

        print("Logits shape: ", logits.shape)
        print("Target shape: ", batch["target"].shape)

        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch["target"]
        ).mean()
        return loss

    grad_fn = jax.grad(_loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state


@jax.jit
def compute_metrics(*, state, batch):
    logits = state.apply_fn({"params": state.params}, batch["input"])
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch["target"]
    ).mean()
    metric_updates = state.metrics.single_from_model_output(
        logits=logits, labels=batch["target"], loss=loss
    )
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)
    return state


# Main training loop
def make_train_bc_model(config):
    data = load_data(config["layouts"]["LAYOUT"])

    def train(key):
        key, subkey = jax.random.split(key)
        train_data, val_data = split_data(data, subkey)

        print("Train data shape: ", train_data["input"].shape)
        print("Val data shape: ", val_data["input"].shape)

        model = BCModel(
            action_dim=len(Actions),
            hidden_dims=[64, 64],
        )

        train_inputs_shape = train_data["input"].shape
        num_train_samples = train_inputs_shape[0]

        num_epochs = config["layouts"]["EPOCHS"]
        batch_size = 64

        input_shape = train_inputs_shape[1:]

        key, subkey = jax.random.split(key)
        params = model.init(subkey, jnp.zeros(input_shape))["params"]

        tx = optax.adam(
            learning_rate=config["layouts"]["LR"], eps=config["layouts"]["ADAM_EPS"]
        )

        train_state = TrainState.create(
            apply_fn=model.apply, params=params, tx=tx, metrics=Metrics.empty()
        )

        def _update_epoch(carry, key):
            state = carry

            permutation = jax.random.permutation(key, num_train_samples)
            train_data_shuffled = jax.tree_util.tree_map(lambda x: x[permutation], train_data)

            def _batch_data(data, batch_size):
                num_samples = data["input"].shape[0]
                num_batches = num_samples // batch_size
                return jax.tree_util.tree_map(
                    lambda x: x[: num_batches * batch_size]
                    .reshape((num_batches, batch_size, -1))
                    .squeeze(),
                    data,
                )

            train_data_batched = _batch_data(train_data_shuffled, batch_size)
            val_data_batched = _batch_data(val_data, batch_size)

            def _update_step(carry, batch):
                state = carry

                state = train_step(state, batch)
                state = compute_metrics(state=state, batch=batch)

                return state, None

            train_data_batched = _batch_data(train_data_shuffled, batch_size)
            state, _ = jax.lax.scan(_update_step, state, train_data_batched)
            train_metrics = state.metrics.compute()
            state = state.replace(metrics=state.metrics.empty())

            # Compute metrics on the test set after each training epoch
            def _val_step(carry, batch):
                state = carry
                state = compute_metrics(state=state, batch=batch)
                return state, None

            val_state, _ = jax.lax.scan(_val_step, state, val_data_batched)
            val_metrics = val_state.metrics.compute()

            return state, (train_metrics, val_metrics)

        carry = train_state
        keys = jax.random.split(key, num_epochs)

        # carry, train_metrics = jax.lax.scan(_update_epoch, carry, keys)

        # print("Starting training loop")

        # for epoch in range(num_epochs):
        #     carry, (train_metrics, val_metrics) = _update_epoch(carry, keys[epoch])

        #     def _format_metrics(metrics):
        #         metric_str = [f"{k}: {v.item()}" for k, v in metrics.items()]
        #         return ", ".join(metric_str)

        #     print(f"Train epoch {epoch}:", _format_metrics(train_metrics))
        #     print(f"Val epoch {epoch}:", _format_metrics(val_metrics))

        carry, metrics = jax.lax.scan(_update_epoch, carry, keys)

        train_state = carry

        # print("Training complete")

        return train_state.params, metrics

    return train
