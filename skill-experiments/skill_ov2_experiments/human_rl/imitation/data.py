from skill_ov2_experiments.human_rl.human.process_dataframes import (
    get_human_human_trajectories,
)
import jax
import jax.numpy as jnp
import os

LAYOUT_TO_TRIAL_NAME = {
    "cramped_room": "cramped_room",
    "asymm_advantages": "asymmetric_advantages",
    "coord_ring": "coordination_ring",
    "forced_coord": "random0",
    "counter_circuit": "random3",
}

_curr_directory = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_curr_directory, "processed_data")


def preprocess_data(data_path, layout):
    trail_layout_name = LAYOUT_TO_TRIAL_NAME[layout]

    data_params = {
        "layouts": [trail_layout_name],
        "check_trajectories": False,
        "featurize_states": True,
        # "featurize_states": False,
        "data_path": data_path,
    }

    processed_trajs = get_human_human_trajectories(**data_params, silent=False)
    inputs, targets = (
        processed_trajs["ep_states"],
        processed_trajs["ep_actions"],
    )

    inputs, targets = jnp.vstack(inputs), jnp.vstack(targets)
    inputs, targets = jnp.array(inputs, dtype=jnp.int32), jnp.array(
        targets, dtype=jnp.int32
    )
    return inputs, targets


def _get_filename(layout):
    file = os.path.join(DATA_PATH, f"{layout}.npy")
    return file


def data_exists(layout):
    file = _get_filename(layout)
    return os.path.exists(file)


# Load and preprocess data
def load_data(layout):
    file = _get_filename(layout)

    if not os.path.exists(file):
        raise ValueError(f"Data for layout {layout} not found")

    with open(file, "rb") as f:
        inputs = jnp.load(f)
        targets = jnp.load(f)

    inputs = inputs.astype(jnp.float32)

    data = {
        "input": inputs,
        "target": targets,
    }
    return data


def save_data(layout, inputs, targets):
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    file = _get_filename(layout)
    with open(file, "wb") as f:
        jnp.save(f, inputs)
        jnp.save(f, targets)


def split_data(data, key, validation_split=0.15):
    # Create validation split
    N = data["input"].shape[0]
    num_val_samples = int(N * validation_split)

    # Use the provided key to shuffle the data
    shuffled_indices = jax.random.permutation(key, N)
    val_indices = shuffled_indices[:num_val_samples]
    train_indices = shuffled_indices[num_val_samples:]

    def _split_data(data, indices):
        return jax.tree_util.tree_map(lambda x: x[indices], data)

    train_data = _split_data(data, train_indices)
    val_data = _split_data(data, val_indices)

    return train_data, val_data
