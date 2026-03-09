from pathlib import Path
import orbax.checkpoint as ocp
from flax.training import orbax_utils
from datetime import datetime
import jax
import jax.numpy as jnp

# get directory of current file
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"


def _get_model_dir(layout, split, run_id):
    # model_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    bc_name = f"{layout}-{run_id}"

    return MODEL_DIR / split / bc_name


def store_bc_checkpoint(config, params, run_id):
    model_dir = _get_model_dir(config["layouts"]["LAYOUT"], config["SPLIT"], run_id)

    orbax_checkpointer = ocp.PyTreeCheckpointer()

    checkpoint = {
        "config": config,
        "params": params,
    }
    save_args = orbax_utils.save_args_from_target(checkpoint)
    orbax_checkpointer.save(model_dir, checkpoint, save_args=save_args, force=True)


def load_bc_checkpoint(layout, split, run_id):
    model_dir = _get_model_dir(layout, split, run_id)

    orbax_checkpointer = ocp.PyTreeCheckpointer()
    ckpt = orbax_checkpointer.restore(model_dir, item=None)

    return ckpt["config"], ckpt["params"]


def remove_indices_and_renormalize(probs, indices):
    assert probs.ndim == 1
    assert indices.ndim == 1

    probs = probs.at[indices].set(0)
    alt_probs = jnp.ones_like(probs).at[indices].set(0)
    sum_probs = probs.sum()
    probs = jax.lax.select(sum_probs > 0, probs, alt_probs)
    return probs / probs.sum()
