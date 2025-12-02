import hydra
import jax
from omegaconf import OmegaConf
import os
import sys

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(DIR))
sys.path.append(os.path.dirname(os.path.dirname(DIR)))

from overcooked_v2_experiments.human_rl.imitation.bc_model import make_train_bc_model
from overcooked_v2_experiments.human_rl.imitation.utils import store_bc_checkpoint
from overcooked_v2_experiments.human_rl.static import (
    CLEAN_2019_HUMAN_DATA_TRAIN,
    CLEAN_2019_HUMAN_DATA_TEST,
    CLEAN_2019_HUMAN_DATA_ALL,
)


def _format_metrics(metrics):
    metric_str = [f"{k}: {v[-1].item()}" for k, v in metrics.items()]
    return ", ".join(metric_str)


@hydra.main(version_base=None, config_path="config", config_name="base")
def main(config):
    config = OmegaConf.to_container(config)

    key = jax.random.PRNGKey(config["SEED"])
    num_seeds = config["NUM_SEEDS"]

    split = config["SPLIT"]
    if split == "train":
        data_path = CLEAN_2019_HUMAN_DATA_TRAIN
    elif split == "test":
        data_path = CLEAN_2019_HUMAN_DATA_TEST
    elif split == "all":
        data_path = CLEAN_2019_HUMAN_DATA_ALL
    else:
        raise ValueError(f"Invalid split: {split}")

    config["DATA_PATH"] = data_path

    # params = train_bc_model(config, key)
    train_jit = jax.jit(make_train_bc_model(config))

    print(f"Training on {config['DATA_PATH']}")

    train_keys = jax.random.split(key, num_seeds)
    out = jax.vmap(train_jit)(train_keys)

    unstacked_out = [
        jax.tree_util.tree_map(lambda x: x[i], out) for i in range(num_seeds)
    ]

    for i, (_, metrics) in enumerate(unstacked_out):
        train_metrics, val_metrics = metrics
        print(
            f"Seed {i}: train: {_format_metrics(train_metrics)}, val: {_format_metrics(val_metrics)}"
        )

    params = [p for p, _ in unstacked_out]
    for i, p in enumerate(params):
        store_bc_checkpoint(config, p, i)


if __name__ == "__main__":
    main()
