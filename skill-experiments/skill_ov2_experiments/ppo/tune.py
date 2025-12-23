import jax
import jax.numpy as jnp
import wandb
from omegaconf import OmegaConf

from skill_ov2_experiments.ppo.run import single_run


def tune(default_config):
    """Hyperparameter sweep with wandb."""

    assert "NUM_ITERATIONS" not in default_config

    import copy

    default_config = OmegaConf.to_container(default_config)
    default_config["NUM_CHECKPOINTS"] = 0
    default_config["NUM_SEEDS"] = 2
    if "FCP" in default_config:
        default_config["NUM_SEEDS"] = 1

    project_name = f"{default_config['wandb']['PROJECT']}-tune"

    def wrapped_make_train():
        wandb.init(project=project_name)
        # update the default params
        config = copy.deepcopy(default_config)
        for k, v in dict(wandb.config).items():
            config["model"][k] = v

        print("running experiment with params:", config)

        out = single_run(config)

    avs_str = "avs-full"
    if "agent_view_size" in default_config["env"]["ENV_KWARGS"]:
        avs_str = f"avs-{default_config['env']['ENV_KWARGS']['agent_view_size']}"

    name = f"ppo_overcooked_{default_config['model']['TYPE']}_{default_config['env']['ENV_KWARGS']['layout']}_{avs_str}"

    sweep_config = {
        "name": name,
        "method": "bayes",
        "metric": {
            "name": "returned_episode_returns",
            "goal": "maximize",
        },
        "parameters": {
            # "NUM_ENVS": {"values": [256, 512]},
            "LR": {"values": [0.00025, 0.0005, 0.00075]},
            "GAE_LAMBDA": {"values": [0.85, 0.9, 0.95]},
            # "ACTIVATION": {"values": ["relu", "tanh"]},
            # "UPDATE_EPOCHS": {"values": [4, 8]},
            # "NUM_MINIBATCHES": {"values": [8, 64]},
            # "CLIP_EPS": {"values": [0.1, 0.2, 0.3]},
            # "ENT_COEF": {"values": [0.01, 0.025]},
            # "NUM_STEPS": {"values": [128, 256]},
            "NUM_ENVS": {"values": [64, 256, 512]},
            # "LR": {"values": [0.0001, 0.0005, 0.00001]},
            # "ACTIVATION": {"values": ["relu", "tanh"]},
            # "UPDATE_EPOCHS": {"values": [2, 4, 8, 16]},
            "NUM_MINIBATCHES": {"values": [16, 32, 64]},
            # "CLIP_EPS": {"values": [0.1, 0.2, 0.3]},
            # "ENT_COEF": {"values": [0.001, 0.01, 0.05]},
            # "NUM_STEPS": {"values": [128, 256, 400]},
        },
    }

    wandb.login()

    if "SWEEP_ID" in default_config:
        sweep_id = default_config["SWEEP_ID"]
    else:
        sweep_id = wandb.sweep(
            sweep_config, entity=default_config["wandb"]["ENTITY"], project=project_name
        )
    wandb.agent(sweep_id, wrapped_make_train, count=1000)
