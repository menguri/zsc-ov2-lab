import os
import pickle
from pathlib import Path
import orbax.checkpoint as ocp
from flax.training import orbax_utils
import chex
from flax.core.frozen_dict import FrozenDict

from overcooked_v2_experiments.ppo.policy import PPOPolicy, PPOParams


def _stored_filenames(filename_base):
    model_filename = os.path.join(filename_base, "model.pkl")
    config_filename = os.path.join(filename_base, "config.pkl")

    return model_filename, config_filename


def store_model(network_params, config, filename_base):
    model_filename, config_filename = _stored_filenames(filename_base)

    with open(model_filename, "wb") as f:
        pickle.dump(network_params, f)
    with open(config_filename, "wb") as f:
        pickle.dump(config, f)


def load_model(filename_base):
    model_filename, config_filename = _stored_filenames(filename_base)

    with open(model_filename, "rb") as f:
        network_params = pickle.load(f)
    with open(config_filename, "rb") as f:
        config = pickle.load(f)

    return network_params, config


def _get_checkpoint_dir(run_base_dir, run_num, checkpoint, final=False):
    ckpt_name = "ckpt_final" if final else f"ckpt_{checkpoint}"
    checkpoint_dir = run_base_dir / f"run_{run_num}" / ckpt_name

    return checkpoint_dir.resolve()


def store_checkpoint(config, params, run_num, checkpoint, final=False):
    checkpoint_dir = _get_checkpoint_dir(
        config["RUN_BASE_DIR"], run_num, checkpoint, final=final
    )

    orbax_checkpointer = ocp.PyTreeCheckpointer()

    # Copy config and convert PosixPath to str to avoid orbax serialization error
    config_copy = dict(config)
    if "RUN_BASE_DIR" in config_copy and isinstance(config_copy["RUN_BASE_DIR"], Path):
        config_copy["RUN_BASE_DIR"] = str(config_copy["RUN_BASE_DIR"])

    checkpoint = {
        "config": config_copy,
        "params": params,
    }
    save_args = orbax_utils.save_args_from_target(checkpoint)
    print(
        f"[DEBUG] store_checkpoint: path={checkpoint_dir} final={final} name={'ckpt_final' if final else f'ckpt_{checkpoint}'}"
    )
    orbax_checkpointer.save(checkpoint_dir, checkpoint, save_args=save_args)


def load_checkpoint(run_dir, run_num, checkpoint):
    checkpoint_dir = _get_checkpoint_dir(run_dir, run_num, checkpoint)
    print(f"[DEBUG] Loading checkpoint from: {checkpoint_dir}")

    orbax_checkpointer = ocp.PyTreeCheckpointer()

    ckpt = orbax_checkpointer.restore(checkpoint_dir, item=None)

    return ckpt["config"], ckpt["params"]


def load_all_checkpoints(run_dir, final_only=True, skip_initial=False):
    first_config = None
    all_checkpoints = {}
    configs = {}
    for run_num_dir in run_dir.iterdir():
        print(f"[DEBUG] Examining directory: {run_num_dir.name}")
        if not run_num_dir.is_dir() or "run_" not in run_num_dir.name:
            print(f"[DEBUG] Skipping (not a run dir): {run_num_dir.name}")
            continue
        run_num = int(run_num_dir.name.split("_")[1])
        print(f"[DEBUG] Processing run_num={run_num}")
        checkpoints = {}
        for checkpoint_dir in run_num_dir.iterdir():
            print(f"[DEBUG]   Found checkpoint dir: {checkpoint_dir.name}")
            if not checkpoint_dir.is_dir() or "ckpt_" not in checkpoint_dir.name:
                print(f"[DEBUG]   Skipping (not a ckpt dir): {checkpoint_dir.name}")
                continue
            if final_only and "final" not in checkpoint_dir.name:
                print(f"[DEBUG]   Skipping (final_only and not final): {checkpoint_dir.name}")
                continue
            if skip_initial and "ckpt_0" in checkpoint_dir.name:
                print(f"[DEBUG]   Skipping (skip_initial): {checkpoint_dir.name}")
                continue
            ckpt_id = checkpoint_dir.name.split("_")[1]
            print(f"[DEBUG]   Loading ckpt_id={ckpt_id} for run_num={run_num}")
            config, params = load_checkpoint(run_dir, run_num, ckpt_id)
            policy = PPOParams(params=params)
            checkpoints[checkpoint_dir.name] = policy
            if not first_config:
                first_config = config
        all_checkpoints[run_num_dir.name] = checkpoints
        configs[run_num_dir.name] = config
        print(f"[DEBUG] Loaded {len(checkpoints)} checkpoints for {run_num_dir.name}")
    return all_checkpoints, first_config, configs
