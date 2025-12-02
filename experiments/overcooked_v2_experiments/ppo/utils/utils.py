from datetime import datetime
from pathlib import Path
import os
import jax
import jax.numpy as jnp


def _infer_run_suffix(config) -> str:
    """Return the suffix (sp/sa/uc/...) used for folder + default run names."""
    exp = config.get("EXP", "")
    if exp and "-" in exp:
        suffix = exp.split("-")[-1]
    else:
        model_name = config["model"]["TYPE"].upper()
        if "NUM_ITERATIONS" in config:
            # state augmentation(k-step state buffer) 실험 구분용
            suffix = "sa"
        elif model_name == "RNN":
            suffix = "sp"
        elif model_name == "CNN":
            suffix = "sp"
        else:
            suffix = model_name.lower()

    confidence_profile = config.get("confidence")
    if confidence_profile:
        confidence_pretty = confidence_profile.replace("_", "-")
        if suffix == "sp":
            suffix = confidence_pretty
        else:
            suffix = f"{suffix}-{confidence_pretty}"

    return suffix


def build_default_run_name(config) -> str:
    agent_view_size = config["env"]["ENV_KWARGS"].get("agent_view_size", None)
    layout_name = config["env"]["ENV_KWARGS"]["layout"]
    model_name = config["model"]["TYPE"]

    avs_str = f"avs-{agent_view_size}" if agent_view_size is not None else "avs-full"
    suffix = _infer_run_suffix(config)

    return f"{suffix}_{layout_name}_{model_name.lower()}_{avs_str}"


def get_run_base_dir(run_id: str, config) -> str:
    optional_prefix = config.get("OPTIONAL_PREFIX", "")

    layout_name = config["env"]["ENV_KWARGS"]["layout"]

    results_dir = "runs"
    if optional_prefix != "":
        results_dir = os.path.join(results_dir, optional_prefix)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    suffix = _infer_run_suffix(config)

    if "FCP" in config:
        dir = Path(config["FCP"])
        f = dir.name
        run_dir = os.path.join(results_dir, f"{timestamp}_{run_id}_{layout_name}_fcp")
    else:
        run_dir = os.path.join(
            results_dir, f"{timestamp}_{run_id}_{layout_name}_{suffix}"
        )
    os.makedirs(run_dir, exist_ok=True)

    print("run_dir", run_dir)

    return Path(run_dir)


def combine_first_two_tree_dim(tree):
    return jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), tree)


def get_num_devices():
    num_devices = 1

    try:
        devices = jax.devices("gpu")
        if devices:
            num_devices = len(devices)
            print(f"GPU is available! Using {num_devices} GPUs.")
        else:
            print("No GPU found, falling back to CPU.")
    except RuntimeError as e:
        print("Warning: Falling back to CPU.")

    return num_devices
