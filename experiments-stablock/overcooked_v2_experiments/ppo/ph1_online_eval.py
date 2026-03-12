import copy
import csv
import json
import os
import tempfile
import shutil
from contextlib import nullcontext
from collections.abc import Mapping
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
import wandb
from PIL import Image

from overcooked_v2_experiments.eval.policy import PolicyPairing
from overcooked_v2_experiments.eval.rollout import get_rollout
from overcooked_v2_experiments.eval.utils import (
    extract_global_full_obs,
    make_eval_env,
    render_state_frame,
    resolve_old_overcooked_flags,
)
from overcooked_v2_experiments.ppo.policy import PPOPolicy


_PH1_MODE_HISTORY: Dict[int, Dict[str, Dict[str, list]]] = {}
_PH1_VIDEO_TABLE_ROWS: Dict[int, list] = {}
_PH1_MODE_HISTORY_MAX_POINTS = 200
_PH1_METRIC_NAMES = (
    "reward_total",
    "reward_penalized",
    "distance",
    "penalty",
    "pred_accuracy",
)


def _to_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default


def _summarize_tilde_state(tilde_state: Optional[np.ndarray]) -> Dict[str, Any]:
    if tilde_state is None:
        return {
            "tilde_type": "normal",
            "tilde_shape": "None",
            "tilde_min": 0.0,
            "tilde_max": 0.0,
            "tilde_mean": 0.0,
            "tilde_nonneg_ratio": 0.0,
        }

    arr = np.asarray(tilde_state)
    nonneg_ratio = float((arr >= 0).mean()) if arr.size > 0 else 0.0
    return {
        "tilde_type": "state",
        "tilde_shape": str(arr.shape),
        "tilde_min": float(arr.min()) if arr.size > 0 else 0.0,
        "tilde_max": float(arr.max()) if arr.size > 0 else 0.0,
        "tilde_mean": float(arr.mean()) if arr.size > 0 else 0.0,
        "tilde_nonneg_ratio": nonneg_ratio,
    }


def _summarize_first_state(rollout) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        first_pos = np.array(rollout.pos_seq[0])  # (A,2)
        for i in range(first_pos.shape[0]):
            out[f"agent_{i}_pos"] = f"({int(first_pos[i, 0])},{int(first_pos[i, 1])})"
    except Exception:
        pass

    # best-effort recipe/inventory info
    try:
        state0 = jax.tree_util.tree_map(lambda x: x[0], rollout.state_seq)
        recipe = getattr(state0, "recipe", None)
        if recipe is not None:
            out["recipe"] = str(recipe)
        recipes = getattr(state0, "recipes", None)
        if recipes is not None:
            out["recipes"] = str(recipes)

        agents = getattr(state0, "agents", None)
        if agents is not None:
            inv = getattr(agents, "inventory", None)
            if inv is not None:
                out["inventories"] = str(inv)
    except Exception:
        pass

    return out


def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def _encode_policy_metric_emb(policy: PPOPolicy, blocked_actor):
    return policy.network.apply(
        policy.params,
        blocked_actor,
        method=policy.network.encode_blocked,
    )


def _encode_tilde_target_emb(policy: PPOPolicy, num_agents: int, tilde_state: Optional[np.ndarray]):
    if tilde_state is None:
        return None
    try:
        tilde = jnp.asarray(tilde_state, dtype=jnp.float32)
        tilde_actor = jnp.stack([tilde for _ in range(int(num_agents))], axis=0)
        return _encode_policy_metric_emb(policy, tilde_actor)
    except Exception:
        return None


def _compute_rollout_distance_to_target_emb(
    rollout,
    env,
    env_name: str,
    policy: PPOPolicy,
    target_emb,
) -> float:
    if target_emb is None:
        return float("nan")
    try:
        leaves = jax.tree_util.tree_leaves(rollout.state_seq)
        if len(leaves) == 0:
            return float("nan")
        num_steps = int(leaves[0].shape[0])
        if num_steps <= 0:
            return float("nan")
        dist_sum = 0.0
        count = 0
        for t in range(num_steps):
            state_t = jax.tree_util.tree_map(lambda x: x[t], rollout.state_seq)
            global_full = extract_global_full_obs(env, state_t, env_name)
            global_full_actor = jnp.stack([global_full for _ in range(env.num_agents)], axis=0)
            z_next = _encode_policy_metric_emb(policy, global_full_actor)
            lat_dist = jnp.sqrt(jnp.sum((z_next - target_emb) ** 2, axis=-1))
            dist_sum += float(lat_dist.mean())
            count += 1
        if count <= 0:
            return float("nan")
        return float(dist_sum / float(count))
    except Exception:
        return float("nan")


def _to_plain_python(obj):
    if isinstance(obj, Mapping):
        return {str(k): _to_plain_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain_python(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    try:
        return os.fspath(obj)
    except Exception:
        return str(obj)


def _extract_tilde_agent_pos(
    tilde_state: Optional[np.ndarray],
    pos_channel_idx: int,
):
    """
    Extract agent position from a specific channel index of tilde state tensor.
    If unavailable, returns (-1, -1).
    """
    if tilde_state is None:
        return (-1, -1)
    try:
        arr = np.asarray(tilde_state)
        if arr.ndim != 3:
            return (-1, -1)
        h, w, c = arr.shape
        if h <= 0 or w <= 0 or c <= 0:
            return (-1, -1)
        ch = min(max(int(pos_channel_idx), 0), c - 1)
        plane = arr[:, :, ch]
        if plane.size == 0:
            return (-1, -1)
        if float(np.max(plane)) <= 0.0:
            return (-1, -1)
        pos_flat = int(np.argmax(plane))
        y, x = np.unravel_index(pos_flat, plane.shape)
        return (int(y), int(x))
    except Exception:
        return (-1, -1)


def _get_agent_pos_channels_from_env(env, env_name: str = "overcooked_v2"):
    """
    In OvercookedV2 default obs, channel layout (per agent obs) starts with:
      [agent_layer, other_agent_layers, ...]
    and each agent layer starts with its position channel.

    We store tilde{s} as agent_0 perspective full obs, so:
      - agent0 position channel = 0
      - agent1 position channel = size(agent_layer)
    """
    if env_name != "overcooked_v2":
        # In classic overcooked obs, the first channels typically correspond
        # to ego/partner occupancy layers.
        return 0, 1

    try:
        num_ingredients = int(env.layout.num_ingredients)
    except Exception:
        num_ingredients = 2

    # pos(1) + dir(4) + inv_layers(2 + num_ingredients)
    agent_layer_size = 1 + 4 + (2 + num_ingredients)
    return 0, int(agent_layer_size)


def _local_video_path(
    run_subdir: str,
    update_step: int,
    mode: str,
    reward_total: float,
    tilde_state: Optional[np.ndarray],
    agent0_pos_channel: int,
    agent1_pos_channel: int,
    run_base_dir: Optional[str] = None,
    seed: Optional[int] = None,
) -> str:
    # 우선순위:
    # 1) RUN_BASE_DIR/eval/run_{seed}/{update_step}/
    # 2) runs/{run_subdir}/eval/run_{seed}/{update_step}/ (fallback)
    run_key = config_run_key = None
    try:
        config_run_key = int(os.environ.get("PH1_EVAL_RUN_KEY", ""))
    except Exception:
        config_run_key = None
    run_key = config_run_key if config_run_key is not None else seed
    run_seed_dir = f"run_{int(run_key)}" if run_key is not None else "run_0"
    ckpt_dir_name = f"ckpt_{int(update_step)}"
    if run_base_dir is not None and len(str(run_base_dir).strip()) > 0:
        base_dir = os.path.join(
            str(run_base_dir),
            "eval",
            run_seed_dir,
            ckpt_dir_name,
        )
    else:
        base_dir = os.path.join(
            os.getcwd(),
            "runs",
            run_subdir,
            "eval",
            run_seed_dir,
            ckpt_dir_name,
        )
    os.makedirs(base_dir, exist_ok=True)

    p0 = _extract_tilde_agent_pos(tilde_state, agent0_pos_channel)
    p1 = _extract_tilde_agent_pos(tilde_state, agent1_pos_channel)

    reward_str = f"{_safe_float(reward_total):.3f}"
    filename = f"{mode}_{reward_str}_ego({p0[0]},{p0[1]})_partner({p1[0]},{p1[1]}).gif"
    return os.path.join(base_dir, filename)


def _resolve_run_key(seed: int) -> int:
    try:
        return int(os.environ.get("PH1_EVAL_RUN_KEY", ""))
    except Exception:
        return int(seed)


def _resolve_eval_seed_dir(
    run_subdir: str,
    run_base_dir: Optional[str],
    run_key: int,
) -> str:
    if run_base_dir is not None and len(str(run_base_dir).strip()) > 0:
        seed_dir = os.path.join(str(run_base_dir), "eval", f"run_{int(run_key)}")
    else:
        seed_dir = os.path.join(
            os.getcwd(),
            "runs",
            run_subdir,
            "eval",
            f"run_{int(run_key)}",
        )
    os.makedirs(seed_dir, exist_ok=True)
    return seed_dir


def _parse_float(v):
    try:
        return float(v)
    except Exception:
        return float("nan")


def _append_local_eval_metrics(
    *,
    eval_seed_dir: str,
    seed: int,
    run_key: int,
    update_step: int,
    mode_metric_means: Dict[str, Dict[str, float]],
    mode_video_saved: Dict[str, str],
) -> str:
    csv_path = os.path.join(eval_seed_dir, "offline_eval_metrics.csv")
    row_keys = [
        "env_step",
        "seed",
        "run_key",
        "mode",
        "reward_total",
        "reward_penalized",
        "distance",
        "distance_with_recent",
        "distance_with_random",
        "penalty",
        "pred_accuracy",
        "video_path",
    ]
    has_header = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row_keys)
        if not has_header:
            writer.writeheader()
        for mode in ("normal", "recent", "random"):
            metrics = mode_metric_means.get(mode, {})
            writer.writerow(
                {
                    "env_step": int(update_step),
                    "seed": int(seed),
                    "run_key": int(run_key),
                    "mode": mode,
                    "reward_total": _parse_float(metrics.get("reward_total", np.nan)),
                    "reward_penalized": _parse_float(metrics.get("reward_penalized", np.nan)),
                    "distance": _parse_float(metrics.get("distance", np.nan)),
                    "distance_with_recent": _parse_float(metrics.get("distance_with_recent", np.nan)),
                    "distance_with_random": _parse_float(metrics.get("distance_with_random", np.nan)),
                    "penalty": _parse_float(metrics.get("penalty", np.nan)),
                    "pred_accuracy": _parse_float(metrics.get("pred_accuracy", np.nan)),
                    "video_path": mode_video_saved.get(mode, ""),
                }
            )
    return csv_path


def _write_local_eval_plots(csv_path: str, plot_dir: str):
    if not os.path.exists(csv_path):
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[PH1-EVAL][WARN] matplotlib unavailable, skip plots: {type(e).__name__}: {e}")
        return

    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                step = int(float(row.get("env_step", "nan")))
            except Exception:
                continue
            rows.append(
                {
                    "env_step": step,
                    "mode": str(row.get("mode", "")),
                    "reward_total": _parse_float(row.get("reward_total", np.nan)),
                    "reward_penalized": _parse_float(row.get("reward_penalized", np.nan)),
                    "distance": _parse_float(row.get("distance", np.nan)),
                    "distance_with_recent": _parse_float(row.get("distance_with_recent", np.nan)),
                    "distance_with_random": _parse_float(row.get("distance_with_random", np.nan)),
                    "penalty": _parse_float(row.get("penalty", np.nan)),
                    "pred_accuracy": _parse_float(row.get("pred_accuracy", np.nan)),
                }
            )
    if len(rows) == 0:
        return

    # Keep latest record per (env_step, mode) when re-evaluated.
    dedup = {}
    for row in rows:
        dedup[(row["env_step"], row["mode"])] = row

    os.makedirs(plot_dir, exist_ok=True)
    mode_order = ("normal", "recent", "random")
    for metric in _PH1_METRIC_NAMES:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        plotted = False
        for mode in mode_order:
            pts = sorted(
                [v for (step, m), v in dedup.items() if m == mode],
                key=lambda x: x["env_step"],
            )
            xs, ys = [], []
            for p in pts:
                y = _parse_float(p.get(metric, np.nan))
                if np.isnan(y):
                    continue
                xs.append(int(p["env_step"]))
                ys.append(float(y))
            if len(xs) > 0:
                ax.plot(xs, ys, marker="o", linewidth=1.5, label=mode)
                plotted = True
        if plotted:
            ax.set_title(f"PH1 offline eval - {metric}")
            ax.set_xlabel("env_step")
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            out_path = os.path.join(plot_dir, f"ph1_{metric}_modes.png")
            fig.savefig(out_path, dpi=160)
        plt.close(fig)


def _maybe_log_combined_mode_charts_wandb(
    seed: int,
    update_step: int,
    mode_metric_means: Dict[str, Dict[str, float]],
    enabled: bool,
):
    if not enabled:
        return
    mode_order = ["normal", "recent", "random"]
    seed_hist = _PH1_MODE_HISTORY.setdefault(seed, {})

    for metric_name in _PH1_METRIC_NAMES:
        metric_hist = seed_hist.setdefault(
            metric_name,
            {"steps": [], "normal": [], "recent": [], "random": []},
        )
        metric_hist["steps"].append(int(update_step))
        for m in mode_order:
            metric_hist[m].append(float(mode_metric_means.get(m, {}).get(metric_name, np.nan)))

        if len(metric_hist["steps"]) > _PH1_MODE_HISTORY_MAX_POINTS:
            metric_hist["steps"] = metric_hist["steps"][-_PH1_MODE_HISTORY_MAX_POINTS:]
            for m in mode_order:
                metric_hist[m] = metric_hist[m][-_PH1_MODE_HISTORY_MAX_POINTS:]

        chart = wandb.plot.line_series(
            xs=metric_hist["steps"],
            ys=[metric_hist["normal"], metric_hist["recent"], metric_hist["random"]],
            keys=mode_order,
            title=f"PH1 {metric_name} (mode comparison)",
            xname="timestep",
        )
        wandb.log(
            {
                f"rng{seed}/eval_ph1/{metric_name};chart": chart,
                "timestep": int(update_step),
            }
        )


def save_ph1_video_snapshot(
    *,
    params,
    config: Dict[str, Any],
    update_step: int,
    recent_tilde_batch: Optional[np.ndarray],
    seed: int,
) -> Optional[str]:
    """
    Save lightweight snapshot for deferred/offline PH1 video generation.
    Stores model params + mode tildes(normal/recent/random metadata).
    """
    try:
        env_cfg = copy.deepcopy(config["env"]) if "env" in config else {}
        env_kwargs = copy.deepcopy(env_cfg.get("ENV_KWARGS", {}))
        layout = env_kwargs.pop("layout")
        old_overcooked, disable_old_auto = resolve_old_overcooked_flags(config)
        env, env_name, _resolved_kwargs = make_eval_env(
            layout,
            env_kwargs,
            old_overcooked=old_overcooked,
            disable_auto=disable_old_auto,
        )

        random_tilde = None
        try:
            key = jax.random.PRNGKey(seed + update_step * 9973)
            _, key_r = jax.random.split(key)
            _, rand_state = env.reset(key_r)
            random_tilde = np.array(
                extract_global_full_obs(env, rand_state, env_name)
            ).astype(np.float32)
        except Exception:
            random_tilde = None

        recent_tilde = None
        if recent_tilde_batch is not None:
            rb = np.asarray(recent_tilde_batch)
            if rb.ndim >= 4 and rb.shape[0] > 0:
                ridx = (seed + update_step) % rb.shape[0]
                recent_tilde = rb[ridx].astype(np.float32)

        run_name = None
        try:
            if wandb.run is not None:
                run_name = wandb.run.name or wandb.run.id
        except Exception:
            run_name = None
        if run_name is None or len(str(run_name).strip()) == 0:
            run_name = f"{layout}_{str(config.get('ALG_NAME', 'PH1-E3T')).lower()}"
        run_subdir = str(run_name).replace("/", "_")

        run_base_dir = None
        try:
            cfg_run_base = config.get("RUN_BASE_DIR", None)
            if cfg_run_base is not None:
                run_base_dir = os.fspath(cfg_run_base)
        except Exception:
            run_base_dir = None

        run_seed_dir = f"run_{int(seed)}"
        ckpt_dir_name = f"ckpt_{int(update_step)}"
        if run_base_dir is not None and len(str(run_base_dir).strip()) > 0:
            snapshot_root = os.path.join(str(run_base_dir), "eval", run_seed_dir)
        else:
            snapshot_root = os.path.join(os.getcwd(), "runs", run_subdir, "eval", run_seed_dir)

        snapshot_dir = os.path.join(snapshot_root, ckpt_dir_name)
        os.makedirs(snapshot_dir, exist_ok=True)

        # Save mode tildes
        np.savez_compressed(
            os.path.join(snapshot_dir, "mode_tildes.npz"),
            has_recent=np.array([recent_tilde is not None], dtype=np.uint8),
            has_random=np.array([random_tilde is not None], dtype=np.uint8),
            recent_tilde=(
                recent_tilde.astype(np.float32)
                if recent_tilde is not None
                else np.zeros((0,), dtype=np.float32)
            ),
            random_tilde=(
                random_tilde.astype(np.float32)
                if random_tilde is not None
                else np.zeros((0,), dtype=np.float32)
            ),
        )

        metadata = {
            "update_step": int(update_step),
            "seed": int(seed),
            "layout": str(layout),
            "env_name": str(env_name),
        }
        try:
            if wandb.run is not None:
                metadata["wandb_run_id"] = str(wandb.run.id)
                metadata["wandb_run_name"] = str(wandb.run.name)
                metadata["wandb_project"] = str(wandb.run.project)
                metadata["wandb_entity"] = str(wandb.run.entity)
        except Exception:
            pass
        with open(os.path.join(snapshot_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        # Save params + config for offline replay
        from flax.training import orbax_utils
        import orbax.checkpoint as ocp

        orbax_checkpointer = ocp.PyTreeCheckpointer()
        snapshot_ckpt_dir = os.path.join(snapshot_dir, "model_ckpt")
        if os.path.isdir(snapshot_ckpt_dir):
            return snapshot_dir

        snapshot_item = {
            "params": params,
            "config": _to_plain_python(dict(config)),
            "update_step": np.int32(update_step),
            "seed": np.int32(seed),
        }
        save_args = orbax_utils.save_args_from_target(snapshot_item)
        orbax_checkpointer.save(snapshot_ckpt_dir, snapshot_item, save_args=save_args)

        # Robust finalize check:
        # in some callback/thread cases we observed tmp-only leftovers.
        # If final dir is absent but tmp exists, promote latest tmp -> final.
        if not os.path.isdir(snapshot_ckpt_dir):
            tmp_prefix = "model_ckpt.orbax-checkpoint-tmp-"
            tmp_dirs = []
            for name in os.listdir(snapshot_dir):
                p = os.path.join(snapshot_dir, name)
                if os.path.isdir(p) and name.startswith(tmp_prefix):
                    tmp_dirs.append(p)
            if len(tmp_dirs) > 0:
                def _tmp_idx(path: str) -> int:
                    bn = os.path.basename(path)
                    try:
                        return int(bn.rsplit("-", 1)[1])
                    except Exception:
                        return -1
                latest_tmp = sorted(tmp_dirs, key=_tmp_idx)[-1]
                try:
                    os.replace(latest_tmp, snapshot_ckpt_dir)
                except Exception:
                    pass

        if not os.path.isdir(snapshot_ckpt_dir):
            raise RuntimeError(f"snapshot checkpoint finalize failed: {snapshot_ckpt_dir}")

        # completion marker for offline scanner/debugging
        done_path = os.path.join(snapshot_dir, "_SNAPSHOT_DONE")
        with open(done_path, "w", encoding="utf-8") as f:
            f.write("ok\n")

        return snapshot_dir
    except Exception:
        return None


def load_ph1_video_snapshot(snapshot_dir: str):
    """Load snapshot produced by `save_ph1_video_snapshot`."""
    import orbax.checkpoint as ocp

    ckpt_dir = os.path.join(snapshot_dir, "model_ckpt")
    tildes_path = os.path.join(snapshot_dir, "mode_tildes.npz")

    if not os.path.isdir(ckpt_dir):
        tmp_prefix = "model_ckpt.orbax-checkpoint-tmp-"
        tmp_dirs = []
        for name in os.listdir(snapshot_dir):
            p = os.path.join(snapshot_dir, name)
            if os.path.isdir(p) and name.startswith(tmp_prefix):
                tmp_dirs.append(p)
        if len(tmp_dirs) > 0:
            def _tmp_idx(path: str) -> int:
                bn = os.path.basename(path)
                try:
                    return int(bn.rsplit("-", 1)[1])
                except Exception:
                    return -1
            ckpt_dir = sorted(tmp_dirs, key=_tmp_idx)[-1]

    orbax_checkpointer = ocp.PyTreeCheckpointer()
    ckpt = orbax_checkpointer.restore(ckpt_dir, item=None)

    mode_tildes = {
        "normal": None,
        "recent": None,
        "random": None,
    }
    if os.path.exists(tildes_path):
        with np.load(tildes_path) as data:
            has_recent = bool(np.asarray(data.get("has_recent", np.array([0])))[0])
            has_random = bool(np.asarray(data.get("has_random", np.array([0])))[0])
            if has_recent:
                mode_tildes["recent"] = np.asarray(data["recent_tilde"]).astype(np.float32)
            if has_random:
                mode_tildes["random"] = np.asarray(data["random_tilde"]).astype(np.float32)

    return ckpt["params"], ckpt["config"], mode_tildes


def _render_video_from_rollout(
    rollout,
    env_kwargs: Dict[str, Any],
    max_steps: int = 400,
    force_full_view: bool = False,
    env_name: str = "overcooked_v2",
):
    agent_view_size = None if force_full_view else env_kwargs.get("agent_view_size", None)

    try:
        num_steps = int(jax.tree_util.tree_leaves(rollout.state_seq)[0].shape[0])
    except Exception:
        num_steps = 0

    target_steps = int(max_steps)
    if target_steps <= 0:
        target_steps = 400
    if num_steps <= 0:
        return None, False, "empty_rollout"

    try:
        frames = []
        last_frame = None
        for t in range(target_steps):
            src_t = min(t, num_steps - 1)
            try:
                state_t = jax.tree_util.tree_map(lambda x: x[src_t], rollout.state_seq)
                frame_t = render_state_frame(state_t, env_name, agent_view_size)
                frame_np = np.asarray(frame_t)
                if frame_np.ndim == 2:
                    frame_np = np.repeat(frame_np[..., None], 3, axis=-1)
                if frame_np.ndim == 3 and frame_np.shape[-1] == 4:
                    frame_np = frame_np[..., :3]
                last_frame = frame_np
            except Exception:
                if last_frame is None:
                    raise
                frame_np = last_frame
            frames.append(frame_np)

        if len(frames) == 0:
            return None, False, "no_frames"

        frames_np = np.stack(frames, axis=0)
        if frames_np.dtype != np.uint8:
            frames_np = np.clip(frames_np, 0, 255).astype(np.uint8)

        is_static = False
        if frames_np.shape[0] >= 2:
            is_static = bool(np.all(frames_np == frames_np[:1]))

        out_path = os.path.join(
            tempfile.gettempdir(),
            f"ph1_eval_{os.getpid()}_{np.random.randint(0, 1_000_000)}.gif",
        )
        pil_frames = [Image.fromarray(frame) for frame in frames_np]
        pil_frames[0].save(
            out_path,
            save_all=True,
            append_images=pil_frames[1:],
            loop=0,
            duration=250,
            optimize=False,
            disposal=2,
        )
        return out_path, is_static, ""
    except Exception as e:
        return None, False, f"{type(e).__name__}: {e}"


def run_ph1_online_eval(
    *,
    params,
    config: Dict[str, Any],
    update_step: int,
    recent_tilde_batch: Optional[np.ndarray],
    seed: int,
    eval_log_video_override: Optional[bool] = None,
    forced_mode_tildes: Optional[Dict[str, Optional[np.ndarray]]] = None,
):
    if not bool(config.get("PH1_ENABLED", False)):
        return

    if not bool(config.get("PH1_EVAL_ENABLED", False)):
        return

    eval_num_seeds = _to_int(config.get("PH1_EVAL_NUM_SEEDS", 1), default=1)
    eval_viz_episodes = _to_int(config.get("PH1_EVAL_VIZ_EPISODES", 1), default=1)
    eval_viz_max_steps = _to_int(config.get("PH1_EVAL_VIZ_MAX_STEPS", 400), default=400)
    eval_rollout_max_steps = _to_int(
        config.get("PH1_EVAL_ROLLOUT_MAX_STEPS", eval_viz_max_steps),
        default=eval_viz_max_steps,
    )
    if eval_rollout_max_steps <= 0:
        eval_rollout_max_steps = None
    if eval_log_video_override is None:
        eval_log_video = bool(config.get("PH1_EVAL_LOG_VIDEO", True))
    else:
        eval_log_video = bool(eval_log_video_override)

    ph1_omega = float(config.get("PH1_OMEGA", 1.0))
    ph1_sigma = float(config.get("PH1_SIGMA", 1.0))
    eval_disable_jit = bool(config.get("PH1_EVAL_DISABLE_JIT", False))
    eval_force_full_view = bool(config.get("PH1_EVAL_FORCE_FULL_VIEW", False))
    try:
        eval_wandb_log = bool(config.get("PH1_EVAL_WANDB_LOG", False)) and (wandb.run is not None)
    except Exception:
        eval_wandb_log = False
    write_plots = bool(config.get("PH1_EVAL_WRITE_PLOTS", True))
    write_local_csv = bool(config.get("PH1_EVAL_WRITE_LOCAL_CSV", True))

    env_cfg = copy.deepcopy(config["env"]) if "env" in config else {}
    env_kwargs = copy.deepcopy(env_cfg.get("ENV_KWARGS", {}))
    layout = env_kwargs.pop("layout")
    old_overcooked, disable_old_auto = resolve_old_overcooked_flags(config)

    # Eval env uses training-time observation setting (e.g., agent_view_size=2)
    env, env_name, _resolved_kwargs = make_eval_env(
        layout,
        env_kwargs,
        old_overcooked=old_overcooked,
        disable_auto=disable_old_auto,
    )
    agent0_pos_channel, agent1_pos_channel = _get_agent_pos_channels_from_env(
        env, env_name=env_name
    )

    stochastic = bool(config.get("PH1_EVAL_STOCHASTIC", False))
    policy = PPOPolicy(params=params, config=config, stochastic=stochastic)
    pairing = PolicyPairing.from_single_policy(policy, env.num_agents)

    key = jax.random.PRNGKey(seed + update_step * 9973)

    # Build three tilde modes
    random_tilde = None
    try:
        key, key_r = jax.random.split(key)
        _, rand_state = env.reset(key_r)
        random_tilde = np.array(
            extract_global_full_obs(env, rand_state, env_name)
        ).astype(np.float32)
    except Exception:
        random_tilde = None

    recent_tilde = None
    if recent_tilde_batch is not None:
        rb = np.asarray(recent_tilde_batch)
        if rb.ndim >= 4 and rb.shape[0] > 0:
            ridx = (seed + update_step) % rb.shape[0]
            recent_tilde = rb[ridx].astype(np.float32)

    if forced_mode_tildes is None:
        mode_to_tilde = {
            "normal": None,
            "recent": recent_tilde,
            "random": random_tilde,
        }
    else:
        mode_to_tilde = {
            "normal": forced_mode_tildes.get("normal", None),
            "recent": forced_mode_tildes.get("recent", None),
            "random": forced_mode_tildes.get("random", None),
        }
        # If snapshot payload is missing any mode tilde, fallback to live-built tildes.
        if mode_to_tilde["recent"] is None:
            mode_to_tilde["recent"] = recent_tilde
        if mode_to_tilde["random"] is None:
            mode_to_tilde["random"] = random_tilde

    recent_target_emb = _encode_tilde_target_emb(
        policy,
        env.num_agents,
        mode_to_tilde.get("recent", None),
    )
    random_target_emb = _encode_tilde_target_emb(
        policy,
        env.num_agents,
        mode_to_tilde.get("random", None),
    )

    # Example target path (fallback):
    # runs/20260203-082046_j2ixpwiw_grounded_coord_simple_e3t_stablock/video/...
    run_name = None
    try:
        if wandb.run is not None:
            run_name = wandb.run.name or wandb.run.id
    except Exception:
        run_name = None

    if run_name is None or len(str(run_name).strip()) == 0:
        run_name = f"{layout}_{str(config.get('ALG_NAME', 'PH1-E3T')).lower()}"
    run_subdir = str(run_name).replace("/", "_")

    run_base_dir = None
    try:
        cfg_run_base = config.get("RUN_BASE_DIR", None)
        if cfg_run_base is not None:
            run_base_dir = os.fspath(cfg_run_base)
    except Exception:
        run_base_dir = None

    mode_metric_means: Dict[str, Dict[str, float]] = {}
    mode_video_saved: Dict[str, str] = {}

    eval_ctx = jax.disable_jit() if eval_disable_jit else nullcontext()
    with eval_ctx:
        for mode, tilde_state in mode_to_tilde.items():
            if mode in ("recent", "random") and tilde_state is None:
                mode_metric_means[mode] = {
                    "reward_total": np.nan,
                    "reward_penalized": np.nan,
                    "distance": np.nan,
                    "distance_with_recent": np.nan,
                    "distance_with_random": np.nan,
                    "penalty": np.nan,
                    "pred_accuracy": np.nan,
                }
                continue

            mode_rewards = []
            mode_pen_rewards = []
            mode_dist = []
            mode_dist_with_recent = []
            mode_dist_with_random = []
            mode_pen = []
            mode_acc = []
            mode_video = None

            keys = jax.random.split(key, eval_num_seeds)
            for i in range(eval_num_seeds):
                rollout = get_rollout(
                    pairing,
                    env,
                    keys[i],
                    algorithm=str(config.get("ALG_NAME", "PH1-E3T")),
                    ph1_forced_tilde_state=tilde_state,
                    ph1_omega=ph1_omega,
                    ph1_sigma=ph1_sigma,
                    max_rollout_steps=eval_rollout_max_steps,
                    env_device=str(config.get("EVAL_ENV_DEVICE", "cpu")),
                )
                mode_rewards.append(float(rollout.total_reward))
                mode_pen_rewards.append(float(rollout.penalized_total_reward))
                mode_dist.append(float(rollout.ph1_distance_mean))
                if mode == "normal":
                    if recent_target_emb is not None:
                        mode_dist_with_recent.append(
                            _compute_rollout_distance_to_target_emb(
                                rollout=rollout,
                                env=env,
                                env_name=env_name,
                                policy=policy,
                                target_emb=recent_target_emb,
                            )
                        )
                    if random_target_emb is not None:
                        mode_dist_with_random.append(
                            _compute_rollout_distance_to_target_emb(
                                rollout=rollout,
                                env=env,
                                env_name=env_name,
                                policy=policy,
                                target_emb=random_target_emb,
                            )
                        )
                mode_pen.append(float(rollout.ph1_penalty_mean))

                if rollout.prediction_accuracy is not None:
                    mode_acc.append(float(np.array(rollout.prediction_accuracy).mean()))

                if eval_log_video and i < eval_viz_episodes and mode_video is None:
                    video_path, is_static_video, render_err = _render_video_from_rollout(
                        rollout,
                        env_kwargs,
                        max_steps=eval_viz_max_steps,
                        force_full_view=eval_force_full_view,
                        env_name=env_name,
                    )
                    if video_path is not None:
                        mode_video = video_path
                        if eval_wandb_log:
                            wandb.log(
                                {
                                    f"rng{seed}/eval_ph1/{mode}/video_is_static": float(is_static_video),
                                    f"rng{seed}/eval_ph1/{mode}/video_frame_count": int(eval_viz_max_steps),
                                    "eval/update_step": update_step,
                                }
                            )
                    else:
                        if eval_wandb_log:
                            wandb.log(
                                {
                                    f"rng{seed}/eval_ph1/{mode}/video_render_error": str(render_err),
                                    "eval/update_step": update_step,
                                }
                            )

            scalar_log = {
                "reward_total": float(np.mean(mode_rewards)) if mode_rewards else np.nan,
                "reward_penalized": float(np.mean(mode_pen_rewards)) if mode_pen_rewards else np.nan,
                "distance": float(np.mean(mode_dist)) if mode_dist else np.nan,
                "distance_with_recent": (
                    float(np.mean(mode_dist_with_recent))
                    if mode_dist_with_recent
                    else np.nan
                ),
                "distance_with_random": (
                    float(np.mean(mode_dist_with_random))
                    if mode_dist_with_random
                    else np.nan
                ),
                "penalty": float(np.mean(mode_pen)) if mode_pen else np.nan,
                "pred_accuracy": float(np.mean(mode_acc)) if mode_acc else np.nan,
                "eval/update_step": update_step,
            }

            mode_metric_means[mode] = {
                "reward_total": scalar_log["reward_total"],
                "reward_penalized": scalar_log["reward_penalized"],
                "distance": scalar_log["distance"],
                "distance_with_recent": scalar_log["distance_with_recent"],
                "distance_with_random": scalar_log["distance_with_random"],
                "penalty": scalar_log["penalty"],
                "pred_accuracy": scalar_log["pred_accuracy"],
            }

            # save mode video locally (no wandb media upload)
            if mode_video is not None:
                try:
                    local_path = _local_video_path(
                        run_subdir=run_subdir,
                        update_step=update_step,
                        mode=mode,
                        reward_total=mode_metric_means[mode]["reward_total"],
                        tilde_state=tilde_state,
                        agent0_pos_channel=agent0_pos_channel,
                        agent1_pos_channel=agent1_pos_channel,
                        run_base_dir=run_base_dir,
                        seed=seed,
                    )
                    shutil.copyfile(mode_video, local_path)
                    mode_video_saved[mode] = local_path
                    if eval_wandb_log:
                        wandb.log(
                            {
                                f"rng{seed}/eval_ph1/{mode}/video_local_path": local_path,
                                "eval/update_step": update_step,
                            }
                        )
                except Exception as e:
                    if eval_wandb_log:
                        wandb.log(
                            {
                                f"rng{seed}/eval_ph1/{mode}/video_save_error": str(e),
                                "eval/update_step": update_step,
                            }
                        )

            # Keep per-mode arrays local for prompt GC in long runs.
            if eval_num_seeds > 0:
                del rollout
            del (
                mode_rewards,
                mode_pen_rewards,
                mode_dist,
                mode_dist_with_recent,
                mode_dist_with_random,
                mode_pen,
                mode_acc,
            )

    csv_path = ""
    plot_dir = ""
    if write_local_csv:
        run_key = _resolve_run_key(seed)
        eval_seed_dir = _resolve_eval_seed_dir(run_subdir, run_base_dir, run_key)
        csv_path = _append_local_eval_metrics(
            eval_seed_dir=eval_seed_dir,
            seed=seed,
            run_key=run_key,
            update_step=update_step,
            mode_metric_means=mode_metric_means,
            mode_video_saved=mode_video_saved,
        )
        plot_dir = os.path.join(eval_seed_dir, "plots")
        if write_plots:
            _write_local_eval_plots(csv_path, plot_dir)

    # Optional wandb chart logging (disabled by default).
    _maybe_log_combined_mode_charts_wandb(
        seed=seed,
        update_step=update_step,
        mode_metric_means=mode_metric_means,
        enabled=eval_wandb_log,
    )

    # NOTE:
    # - video media is intentionally NOT uploaded to wandb.
    # - table logging is disabled; keep graph/scalar logs only.
    return {
        "csv_path": csv_path,
        "plot_dir": plot_dir,
        "mode_metric_means": mode_metric_means,
        "mode_video_saved": mode_video_saved,
        "plots_written": bool(write_plots),
    }
