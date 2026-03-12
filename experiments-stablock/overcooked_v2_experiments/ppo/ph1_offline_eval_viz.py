import argparse
import gc
import json
import os
import shutil
from contextlib import nullcontext
from typing import Dict, Optional, Tuple

import jax
import numpy as np

from overcooked_v2_experiments.eval.policy import PolicyPairing
from overcooked_v2_experiments.eval.rollout import get_rollout
from overcooked_v2_experiments.eval.utils import (
    extract_global_full_obs,
    make_eval_env,
    resolve_old_overcooked_flags,
)
from overcooked_v2_experiments.ppo.ph1_online_eval import (
    _extract_tilde_agent_pos,
    _get_agent_pos_channels_from_env,
    _render_video_from_rollout,
    load_ph1_video_snapshot,
)
from overcooked_v2_experiments.ppo.policy import PPOPolicy


def _snapshot_step(path: str) -> Optional[int]:
    base = os.path.basename(path)
    if base == "ckpt_final":
        return 10**18
    if base.startswith("ckpt_"):
        try:
            return int(base.split("_", 1)[1])
        except Exception:
            return None
    if base.isdigit():
        return int(base)
    if base.startswith("step_"):
        try:
            return int(base.split("_")[1])
        except Exception:
            return None
    return None


def _is_valid_snapshot_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    if _snapshot_step(path) is None:
        return False
    ckpt_dir = os.path.join(path, "model_ckpt")
    done_marker = os.path.join(path, "_SNAPSHOT_DONE")
    if os.path.exists(done_marker) or os.path.isdir(ckpt_dir):
        return True
    try:
        for sub in os.listdir(path):
            if sub.startswith("model_ckpt.orbax-checkpoint-tmp-") and os.path.isdir(os.path.join(path, sub)):
                return True
    except Exception:
        return False
    return False


def _to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def _read_metadata(snapshot_dir: str):
    path = os.path.join(snapshot_dir, "metadata.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _run_key_from_dir(run_dir: str) -> int:
    name = os.path.basename(os.path.normpath(run_dir))
    if name.startswith("run_") or name.startswith("seed_"):
        try:
            return int(name.split("_", 1)[1])
        except Exception:
            return 0
    return 0


def _collect_run_dirs(snapshot_root: str):
    if not os.path.isdir(snapshot_root):
        return []
    run_dirs = []
    for name in sorted(os.listdir(snapshot_root)):
        p = os.path.join(snapshot_root, name)
        if os.path.isdir(p) and (name.startswith("run_") or name.startswith("seed_")):
            run_dirs.append(p)
    if len(run_dirs) > 0:
        return run_dirs
    # Legacy single-run layout under eval/
    return [snapshot_root]


def _pick_final_snapshot(run_dir: str) -> Optional[str]:
    final_dir = os.path.join(run_dir, "ckpt_final")
    if _is_valid_snapshot_dir(final_dir):
        return final_dir

    candidates = []
    try:
        for name in os.listdir(run_dir):
            p = os.path.join(run_dir, name)
            if _is_valid_snapshot_dir(p):
                candidates.append(p)
    except Exception:
        return None
    if len(candidates) == 0:
        return None
    return sorted(candidates, key=lambda x: _snapshot_step(x) or -1)[-1]


def _clear_jax_runtime_state():
    try:
        jax.clear_caches()
    except Exception:
        pass
    gc.collect()


def _tilde_name_positions(
    tilde_state,
    agent0_pos_channel: int,
    agent1_pos_channel: int,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    p0 = _extract_tilde_agent_pos(tilde_state, agent0_pos_channel)
    p1 = _extract_tilde_agent_pos(tilde_state, agent1_pos_channel)
    return p0, p1


def _safe_float(v, default=float("nan")) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _build_video_path(
    run_base_dir: str,
    run_key: int,
    mode: str,
    ego_pos: Tuple[int, int],
    par_pos: Tuple[int, int],
    reward_total: float,
) -> str:
    video_dir = os.path.join(str(run_base_dir), "eval", "video")
    os.makedirs(video_dir, exist_ok=True)
    base_name = (
        f"run_{int(run_key)}_{str(mode).strip().lower()}_ego({int(ego_pos[0])},{int(ego_pos[1])})"
        f"_par({int(par_pos[0])},{int(par_pos[1])})_reward{_safe_float(reward_total):.3f}.gif"
    )
    out_path = os.path.join(video_dir, base_name)
    if not os.path.exists(out_path):
        return out_path

    stem, ext = os.path.splitext(base_name)
    idx = 2
    while True:
        cand = os.path.join(video_dir, f"{stem}_{idx}{ext}")
        if not os.path.exists(cand):
            return cand
        idx += 1


def main():
    parser = argparse.ArgumentParser(description="PH1 offline final-checkpoint video extraction.")
    parser.add_argument("--run-base-dir", type=str, required=True)
    parser.add_argument("--viz-max-steps", type=int, default=400)
    parser.add_argument("--disable-jit", type=str, default="True")
    parser.add_argument("--env-device", type=str, default="cpu", help="Env interaction device: cpu|gpu")
    args = parser.parse_args()
    args.run_base_dir = os.path.abspath(str(args.run_base_dir))

    viz_steps = 400
    if int(args.viz_max_steps) != viz_steps:
        print(f"[PH1-EVAL-VIZ] override viz_max_steps {int(args.viz_max_steps)} -> {viz_steps}")
    disable_jit = _to_bool(args.disable_jit)
    env_device = str(args.env_device).strip().lower()

    snapshot_root = os.path.join(args.run_base_dir, "eval")
    run_dirs = _collect_run_dirs(snapshot_root)
    if len(run_dirs) == 0:
        print(f"[PH1-EVAL-VIZ] No run directories found: {snapshot_root}")
        return

    videos_written = 0
    runs_processed = 0
    for run_dir in run_dirs:
        run_key = _run_key_from_dir(run_dir)
        final_snapshot = _pick_final_snapshot(run_dir)
        if final_snapshot is None:
            print(f"[PH1-EVAL-VIZ][WARN] no valid snapshot in {run_dir}")
            continue
        if os.path.basename(final_snapshot) != "ckpt_final":
            print(
                f"[PH1-EVAL-VIZ][WARN] ckpt_final missing for run_{int(run_key)}; using {os.path.basename(final_snapshot)}"
            )

        try:
            params, config, mode_tildes = load_ph1_video_snapshot(final_snapshot)
            config = dict(config)
        except Exception as e:
            print(f"[PH1-EVAL-VIZ][WARN] snapshot load failed run_{int(run_key)}: {type(e).__name__}: {e}")
            continue

        metadata = _read_metadata(final_snapshot)
        seed = int(metadata.get("seed", 42 + int(run_key)))

        try:
            env_cfg = dict(config.get("env", {}))
            env_kwargs = dict(env_cfg.get("ENV_KWARGS", {}))
            layout = env_kwargs.pop("layout")
        except Exception as e:
            print(f"[PH1-EVAL-VIZ][WARN] invalid env config run_{int(run_key)}: {type(e).__name__}: {e}")
            continue

        old_overcooked, disable_old_auto = resolve_old_overcooked_flags(config)
        env, env_name, _resolved_kwargs = make_eval_env(
            layout,
            env_kwargs,
            old_overcooked=old_overcooked,
            disable_auto=disable_old_auto,
        )
        agent0_pos_channel, agent1_pos_channel = _get_agent_pos_channels_from_env(
            env, env_name=env_name
        )
        config["EVAL_ENV_DEVICE"] = env_device

        stochastic = bool(config.get("PH1_EVAL_STOCHASTIC", False))
        policy = PPOPolicy(params=params, config=config, stochastic=stochastic)
        pairing = PolicyPairing.from_single_policy(policy, env.num_agents)

        recent_tilde = mode_tildes.get("recent")
        random_tilde = mode_tildes.get("random")
        if recent_tilde is None:
            try:
                key_r = jax.random.PRNGKey(seed + int(run_key) * 17 + 1)
                _, st = env.reset(key_r)
                recent_tilde = np.asarray(
                    extract_global_full_obs(env, st, env_name)
                ).astype(np.float32)
            except Exception:
                recent_tilde = None
        if random_tilde is None:
            try:
                key_q = jax.random.PRNGKey(seed + int(run_key) * 17 + 2)
                _, st = env.reset(key_q)
                random_tilde = np.asarray(
                    extract_global_full_obs(env, st, env_name)
                ).astype(np.float32)
            except Exception:
                random_tilde = None

        modes = {
            "normal": None,
            "recent": recent_tilde,
            "random": random_tilde,
        }

        eval_ctx = jax.disable_jit() if disable_jit else nullcontext()
        with eval_ctx:
            for mode, tilde in modes.items():
                if mode in ("recent", "random") and tilde is None:
                    print(f"[PH1-EVAL-VIZ][WARN] skip run_{int(run_key)} mode={mode}: missing tilde state")
                    continue
                key = jax.random.PRNGKey(seed + int(run_key) * 101 + (0 if mode == "normal" else (1 if mode == "recent" else 2)))
                rollout = get_rollout(
                    pairing,
                    env,
                    key,
                    algorithm=str(config.get("ALG_NAME", "PH1-E3T")),
                    ph1_forced_tilde_state=tilde,
                    ph1_omega=float(config.get("PH1_OMEGA", 1.0)),
                    ph1_sigma=float(config.get("PH1_SIGMA", 1.0)),
                    max_rollout_steps=int(viz_steps),
                    env_device=str(config.get("EVAL_ENV_DEVICE", "cpu")),
                )
                reward_total = float(rollout.total_reward)
                temp_video, _, render_err = _render_video_from_rollout(
                    rollout,
                    env_kwargs,
                    max_steps=int(viz_steps),
                    force_full_view=False,
                    env_name=env_name,
                )
                if temp_video is None:
                    print(
                        f"[PH1-EVAL-VIZ][WARN] render failed run_{int(run_key)} mode={mode}: {render_err}"
                    )
                    continue
                ego_pos, par_pos = _tilde_name_positions(
                    tilde,
                    agent0_pos_channel,
                    agent1_pos_channel,
                )
                out_path = _build_video_path(
                    args.run_base_dir,
                    run_key,
                    mode,
                    ego_pos,
                    par_pos,
                    reward_total,
                )
                shutil.copyfile(temp_video, out_path)
                try:
                    os.remove(temp_video)
                except Exception:
                    pass
                videos_written += 1
                print(
                    f"[PH1-EVAL-VIZ] saved run_{int(run_key)} mode={mode} reward={reward_total:.3f} path={out_path}"
                )

        runs_processed += 1
        _clear_jax_runtime_state()

    print(f"[PH1-EVAL-VIZ] Done. runs_processed={runs_processed} videos_written={videos_written}")


if __name__ == "__main__":
    main()
