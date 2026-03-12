import argparse
import csv
import gc
import json
import os
import shutil
from contextlib import nullcontext
from typing import Any, Dict, List, Optional

import jax
import numpy as np
import orbax.checkpoint as ocp

from overcooked_v2_experiments.eval.policy import PolicyPairing
from overcooked_v2_experiments.eval.rollout import get_rollout
from overcooked_v2_experiments.eval.utils import (
    extract_global_full_obs,
    make_eval_env,
    resolve_old_overcooked_flags,
)
from overcooked_v2_experiments.ppo.ph1_online_eval import (
    _render_video_from_rollout,
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


def _collect_snapshot_dirs(snapshot_root: str):
    if not os.path.isdir(snapshot_root):
        return []
    dirs = []
    for name in os.listdir(snapshot_root):
        p = os.path.join(snapshot_root, name)
        if not (os.path.isdir(p) and _snapshot_step(p) is not None):
            continue
        ckpt_dir = os.path.join(p, "model_ckpt")
        done_marker = os.path.join(p, "_SNAPSHOT_DONE")
        has_tmp = False
        try:
            has_tmp = any(
                x.startswith("model_ckpt.orbax-checkpoint-tmp-") and os.path.isdir(os.path.join(p, x))
                for x in os.listdir(p)
            )
        except Exception:
            has_tmp = False
        if os.path.exists(done_marker) or os.path.isdir(ckpt_dir) or has_tmp:
            dirs.append(p)
    return sorted(dirs, key=lambda p: _snapshot_step(p) or 10**18)


def _collect_snapshot_dirs_multi(snapshot_root: str):
    # Prefer eval/seed_{seed}/ckpt_{step}; fallback to legacy layouts.
    if not os.path.isdir(snapshot_root):
        return []
    run_seed_dirs = []
    for name in os.listdir(snapshot_root):
        p = os.path.join(snapshot_root, name)
        if os.path.isdir(p) and (name.startswith("seed_") or name.startswith("run_")):
            run_seed_dirs.append(p)
    if len(run_seed_dirs) == 0:
        return _collect_snapshot_dirs(snapshot_root)
    out = []
    for d in sorted(run_seed_dirs):
        out.extend(_collect_snapshot_dirs(d))
    return out


def _run_key_from_snapshot(snapshot_dir: str) -> int:
    p = snapshot_dir
    while True:
        p2 = os.path.dirname(p)
        if p2 == p:
            break
        bn = os.path.basename(p)
        if bn.startswith("run_"):
            try:
                return int(bn.split("_", 1)[1])
            except Exception:
                return 0
        p = p2
    return 0


def _read_metadata(snapshot_dir: str):
    path = os.path.join(snapshot_dir, "metadata.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _read_mode_tildes(snapshot_dir: str):
    mode_tildes = {"normal": None, "recent": None, "random": None}
    tildes_path = os.path.join(snapshot_dir, "mode_tildes.npz")
    if not os.path.exists(tildes_path):
        return mode_tildes
    try:
        with np.load(tildes_path) as data:
            has_recent = bool(np.asarray(data.get("has_recent", np.array([0])))[0])
            has_random = bool(np.asarray(data.get("has_random", np.array([0])))[0])
            if has_recent:
                mode_tildes["recent"] = np.asarray(data["recent_tilde"]).astype(np.float32)
            if has_random:
                mode_tildes["random"] = np.asarray(data["random_tilde"]).astype(np.float32)
    except Exception:
        pass
    return mode_tildes


def _load_dual_snapshot(snapshot_dir: str):
    ckpt_dir = os.path.join(snapshot_dir, "model_ckpt")
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    ckpt = orbax_checkpointer.restore(ckpt_dir, item=None)
    params_spec = ckpt.get("params_spec", ckpt.get("params", None))
    params_ind = ckpt.get("params_ind", ckpt.get("params", None))
    config = ckpt.get("config", {})
    ckpt_seed = ckpt.get("seed", None)
    ckpt_update_step = ckpt.get("update_step", None)
    mode_tildes = _read_mode_tildes(snapshot_dir)
    return params_spec, params_ind, config, mode_tildes, ckpt_seed, ckpt_update_step


def _to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def _normalize_env_step(raw_step: int, config: dict, eval_every_env_steps: int) -> int:
    """
    Backward compatibility:
    - old snapshots may store update-step (e.g., 13, 25)
    - new snapshots store env-step (e.g., 200000, 400000)
    """
    step = int(raw_step)
    if eval_every_env_steps <= 0:
        return step
    model_cfg = dict(config.get("model", {})) if isinstance(config.get("model", {}), dict) else {}
    num_steps = int(model_cfg.get("NUM_STEPS", 1))
    num_envs = int(model_cfg.get("NUM_ENVS", 1))
    steps_per_update = max(1, num_steps * num_envs)
    if step < eval_every_env_steps:
        cand = int(step * steps_per_update)
        if cand >= eval_every_env_steps and (cand % eval_every_env_steps) == 0:
            return cand
    return step


def _parse_float(v):
    try:
        return float(v)
    except Exception:
        return float("nan")


def _clear_jax_runtime_state():
    try:
        jax.clear_caches()
    except Exception:
        pass
    gc.collect()


def _append_local_records(eval_seed_dir: str, rows: List[Dict[str, Any]]) -> str:
    csv_path = os.path.join(eval_seed_dir, "offline_eval_metrics_ph2.csv")
    fieldnames = [
        "env_step",
        "seed",
        "run_key",
        "pair_name",
        "mode",
        "reward_total",
        "video_path",
    ]
    has_header = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not has_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "env_step": int(row.get("env_step", 0)),
                    "seed": int(row.get("seed", 0)),
                    "run_key": int(row.get("run_key", 0)),
                    "pair_name": str(row.get("pair_name", "")),
                    "mode": str(row.get("mode", "")),
                    "reward_total": _parse_float(row.get("reward_total", np.nan)),
                    "video_path": str(row.get("video_path", "")),
                }
            )
    return csv_path


def _write_local_plots(csv_path: str, plot_dir: str):
    if not os.path.exists(csv_path):
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[PH2-OFFLINE][WARN] matplotlib unavailable, skip plots: {type(e).__name__}: {e}")
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
                    "pair_name": str(row.get("pair_name", "")),
                    "mode": str(row.get("mode", "")),
                    "reward_total": _parse_float(row.get("reward_total", np.nan)),
                }
            )
    if len(rows) == 0:
        return

    # Keep latest value per (pair, mode, env_step).
    dedup = {}
    for row in rows:
        dedup[(row["pair_name"], row["mode"], row["env_step"])] = row

    os.makedirs(plot_dir, exist_ok=True)
    pairs = sorted({k[0] for k in dedup.keys()})
    for pair_name in pairs:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        plotted = False
        for mode in ("normal", "recent", "random"):
            pts = sorted(
                [v for (p, m, _), v in dedup.items() if p == pair_name and m == mode],
                key=lambda x: x["env_step"],
            )
            xs, ys = [], []
            for p in pts:
                y = _parse_float(p.get("reward_total", np.nan))
                if np.isnan(y):
                    continue
                xs.append(int(p["env_step"]))
                ys.append(float(y))
            if len(xs) > 0:
                ax.plot(xs, ys, marker="o", linewidth=1.5, label=mode)
                plotted = True
        if plotted:
            ax.set_title(f"PH2 offline eval - {pair_name} reward_total")
            ax.set_xlabel("env_step")
            ax.set_ylabel("reward_total")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(plot_dir, f"ph2_{pair_name}_reward_total.png"), dpi=160)
        plt.close(fig)


def _save_pair_video(run_base_dir: str, step: int, pair_name: str, mode: str, reward_total: float, temp_video_path: str, run_key: int):
    step_dir = os.path.join(run_base_dir, "eval", f"run_{int(run_key)}", f"ckpt_{int(step)}")
    os.makedirs(step_dir, exist_ok=True)
    out_name = f"{pair_name}_{mode}_{float(reward_total):.3f}.gif"
    out_path = os.path.join(step_dir, out_name)
    shutil.copyfile(temp_video_path, out_path)
    return out_path


def _evaluate_pair(
    *,
    pair_name: str,
    policy_a: PPOPolicy,
    policy_b: PPOPolicy,
    config: dict,
    mode_tildes: dict,
    step: int,
    seed: int,
    run_key: int,
    should_video: bool,
    viz_max_steps: int,
    local_rows: List[Dict[str, Any]],
):
    env_cfg = dict(config.get("env", {}))
    env_kwargs = dict(env_cfg.get("ENV_KWARGS", {}))
    layout = env_kwargs.pop("layout")
    old_overcooked, disable_old_auto = resolve_old_overcooked_flags(config)
    env, env_name, _resolved_kwargs = make_eval_env(
        layout,
        env_kwargs,
        old_overcooked=old_overcooked,
        disable_auto=disable_old_auto,
    )

    recent_tilde = mode_tildes.get("recent")
    random_tilde = mode_tildes.get("random")
    if recent_tilde is None:
        try:
            key_r = jax.random.PRNGKey(seed + step * 13 + 1)
            _, st = env.reset(key_r)
            recent_tilde = np.asarray(
                extract_global_full_obs(env, st, env_name)
            ).astype(np.float32)
        except Exception:
            recent_tilde = None
    if random_tilde is None:
        try:
            key_q = jax.random.PRNGKey(seed + step * 13 + 2)
            _, st = env.reset(key_q)
            random_tilde = np.asarray(
                extract_global_full_obs(env, st, env_name)
            ).astype(np.float32)
        except Exception:
            random_tilde = None

    modes = {"normal": None, "recent": recent_tilde, "random": random_tilde}
    eval_force_full_view = bool(config.get("EVAL_FORCE_FULL_VIEW", False))
    for mode, tilde in modes.items():
        if mode in ("recent", "random") and tilde is None:
            local_rows.append(
                {
                    "env_step": int(step),
                    "seed": int(seed),
                    "run_key": int(run_key),
                    "pair_name": str(pair_name),
                    "mode": mode,
                    "reward_total": np.nan,
                    "video_path": "",
                }
            )
            continue
        key = jax.random.PRNGKey(seed + step * 31 + (0 if mode == "normal" else (1 if mode == "recent" else 2)))
        rollout = get_rollout(
            PolicyPairing(policy_a, policy_b),
            env,
            key,
            algorithm="PH1-E3T",
            ph1_forced_tilde_state=tilde,
            ph1_omega=float(config.get("PH1_OMEGA", 1.0)),
            ph1_sigma=float(config.get("PH1_SIGMA", 1.0)),
            max_rollout_steps=(int(viz_max_steps) if int(viz_max_steps) > 0 else None),
            env_device=str(config.get("EVAL_ENV_DEVICE", "cpu")),
        )
        r = float(rollout.total_reward)
        video_out_path = ""

        if should_video:
            video_path, _, render_err = _render_video_from_rollout(
                rollout,
                env_kwargs,
                max_steps=int(viz_max_steps),
                force_full_view=eval_force_full_view,
                env_name=env_name,
            )
            if video_path is not None:
                video_out_path = _save_pair_video(
                    str(config.get("RUN_BASE_DIR", "")),
                    step,
                    pair_name,
                    mode,
                    r,
                    video_path,
                    run_key,
                )
            elif render_err:
                print(
                    f"[PH2-OFFLINE][WARN] video render failed pair={pair_name} mode={mode} step={int(step)}: {render_err}"
                )

        local_rows.append(
            {
                "env_step": int(step),
                "seed": int(seed),
                "run_key": int(run_key),
                "pair_name": str(pair_name),
                "mode": mode,
                "reward_total": float(r),
                "video_path": video_out_path,
            }
        )


def _evaluate_ind_ind(
    *,
    policy_ind: PPOPolicy,
    config: dict,
    step: int,
    seed: int,
    run_key: int,
    should_video: bool,
    viz_max_steps: int,
    local_rows: List[Dict[str, Any]],
):
    env_cfg = dict(config.get("env", {}))
    env_kwargs = dict(env_cfg.get("ENV_KWARGS", {}))
    layout = env_kwargs.pop("layout")
    old_overcooked, disable_old_auto = resolve_old_overcooked_flags(config)
    env, env_name, _resolved_kwargs = make_eval_env(
        layout,
        env_kwargs,
        old_overcooked=old_overcooked,
        disable_auto=disable_old_auto,
    )
    key = jax.random.PRNGKey(seed + step * 97)
    rollout = get_rollout(
        PolicyPairing(policy_ind, policy_ind),
        env,
        key,
        algorithm="PH1-E3T",
        ph1_forced_tilde_state=None,
        ph1_omega=float(config.get("PH1_OMEGA", 1.0)),
        ph1_sigma=float(config.get("PH1_SIGMA", 1.0)),
        max_rollout_steps=(int(viz_max_steps) if int(viz_max_steps) > 0 else None),
        env_device=str(config.get("EVAL_ENV_DEVICE", "cpu")),
    )
    r = float(rollout.total_reward)
    video_out_path = ""
    eval_force_full_view = bool(config.get("EVAL_FORCE_FULL_VIEW", False))
    if should_video:
        video_path, _, render_err = _render_video_from_rollout(
            rollout,
            env_kwargs,
            max_steps=int(viz_max_steps),
            force_full_view=eval_force_full_view,
            env_name=env_name,
        )
        if video_path is not None:
            video_out_path = _save_pair_video(
                str(config.get("RUN_BASE_DIR", "")),
                step,
                "ind_ind",
                "normal",
                r,
                video_path,
                run_key,
            )
        elif render_err:
            print(f"[PH2-OFFLINE][WARN] video render failed pair=ind_ind mode=normal step={int(step)}: {render_err}")

    local_rows.append(
        {
            "env_step": int(step),
            "seed": int(seed),
            "run_key": int(run_key),
            "pair_name": "ind_ind",
            "mode": "normal",
            "reward_total": float(r),
            "video_path": video_out_path,
        }
    )


def main():
    parser = argparse.ArgumentParser(description="PH2 offline evaluation from eval snapshots.")
    parser.add_argument("--run-base-dir", type=str, required=True)
    parser.add_argument("--eval-every-env-steps", type=int, default=200000)
    parser.add_argument("--video-every-env-steps", type=int, default=1000000)
    parser.add_argument("--log-video", type=str, default="False")
    parser.add_argument("--viz-max-steps", type=int, default=400)
    parser.add_argument("--disable-jit", type=str, default="True", help="Disable JAX JIT during offline eval (True/False)")
    parser.add_argument("--env-device", type=str, default="cpu", help="Env interaction device: cpu|gpu")
    args = parser.parse_args()
    args.run_base_dir = os.path.abspath(str(args.run_base_dir))
    disable_jit = _to_bool(args.disable_jit)
    viz_steps = 400
    if int(args.viz_max_steps) != viz_steps:
        print(f"[PH2-OFFLINE] override viz_max_steps {int(args.viz_max_steps)} -> {viz_steps}")

    snapshot_root = os.path.join(args.run_base_dir, "eval")
    snapshot_dirs = _collect_snapshot_dirs_multi(snapshot_root)
    if len(snapshot_dirs) == 0:
        print(f"[PH2-OFFLINE] No snapshots found: {snapshot_root}")
        return

    seen_eval_steps_by_seed = {}

    def _seed_key(snapshot_dir: str, seed: int) -> int:
        if seed is not None:
            return int(seed)
        p = snapshot_dir
        while True:
            p2 = os.path.dirname(p)
            if p2 == p:
                break
            bn = os.path.basename(p)
            if bn.startswith("seed_"):
                try:
                    return int(bn.split("_", 1)[1])
                except Exception:
                    return 0
            if bn.startswith("run_"):
                try:
                    return int(bn.split("_", 1)[1])
                except Exception:
                    return 0
            p = p2
        return 0

    processed = 0
    csv_and_plot_dirs = set()
    for snapshot_dir in snapshot_dirs:
        raw_step = _snapshot_step(snapshot_dir)
        if raw_step is None:
            continue

        params_spec, params_ind, config, mode_tildes, ckpt_seed, ckpt_update_step = _load_dual_snapshot(snapshot_dir)
        metadata = _read_metadata(snapshot_dir)
        config = dict(config)
        config["RUN_BASE_DIR"] = args.run_base_dir
        config["EVAL_ENV_DEVICE"] = str(args.env_device).strip().lower()
        config["EVAL_FORCE_FULL_VIEW"] = False

        if ckpt_seed is not None:
            seed = int(np.asarray(ckpt_seed).item() if hasattr(ckpt_seed, "item") else ckpt_seed)
        else:
            seed = int(metadata.get("seed", 0))
        run_key = _run_key_from_snapshot(snapshot_dir)
        if "env_step" in metadata:
            stored_step = int(metadata.get("env_step", raw_step))
        elif ckpt_update_step is not None:
            stored_step = int(
                np.asarray(ckpt_update_step).item()
                if hasattr(ckpt_update_step, "item")
                else ckpt_update_step
            )
        else:
            stored_step = int(metadata.get("update_step", raw_step))
        update_step = _normalize_env_step(stored_step, config, int(args.eval_every_env_steps))
        seed_key = _seed_key(snapshot_dir, seed)
        if args.eval_every_env_steps > 0 and (update_step % int(args.eval_every_env_steps)) != 0:
            continue
        seen = seen_eval_steps_by_seed.setdefault(seed_key, set())
        if update_step in seen:
            continue
        seen.add(update_step)

        should_video = False
        if _to_bool(args.log_video):
            if args.video_every_env_steps > 0:
                should_video = (update_step % int(args.video_every_env_steps)) == 0
            else:
                should_video = True

        if params_spec is None and params_ind is None:
            continue
        if params_spec is None:
            params_spec = params_ind
        if params_ind is None:
            params_ind = params_spec

        config_spec = dict(config)
        config_ind = dict(config)
        # PH2 training compatibility:
        # use run-config flags when present, and default to blocked-input enabled.
        config_spec["LEARNER_USE_BLOCKED_INPUT"] = bool(
            config.get("PH2_SPEC_USE_BLOCKED_INPUT", True)
        )
        config_ind["LEARNER_USE_BLOCKED_INPUT"] = bool(
            config.get("PH2_IND_USE_BLOCKED_INPUT", True)
        )

        policy_spec = PPOPolicy(params=params_spec, config=config_spec, stochastic=False)
        policy_ind = PPOPolicy(params=params_ind, config=config_ind, stochastic=False)
        local_rows = []

        eval_ctx = jax.disable_jit() if disable_jit else nullcontext()
        with eval_ctx:
            _evaluate_pair(
                pair_name="spec_spec",
                policy_a=policy_spec,
                policy_b=policy_spec,
                config=config,
                mode_tildes=mode_tildes,
                step=update_step,
                seed=seed,
                run_key=run_key,
                should_video=should_video,
                viz_max_steps=int(viz_steps),
                local_rows=local_rows,
            )
            _evaluate_pair(
                pair_name="spec_ind",
                policy_a=policy_spec,
                policy_b=policy_ind,
                config=config,
                mode_tildes=mode_tildes,
                step=update_step,
                seed=seed,
                run_key=run_key,
                should_video=should_video,
                viz_max_steps=int(viz_steps),
                local_rows=local_rows,
            )
            _evaluate_ind_ind(
                policy_ind=policy_ind,
                config=config,
                step=update_step,
                seed=seed,
                run_key=run_key,
                should_video=should_video,
                viz_max_steps=int(viz_steps),
                local_rows=local_rows,
            )
        eval_seed_dir = os.path.join(args.run_base_dir, "eval", f"run_{int(run_key)}")
        os.makedirs(eval_seed_dir, exist_ok=True)
        csv_path = _append_local_records(eval_seed_dir, local_rows)
        plot_dir = os.path.join(eval_seed_dir, "plots")
        csv_and_plot_dirs.add((str(csv_path), str(plot_dir)))
        print(f"[PH2-OFFLINE] metrics_csv={csv_path}")
        print(f"[PH2-OFFLINE] plots_dir={plot_dir}")

        processed += 1
        _clear_jax_runtime_state()

    for csv_path, plot_dir in sorted(csv_and_plot_dirs):
        try:
            _write_local_plots(csv_path, plot_dir)
        except Exception as e:
            print(f"[PH2-OFFLINE][WARN] plot generation failed csv={csv_path}: {type(e).__name__}: {e}")

    print(f"[PH2-OFFLINE] Done. processed={processed}")


if __name__ == "__main__":
    main()
