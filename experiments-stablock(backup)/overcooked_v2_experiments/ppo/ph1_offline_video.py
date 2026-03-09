import argparse
import csv
import gc
import json
import os
from typing import List, Optional

import jax
import numpy as np

from overcooked_v2_experiments.ppo.ph1_online_eval import (
    load_ph1_video_snapshot,
    _write_local_eval_plots,
    run_ph1_online_eval,
)


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


def _collect_snapshot_dirs(snapshot_root: str) -> List[str]:
    dirs = []
    if not os.path.isdir(snapshot_root):
        return dirs
    for name in os.listdir(snapshot_root):
        path = os.path.join(snapshot_root, name)
        if not (os.path.isdir(path) and _snapshot_step(path) is not None):
            continue
        ckpt_dir = os.path.join(path, "model_ckpt")
        done_marker = os.path.join(path, "_SNAPSHOT_DONE")
        has_tmp = False
        if not os.path.isdir(ckpt_dir):
            for sub in os.listdir(path):
                if sub.startswith("model_ckpt.orbax-checkpoint-tmp-") and os.path.isdir(os.path.join(path, sub)):
                    has_tmp = True
                    break
        if os.path.exists(done_marker) or os.path.isdir(ckpt_dir) or has_tmp:
            dirs.append(path)
    return sorted(dirs, key=lambda p: _snapshot_step(p) or 10**18)


def _collect_snapshot_dirs_multi(snapshot_root: str) -> List[str]:
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
    meta_path = os.path.join(snapshot_dir, "metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


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


def _clear_jax_runtime_state():
    try:
        jax.clear_caches()
    except Exception:
        pass
    gc.collect()


def _safe_float(v, default=float("nan")) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _extract_env_name_from_config(config: dict) -> str:
    try:
        env_cfg = config.get("env", {}) if isinstance(config, dict) else {}
        env_kwargs = env_cfg.get("ENV_KWARGS", {}) if isinstance(env_cfg, dict) else {}
        layout = env_kwargs.get("layout", "")
        if str(layout).strip():
            return str(layout).strip()
    except Exception:
        pass
    try:
        env_name = config.get("ENV_NAME", "")
        if str(env_name).strip():
            return str(env_name).strip()
    except Exception:
        pass
    return ""


def _write_eval_analysis_csv(run_base_dir: str, rows: List[dict]) -> str:
    out_path = os.path.join(str(run_base_dir), "eval", "offline_eval_analysis.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = [
        "run",
        "env",
        "mode",
        "env_step",
        "omega",
        "sigma",
        "total_reward",
        "distance",
        "distance_with_recent",
        "distance_with_random",
        "pred_accuracy",
    ]

    dedup = {}

    if os.path.exists(out_path):
        try:
            with open(out_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        run_name = str(row.get("run", "")).strip()
                        if not run_name:
                            old_seed = int(float(row.get("seed", "0")))
                            run_name = f"seed={old_seed}"
                        env_name = str(row.get("env", "")).strip()
                        mode_name = str(row.get("mode", "avg")).strip() or "avg"
                        step_i = int(float(row.get("env_step", row.get("step", "0"))))
                    except Exception:
                        continue
                    dedup[(run_name, env_name, mode_name, step_i)] = {
                        "run": run_name,
                        "env": env_name,
                        "mode": mode_name,
                        "env_step": step_i,
                        "omega": _safe_float(row.get("omega", float("nan"))),
                        "sigma": _safe_float(row.get("sigma", float("nan"))),
                        "total_reward": round(
                            _safe_float(
                                row.get("total_reward", row.get("reward_total", float("nan"))),
                                float("nan"),
                            ),
                            2,
                        ),
                        "distance": round(
                            _safe_float(
                                row.get("distance", float("nan")),
                                float("nan"),
                            ),
                            4,
                        ),
                        "distance_with_recent": round(
                            _safe_float(
                                row.get("distance_with_recent", float("nan")),
                                float("nan"),
                            ),
                            4,
                        ),
                        "distance_with_random": round(
                            _safe_float(
                                row.get("distance_with_random", float("nan")),
                                float("nan"),
                            ),
                            4,
                        ),
                        "pred_accuracy": round(_safe_float(row.get("pred_accuracy", float("nan"))), 2),
                    }
        except Exception:
            pass

    for row in rows:
        try:
            run_name = str(row["run"])
            env_name = str(row.get("env", ""))
            mode_name = str(row.get("mode", "avg"))
            step_i = int(row["env_step"])
        except Exception:
            continue
        dedup[(run_name, env_name, mode_name, step_i)] = {
            "run": run_name,
            "env": env_name,
            "mode": mode_name,
            "env_step": step_i,
            "omega": _safe_float(row.get("omega", float("nan"))),
            "sigma": _safe_float(row.get("sigma", float("nan"))),
            "total_reward": round(
                _safe_float(
                    row.get("total_reward", row.get("reward_total", float("nan"))),
                    float("nan"),
                ),
                2,
            ),
            "distance": round(
                _safe_float(row.get("distance", float("nan"))),
                4,
            ),
            "distance_with_recent": round(
                _safe_float(row.get("distance_with_recent", float("nan"))),
                4,
            ),
            "distance_with_random": round(
                _safe_float(row.get("distance_with_random", float("nan"))),
                4,
            ),
            "pred_accuracy": round(_safe_float(row.get("pred_accuracy", float("nan"))), 2),
        }

    mode_order = {"normal": 0, "recent": 1, "random": 2, "avg": 3}

    def _run_sort_key(run_name: str):
        s = str(run_name).strip()
        if s.startswith("run_"):
            try:
                return int(s.split("_", 1)[1])
            except Exception:
                return 10**9
        if s.startswith("seed="):
            try:
                return int(s.split("=", 1)[1])
            except Exception:
                return 10**9
        return 10**9

    ordered = sorted(
        dedup.values(),
        key=lambda x: (
            _run_sort_key(x.get("run", "")),
            str(x.get("run", "")),
            str(x.get("env", "")),
            mode_order.get(str(x.get("mode", "")), 99),
            int(x.get("env_step", 0)),
        ),
    )
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ordered)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Run PH1 eval/video offline from saved snapshots.")
    parser.add_argument("--run-base-dir", type=str, required=True, help="Run base dir containing eval/{step}")
    parser.add_argument("--eval-every-env-steps", type=int, default=200000)
    parser.add_argument("--video-every-env-steps", type=int, default=1000000)
    parser.add_argument("--viz-max-steps", type=int, default=400)
    parser.add_argument("--log-video", type=str, default="False", help="Enable offline video rendering (True/False)")
    parser.add_argument("--disable-jit", type=str, default="True", help="Disable JAX JIT during offline eval (True/False)")
    parser.add_argument("--env-device", type=str, default="cpu", help="Env interaction device: cpu|gpu")
    parser.add_argument(
        "--eval-analysis",
        type=str,
        default="False",
        help="Write aggregated eval csv under eval/offline_eval_analysis.csv (True/False)",
    )
    args = parser.parse_args()
    args.run_base_dir = os.path.abspath(str(args.run_base_dir))
    viz_steps = 400
    if int(args.viz_max_steps) != viz_steps:
        print(f"[PH1-OFFLINE] override viz_max_steps {int(args.viz_max_steps)} -> {viz_steps}")

    snapshot_root = os.path.join(args.run_base_dir, "eval")
    snapshot_dirs = _collect_snapshot_dirs_multi(snapshot_root)
    if len(snapshot_dirs) == 0:
        print(f"[PH1-OFFLINE] No snapshots found: {snapshot_root}")
        return

    seen_eval_steps_by_seed = {}

    def _seed_key(snapshot_dir: str, metadata: dict) -> int:
        if "seed" in metadata:
            try:
                return int(metadata["seed"])
            except Exception:
                pass
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

    eval_analysis_enabled = _to_bool(args.eval_analysis)
    flush_every_snapshots = 10
    processed = 0
    csv_and_plot_dirs = set()
    analysis_rows: List[dict] = []
    analysis_csv_path_latest: Optional[str] = None
    analysis_dirty = False
    for snapshot_dir in snapshot_dirs:
        raw_step = _snapshot_step(snapshot_dir)
        if raw_step is None:
            continue
        try:
            params, config, mode_tildes = load_ph1_video_snapshot(snapshot_dir)
            metadata = _read_metadata(snapshot_dir)
            config = dict(config)
            seed_key = _seed_key(snapshot_dir, metadata)

            stored_step = int(metadata.get("env_step", metadata.get("update_step", raw_step)))
            env_step = _normalize_env_step(stored_step, config, int(args.eval_every_env_steps))
            if args.eval_every_env_steps > 0 and (env_step % int(args.eval_every_env_steps)) != 0:
                continue
            seen = seen_eval_steps_by_seed.setdefault(seed_key, set())
            if env_step in seen:
                continue
            seen.add(env_step)

            log_video = _to_bool(args.log_video)
            should_video = False
            if log_video:
                if args.video_every_env_steps > 0:
                    should_video = (env_step % int(args.video_every_env_steps)) == 0
                else:
                    should_video = True

            seed = int(metadata.get("seed", 0))
            run_key = _run_key_from_snapshot(snapshot_dir)

            # User requirement:
            # - normal/recent/random each 1 episode
            # - video length 400 steps
            config["PH1_ENABLED"] = bool(config.get("PH1_ENABLED", True))
            config["PH1_EVAL_ENABLED"] = True
            config["PH1_EVAL_NUM_SEEDS"] = 1
            config["PH1_EVAL_VIZ_EPISODES"] = 1 if should_video else 0
            config["PH1_EVAL_VIZ_MAX_STEPS"] = int(viz_steps)
            config["PH1_EVAL_ROLLOUT_MAX_STEPS"] = int(viz_steps)
            config["PH1_EVAL_LOG_VIDEO"] = bool(should_video)
            config["PH1_EVAL_WANDB_LOG"] = False
            config["PH1_EVAL_DISABLE_JIT"] = _to_bool(args.disable_jit)
            config["PH1_EVAL_FORCE_FULL_VIEW"] = False
            config["PH1_EVAL_WRITE_PLOTS"] = False
            config["PH1_EVAL_WRITE_LOCAL_CSV"] = not eval_analysis_enabled
            config["EVAL_ENV_DEVICE"] = str(args.env_device).strip().lower()

            print(
                f"[PH1-OFFLINE] raw_step={raw_step} env_step={env_step} eval=1 video={int(should_video)} log_video={int(log_video)} snapshot={snapshot_dir}"
            )
            old_run_key = os.environ.get("PH1_EVAL_RUN_KEY")
            os.environ["PH1_EVAL_RUN_KEY"] = str(int(run_key))
            try:
                out = run_ph1_online_eval(
                    params=params,
                    config=config,
                    update_step=env_step,
                    recent_tilde_batch=None,
                    seed=seed,
                    eval_log_video_override=bool(should_video),
                    forced_mode_tildes=mode_tildes,
                )
                if isinstance(out, dict):
                    csv_path = out.get("csv_path")
                    plot_dir = out.get("plot_dir")
                    if csv_path:
                        print(f"[PH1-OFFLINE] metrics_csv={csv_path}")
                    if plot_dir:
                        print(f"[PH1-OFFLINE] plots_dir={plot_dir}")
                    if csv_path and plot_dir:
                        csv_and_plot_dirs.add((str(csv_path), str(plot_dir)))
                    if eval_analysis_enabled:
                        mode_metric_means = dict(out.get("mode_metric_means", {}))
                        env_name = _extract_env_name_from_config(config)
                        for mode in ("normal", "recent", "random"):
                            mode_metrics = mode_metric_means.get(mode, {})
                            analysis_rows.append(
                                {
                                    "run": f"run_{int(run_key)}",
                                    "env": env_name,
                                    "mode": mode,
                                    "env_step": int(env_step),
                                    "omega": _safe_float(config.get("PH1_OMEGA", float("nan"))),
                                    "sigma": _safe_float(config.get("PH1_SIGMA", float("nan"))),
                                    "total_reward": _safe_float(mode_metrics.get("reward_total", np.nan)),
                                    "distance": _safe_float(mode_metrics.get("distance", np.nan)),
                                    "distance_with_recent": _safe_float(
                                        mode_metrics.get("distance_with_recent", np.nan)
                                    ),
                                    "distance_with_random": _safe_float(
                                        mode_metrics.get("distance_with_random", np.nan)
                                    ),
                                    "pred_accuracy": _safe_float(mode_metrics.get("pred_accuracy", np.nan)),
                                }
                            )
                        analysis_dirty = True
            finally:
                if old_run_key is None:
                    os.environ.pop("PH1_EVAL_RUN_KEY", None)
                else:
                    os.environ["PH1_EVAL_RUN_KEY"] = old_run_key
            processed += 1
            if eval_analysis_enabled and analysis_dirty and (processed % flush_every_snapshots) == 0:
                analysis_csv_path_latest = _write_eval_analysis_csv(args.run_base_dir, analysis_rows)
                analysis_dirty = False
        except Exception as e:
            print(f"[PH1-OFFLINE][WARN] skip snapshot={snapshot_dir} reason={type(e).__name__}: {e}")
            continue
        finally:
            _clear_jax_runtime_state()

    for csv_path, plot_dir in sorted(csv_and_plot_dirs):
        try:
            _write_local_eval_plots(csv_path, plot_dir)
        except Exception as e:
            print(f"[PH1-OFFLINE][WARN] plot generation failed csv={csv_path}: {type(e).__name__}: {e}")

    if eval_analysis_enabled:
        if analysis_dirty or analysis_csv_path_latest is None:
            analysis_csv_path_latest = _write_eval_analysis_csv(args.run_base_dir, analysis_rows)
            analysis_dirty = False
        print(f"[PH1-OFFLINE] analysis_csv={analysis_csv_path_latest}")

    print(f"[PH1-OFFLINE] Done. processed={processed}")


if __name__ == "__main__":
    main()
