#!/usr/bin/env python3
import argparse
import copy
import math
import os
import secrets
import shutil
import json
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_MODES = ("normal", "recent", "random")
ComboKey = Tuple[float, float, float]

# ---------------------------------------------------------------------------
# Preset defaults (edit here if you want to run without long CLI arguments)
# ---------------------------------------------------------------------------
# Select task first.
PRESET_TASK = "summary"  # summary | distance-distribution-eval

# Shared defaults.
PRESET_DISTANCE_METRIC = "distance"

# --- Summary task presets ----------------------------------------------------
PRESET_START_RUN = "20260223-210235_tbxw4cln_counter_circuit_e3t_ph1"
PRESET_END_RUN = "20260223-234643_aoeewekf_counter_circuit_e3t_ph1"
PRESET_PHASE = "ph1"  # ph1 | ph2 | auto
PRESET_ENV = "counter_circuit"  # None for all envs
PRESET_METRIC = "total_reward"  # currently total_reward is recommended
PRESET_COMBOS_PER_PLOT = 4
PRESET_XTICK_STEP = 1_000_000
PRESET_SMOOTH_WINDOW = 2
PRESET_LEGEND_FONT_SIZE = 14
PRESET_LEGEND_TITLE_FONT_SIZE = 15
eval_env = 'counter_circuit'

# --- Distance-distribution-eval task presets --------------------------------
# `PRESET_DISTANCE_EVAL_RUN` can be a run folder name under runs-dir
# or an absolute path to a run folder.
PRESET_DISTANCE_EVAL_RUN = "20260216-122942_jp12tm1r_grounded_coord_simple_e3t_ph1"
PRESET_DISTANCE_EVAL_MODES = ("recent", "random")
PRESET_DISTANCE_EVAL_MAX_STEPS = 400
PRESET_DISTANCE_EVAL_SEED = 42
PRESET_DISTANCE_EVAL_ENV_DEVICE = "cpu"
PRESET_DISTANCE_EVAL_LOG_VIDEO = True


def _resolve_runs_in_range(runs_dir: str, start_run: str, end_run: str) -> List[str]:
    all_runs = sorted(
        [
            d
            for d in os.listdir(runs_dir)
            if os.path.isdir(os.path.join(runs_dir, d))
        ]
    )
    if start_run not in all_runs:
        raise ValueError(f"start-run not found: {start_run}")
    if end_run not in all_runs:
        raise ValueError(f"end-run not found: {end_run}")
    s = all_runs.index(start_run)
    e = all_runs.index(end_run)
    if s > e:
        s, e = e, s
    return all_runs[s : e + 1]


def _infer_phase_from_name(run_name: str) -> str:
    if "_ph2" in run_name:
        return "ph2"
    return "ph1"


def _fmt_combo(omega: float, sigma: float, beta_end: float) -> str:
    beta_label = f"{beta_end:g}" if np.isfinite(beta_end) else "nan"
    return f"omega:{omega:g},sigma:{sigma:g},beta_end:{beta_label}"


def _split_combos(
    combos: List[ComboKey],
    per_plot: int,
) -> List[List[ComboKey]]:
    n = len(combos)
    if n == 0:
        return []
    return [combos[i : i + per_plot] for i in range(0, n, per_plot)]


def _format_step_label(v: float) -> str:
    x = float(v)
    ax = abs(x)
    if ax >= 1_000_000:
        y = x / 1_000_000.0
        return f"{int(y)}m" if abs(y - round(y)) < 1e-8 else f"{y:.1f}m"
    if ax >= 1_000:
        y = x / 1_000.0
        return f"{int(y)}k" if abs(y - round(y)) < 1e-8 else f"{y:.1f}k"
    return f"{int(x)}"


def _nice_step(raw_step: float, min_step: int) -> float:
    if raw_step <= 0:
        return float(max(1, min_step))
    p = 10 ** math.floor(math.log10(raw_step))
    for m in (1, 2, 5, 10):
        s = m * p
        if s >= raw_step:
            return float(max(s, min_step))
    return float(max(10 * p, min_step))


def _build_sparse_ticks(x_min: int, x_max: int, min_step: int, max_ticks: int = 5) -> np.ndarray:
    if x_max <= x_min:
        return np.array([x_min], dtype=float)
    raw_step = (x_max - x_min) / max(1, (max_ticks - 1))
    step = _nice_step(raw_step, min_step=min_step)
    start = math.floor(x_min / step) * step
    ticks = np.arange(start, x_max + step * 0.51, step, dtype=float)
    ticks = ticks[(ticks >= x_min) & (ticks <= x_max)]
    if ticks.size == 0:
        ticks = np.array([x_min, x_max], dtype=float)
    if ticks[0] != x_min:
        ticks = np.insert(ticks, 0, float(x_min))
    if ticks[-1] != x_max:
        ticks = np.append(ticks, float(x_max))
    if ticks.size > max_ticks:
        idx = np.linspace(0, ticks.size - 1, num=max_ticks)
        ticks = ticks[np.unique(np.round(idx).astype(int))]
        ticks[0] = float(x_min)
        ticks[-1] = float(x_max)
    return ticks


def _aggregate_mode(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    agg = (
        df.groupby(["omega", "sigma", "beta_end", "env_step"], as_index=False, dropna=False)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    return agg


def _to_numeric_or_nan(v) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _combo_mask(df: pd.DataFrame, omega: float, sigma: float, beta_end: float) -> pd.Series:
    mask = np.isclose(df["omega"].to_numpy(dtype=float), float(omega), equal_nan=False)
    mask &= np.isclose(df["sigma"].to_numpy(dtype=float), float(sigma), equal_nan=False)
    beta_col = df["beta_end"].to_numpy(dtype=float)
    if np.isfinite(beta_end):
        mask &= np.isclose(beta_col, float(beta_end), equal_nan=False)
    else:
        mask &= np.isnan(beta_col)
    return pd.Series(mask, index=df.index)


def _iter_run_ckpt_dirs(run_dir: str) -> List[str]:
    out = []
    if not os.path.isdir(run_dir):
        return out
    for name in sorted(os.listdir(run_dir)):
        seed_dir = os.path.join(run_dir, name)
        if not (os.path.isdir(seed_dir) and (name.startswith("run_") or name.startswith("seed_"))):
            continue
        for ckpt_name in ("ckpt_final", "ckpt_0"):
            ckpt_dir = os.path.join(seed_dir, ckpt_name)
            if os.path.isdir(ckpt_dir):
                out.append(ckpt_dir)
    return out


def _read_beta_end_from_orbax_ckpt(run_dir: str) -> float:
    # Preferred source: restore checkpoint config directly.
    try:
        import orbax.checkpoint as ocp
    except Exception:
        return float("nan")

    for ckpt_dir in _iter_run_ckpt_dirs(run_dir):
        try:
            checkpointer = ocp.PyTreeCheckpointer()
            ckpt = checkpointer.restore(ckpt_dir, item=None)
            if not isinstance(ckpt, dict):
                continue
            cfg = ckpt.get("config", {})
            if not isinstance(cfg, dict):
                continue
            val = cfg.get("PH1_BETA_END", cfg.get("PH1_BETA", float("nan")))
            beta_end = _to_numeric_or_nan(val)
            if np.isfinite(beta_end):
                return float(beta_end)
        except Exception:
            continue
    return float("nan")


def _read_beta_end_from_snapshot_loader(run_dir: str) -> float:
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        from overcooked_v2_experiments.ppo.ph1_online_eval import load_ph1_video_snapshot
    except Exception:
        return float("nan")
    for ckpt_dir in _iter_run_ckpt_dirs(run_dir):
        try:
            _, config, _ = load_ph1_video_snapshot(ckpt_dir)
            cfg = dict(config) if isinstance(config, dict) else {}
            val = cfg.get("PH1_BETA_END", cfg.get("PH1_BETA", float("nan")))
            beta_end = _to_numeric_or_nan(val)
            if np.isfinite(beta_end):
                return float(beta_end)
        except Exception:
            continue
    return float("nan")


def _read_beta_end_from_strings(run_dir: str) -> float:
    # Last-resort fallback for environments without orbax restore path.
    for ckpt_dir in _iter_run_ckpt_dirs(run_dir):
        strings_path = os.path.join(ckpt_dir, "_strings.json")
        if not os.path.exists(strings_path):
            continue
        try:
            with open(strings_path, "r", encoding="utf-8") as f:
                strings = json.load(f)
            if not isinstance(strings, dict):
                continue
            val = strings.get("config.PH1_BETA_END", None)
            if val is None:
                val = strings.get("config.PH1_BETA", None)
            beta_end = _to_numeric_or_nan(val)
            if np.isfinite(beta_end):
                return float(beta_end)
        except Exception:
            continue
    return float("nan")


def _resolve_run_beta_end(runs_dir: str, run_name: str) -> float:
    run_dir = os.path.join(runs_dir, run_name)
    beta_end = _read_beta_end_from_orbax_ckpt(run_dir)
    if np.isfinite(beta_end):
        return float(beta_end)
    beta_end = _read_beta_end_from_snapshot_loader(run_dir)
    if np.isfinite(beta_end):
        return float(beta_end)
    return float(_read_beta_end_from_strings(run_dir))


def _to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off", ""}:
        return False
    raise ValueError(f"Invalid boolean value: {v}")


def _smooth_series(y: np.ndarray, window: int) -> np.ndarray:
    w = max(1, int(window))
    if w <= 1:
        return y
    return pd.Series(y).rolling(window=w, min_periods=1, center=True).mean().to_numpy()


def _plot_env_phase(
    df: pd.DataFrame,
    env_name: str,
    phase_name: str,
    metric: str,
    distance_metric: str,
    combos: List[ComboKey],
    part_idx: int,
    part_total: int,
    xtick_step: int,
    smooth_window: int,
    legend_font_size: int,
    legend_title_font_size: int,
    modes: Tuple[str, ...],
    output_path: str,
) -> None:
    fig, axes = plt.subplots(
        2,
        len(modes),
        figsize=(30, 14),
        sharex="col",
        sharey="row",
    )
    if len(modes) == 1:
        axes = np.array(axes).reshape(2, 1)

    n_combo = max(1, len(combos))
    cmap = plt.colormaps.get_cmap("tab20").resampled(n_combo)
    combo_to_color = {combo: cmap(i) for i, combo in enumerate(combos)}

    legend_handles = []
    legend_labels = []
    all_max_reward_y = []
    all_max_distance_y = []
    normal_df = df[df["mode"] == "normal"].copy()
    baseline_recent_map: Dict[ComboKey, float] = {}
    baseline_random_map: Dict[ComboKey, float] = {}
    if "distance_with_recent" in normal_df.columns:
        grp_recent = (
            normal_df.groupby(["omega", "sigma", "beta_end"], as_index=False, dropna=False)["distance_with_recent"]
            .mean()
            .dropna(subset=["distance_with_recent"])
        )
        for _, r in grp_recent.iterrows():
            baseline_recent_map[
                (
                    float(r["omega"]),
                    float(r["sigma"]),
                    _to_numeric_or_nan(r.get("beta_end", float("nan"))),
                )
            ] = float(r["distance_with_recent"])
    if "distance_with_random" in normal_df.columns:
        grp_random = (
            normal_df.groupby(["omega", "sigma", "beta_end"], as_index=False, dropna=False)["distance_with_random"]
            .mean()
            .dropna(subset=["distance_with_random"])
        )
        for _, r in grp_random.iterrows():
            baseline_random_map[
                (
                    float(r["omega"]),
                    float(r["sigma"]),
                    _to_numeric_or_nan(r.get("beta_end", float("nan"))),
                )
            ] = float(r["distance_with_random"])

    x_min = int(df["env_step"].min())
    x_max = int(df["env_step"].max())
    ticks = _build_sparse_ticks(
        x_min=x_min,
        x_max=x_max,
        min_step=max(1, int(xtick_step)),
        max_ticks=5,
    )

    for col_idx, mode in enumerate(modes):
        reward_ax = axes[0, col_idx]
        distance_ax = axes[1, col_idx]
        mode_df = df[df["mode"] == mode].copy()
        if mode_df.empty:
            reward_ax.set_title(f"{mode} (no data)", fontsize=16)
            reward_ax.grid(True, alpha=0.25)
            distance_ax.grid(True, alpha=0.25)
            continue

        reward_agg = _aggregate_mode(mode_df, metric)
        distance_agg = _aggregate_mode(mode_df, distance_metric)

        for omega, sigma, beta_end in combos:
            label = _fmt_combo(omega, sigma, beta_end)
            reward_cdf = reward_agg[_combo_mask(reward_agg, omega, sigma, beta_end)].sort_values("env_step")
            distance_cdf = distance_agg[_combo_mask(distance_agg, omega, sigma, beta_end)].sort_values("env_step")

            line = None
            if not reward_cdf.empty:
                reward_y_mean = reward_cdf["mean"].to_numpy()
                reward_y_smooth = _smooth_series(reward_y_mean, smooth_window)
                line, = reward_ax.plot(
                    reward_cdf["env_step"].to_numpy(),
                    reward_y_smooth,
                    color=combo_to_color[(omega, sigma, beta_end)],
                    linewidth=2.2,
                    marker="o",
                    markersize=3.2,
                    label=label,
                )
                all_max_reward_y.append(float(np.nanmax(reward_y_smooth)))

            if not distance_cdf.empty:
                distance_y_mean = distance_cdf["mean"].to_numpy()
                distance_y_smooth = _smooth_series(distance_y_mean, smooth_window)
                dist_line, = distance_ax.plot(
                    distance_cdf["env_step"].to_numpy(),
                    distance_y_smooth,
                    color=combo_to_color[(omega, sigma, beta_end)],
                    linewidth=2.2,
                    marker="o",
                    markersize=3.2,
                    label=label,
                )
                all_max_distance_y.append(float(np.nanmax(distance_y_smooth)))
                if line is None:
                    line = dist_line

            # Add constant baselines from normal-mode cross-target distances.
            combo_key: ComboKey = (float(omega), float(sigma), float(beta_end))
            if mode == "recent":
                baseline_recent = baseline_recent_map.get(combo_key, np.nan)
                if np.isfinite(baseline_recent):
                    distance_ax.axhline(
                        y=float(baseline_recent),
                        color=combo_to_color[(omega, sigma, beta_end)],
                        linewidth=1.6,
                        linestyle="--",
                        alpha=0.85,
                    )
                    all_max_distance_y.append(float(baseline_recent))
            elif mode == "random":
                baseline_random = baseline_random_map.get(combo_key, np.nan)
                if np.isfinite(baseline_random):
                    distance_ax.axhline(
                        y=float(baseline_random),
                        color=combo_to_color[(omega, sigma, beta_end)],
                        linewidth=1.6,
                        linestyle="--",
                        alpha=0.85,
                    )
                    all_max_distance_y.append(float(baseline_random))

            if line is None:
                continue
            if label not in legend_labels:
                legend_labels.append(label)
                legend_handles.append(line)

        reward_ax.set_title(f"{mode}", fontsize=18, pad=12)
        reward_ax.grid(True, alpha=0.25)
        reward_ax.tick_params(axis="both", labelsize=11)
        reward_ax.set_xticks(ticks)
        reward_ax.set_xticklabels([_format_step_label(t) for t in ticks], fontsize=11)
        reward_ax.set_xlim(max(0, x_min), x_max)

        distance_ax.set_xlabel("env_step", fontsize=14)
        distance_ax.grid(True, alpha=0.25)
        distance_ax.tick_params(axis="both", labelsize=11)
        distance_ax.set_xticks(ticks)
        distance_ax.set_xticklabels([_format_step_label(t) for t in ticks], fontsize=11)
        distance_ax.set_xlim(max(0, x_min), x_max)

    axes[0, 0].set_ylabel(metric, fontsize=14)
    axes[1, 0].set_ylabel(distance_metric, fontsize=14)
    reward_ymax = max(1.0, max(all_max_reward_y) if all_max_reward_y else 1.0) * 1.05
    distance_ymax = max(1.0, max(all_max_distance_y) if all_max_distance_y else 1.0) * 1.05
    for ax in axes[0]:
        ax.set_ylim(bottom=0.0, top=reward_ymax)
    for ax in axes[1]:
        ax.set_ylim(bottom=0.0, top=distance_ymax)
    fig.suptitle(
        f"{phase_name.upper()} reward+distance ({env_name}) - grouped by omega/sigma/beta_end "
        f"[part {part_idx + 1}/{part_total}, smooth={max(1, int(smooth_window))}]",
        fontsize=20,
        y=0.98,
    )

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="center left",
            bbox_to_anchor=(0.80, 0.5),
            fontsize=max(8, int(legend_font_size)),
            title="omega,sigma,beta_end",
            title_fontsize=max(9, int(legend_title_font_size)),
            frameon=False,
            markerscale=1.4,
            handlelength=2.2,
        )

    fig.subplots_adjust(right=0.78)
    fig.tight_layout(rect=[0.02, 0.03, 0.78, 0.95])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _resolve_single_run_dir(runs_dir: str, run_dir: str) -> str:
    run_dir = str(run_dir).strip()
    if not run_dir:
        raise ValueError("--run-dir is required in distance-distribution-eval mode.")
    if os.path.isabs(run_dir):
        if not os.path.isdir(run_dir):
            raise ValueError(f"run-dir not found: {run_dir}")
        return os.path.abspath(run_dir)
    candidate = os.path.join(runs_dir, run_dir)
    if not os.path.isdir(candidate):
        raise ValueError(f"run-dir not found under runs-dir: {candidate}")
    return os.path.abspath(candidate)


def _snapshot_run_key(snapshot_dir: str) -> int:
    for part in os.path.normpath(snapshot_dir).split(os.sep):
        if part.startswith("run_") or part.startswith("seed_"):
            try:
                return int(part.split("_", 1)[1])
            except Exception:
                return 0
    return 0


def _collect_final_snapshot_dirs(run_base_dir: str) -> List[str]:
    eval_dir = os.path.join(run_base_dir, "eval")
    if not os.path.isdir(eval_dir):
        return []
    out = []
    for name in sorted(os.listdir(eval_dir)):
        root = os.path.join(eval_dir, name)
        if not os.path.isdir(root):
            continue
        if name.startswith("run_") or name.startswith("seed_"):
            ckpt_final = os.path.join(root, "ckpt_final")
            if os.path.isdir(ckpt_final):
                out.append(ckpt_final)
    legacy_final = os.path.join(eval_dir, "ckpt_final")
    if os.path.isdir(legacy_final):
        out.append(legacy_final)
    return sorted(out, key=lambda p: (_snapshot_run_key(p), p))


def _plot_episode_reward_distance(
    *,
    mode_curves: Dict[str, Dict[str, np.ndarray]],
    modes: Tuple[str, ...],
    run_name: str,
    run_key: int,
    ckpt_name: str,
    episode_seed: int,
    output_path: str,
    distance_metric: str,
) -> None:
    fig, axes = plt.subplots(1, len(modes), figsize=(8 * max(1, len(modes)), 6))
    if len(modes) == 1:
        axes = [axes]

    for ax, mode in zip(axes, modes):
        curve = mode_curves.get(mode)
        if curve is None:
            ax.set_title(f"{mode} (no data)", fontsize=15)
            ax.grid(True, alpha=0.25)
            ax.set_xlabel("episode_step", fontsize=12)
            continue

        x = curve["episode_step"]
        y_reward = curve["cumulative_reward"]
        y_dist = curve["distance"]

        l_reward, = ax.plot(
            x,
            y_reward,
            color="#1f77b4",
            linewidth=2.2,
            label="cumulative_reward",
        )
        ax.set_xlabel("episode_step", fontsize=12)
        ax.set_ylabel("cumulative_reward", fontsize=12, color="#1f77b4")
        ax.tick_params(axis="y", labelcolor="#1f77b4")
        ax.grid(True, alpha=0.25)

        dist_ax = ax.twinx()
        l_dist, = dist_ax.plot(
            x,
            y_dist,
            color="#d62728",
            linewidth=2.0,
            label=distance_metric,
        )
        dist_ax.set_ylabel(distance_metric, fontsize=12, color="#d62728")
        dist_ax.tick_params(axis="y", labelcolor="#d62728")

        lines = [l_reward, l_dist]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc="best", fontsize=10)
        ax.set_title(
            f"{mode} | reward={curve['total_reward']:.2f}, dist={curve['distance_mean']:.4f}",
            fontsize=13,
        )

    fig.suptitle(
        f"{run_name} | run_{run_key} | {ckpt_name} | seed={episode_seed}",
        fontsize=15,
        y=0.98,
    )
    fig.tight_layout(rect=[0.02, 0.03, 0.98, 0.94])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _run_distance_distribution_eval(args: argparse.Namespace) -> None:
    # Lazy imports: keep summary mode lightweight and dependency-stable.
    try:
        import jax
        from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2
        from overcooked_v2_experiments.eval.policy import PolicyPairing
        from overcooked_v2_experiments.eval.rollout import get_rollout
        from overcooked_v2_experiments.ppo.ph1_online_eval import (
            _get_agent_pos_channels_from_env,
            _render_video_from_rollout,
            load_ph1_video_snapshot,
        )
        from overcooked_v2_experiments.ppo.policy import PPOPolicy
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "distance-distribution-eval mode requires JAX/JAXMARL runtime "
            "(e.g., jax, jaxlib, jaxmarl)."
        ) from e

    run_base_dir = _resolve_single_run_dir(args.runs_dir, args.run_dir)
    snapshot_dirs = _collect_final_snapshot_dirs(run_base_dir)
    if not snapshot_dirs:
        raise RuntimeError(f"No ckpt_final snapshots found under: {run_base_dir}/eval")

    distance_eval_modes = tuple(str(m).strip() for m in args.distance_eval_modes if str(m).strip())
    if len(distance_eval_modes) == 0:
        raise ValueError("--distance-eval-modes must contain at least one mode.")
    distance_eval_log_video = _to_bool(args.distance_eval_log_video)

    default_plots_root = os.path.abspath(os.path.join(args.runs_dir, "plots"))
    requested_output_dir = os.path.abspath(args.output_dir) if args.output_dir else ""
    if (
        (not requested_output_dir)
        or (requested_output_dir == default_plots_root)
        or requested_output_dir.startswith(default_plots_root + os.sep)
    ):
        output_dir = os.path.join(run_base_dir, "eval")
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    saved = []
    saved_videos = []
    run_name = os.path.basename(run_base_dir.rstrip(os.sep))
    for snapshot_dir in snapshot_dirs:
        try:
            params, config, mode_tildes = load_ph1_video_snapshot(snapshot_dir)
        except Exception as e:
            print(f"[WARN] skip snapshot load failed: {snapshot_dir} ({e})")
            continue

        config = dict(config) if isinstance(config, dict) else {}
        env_cfg = copy.deepcopy(config.get("env", {})) if isinstance(config.get("env", {}), dict) else {}
        env_kwargs = copy.deepcopy(env_cfg.get("ENV_KWARGS", {})) if isinstance(env_cfg.get("ENV_KWARGS", {}), dict) else {}
        layout = env_kwargs.pop("layout", None)
        if not layout:
            print(f"[WARN] skip snapshot without env layout: {snapshot_dir}")
            continue

        env = OvercookedV2(layout=layout, **env_kwargs)
        agent0_pos_channel, agent1_pos_channel = _get_agent_pos_channels_from_env(env)
        stochastic = bool(config.get("PH1_EVAL_STOCHASTIC", False))
        force_full_view = bool(config.get("PH1_EVAL_FORCE_FULL_VIEW", False))
        policy = PPOPolicy(params=params, config=config, stochastic=stochastic)
        pairing = PolicyPairing.from_single_policy(policy, env.num_agents)
        algorithm = str(config.get("ALG_NAME", "PH1-E3T"))
        if "PH1" not in algorithm:
            algorithm = f"PH1-{algorithm}"
        ph1_omega = float(config.get("PH1_OMEGA", 1.0))
        ph1_sigma = float(config.get("PH1_SIGMA", 1.0))
        ckpt_name = os.path.basename(snapshot_dir)

        run_key = _snapshot_run_key(snapshot_dir)
        episode_seed = int(args.distance_eval_seed) + int(run_key) * 1009
        rollout_key = jax.random.PRNGKey(episode_seed)
        random_tilde = None
        try:
            tilde_key = jax.random.PRNGKey(episode_seed + 17777)
            _, rand_state = env.reset(tilde_key)
            random_tilde = np.array(env.get_obs_default(rand_state)[0]).astype(np.float32)
        except Exception as e:
            print(f"[WARN] {snapshot_dir}: failed to build random tilde from reset ({e})")

        mode_curves: Dict[str, Dict[str, np.ndarray]] = {}
        for mode in distance_eval_modes:
            mode_lc = mode.lower()
            if mode_lc == "normal":
                tilde_state = None
            elif mode_lc == "recent":
                tilde_state = mode_tildes.get("recent", None)
            elif mode_lc == "random":
                tilde_state = random_tilde
            else:
                tilde_state = mode_tildes.get(mode_lc, None)

            if mode_lc in ("recent", "random") and tilde_state is None:
                print(f"[WARN] {snapshot_dir}: {mode_lc} tilde is missing, skip mode.")
                continue

            rollout = get_rollout(
                pairing,
                env,
                rollout_key,
                algorithm=algorithm,
                ph1_forced_tilde_state=tilde_state,
                ph1_omega=ph1_omega,
                ph1_sigma=ph1_sigma,
                max_rollout_steps=max(1, int(args.distance_eval_max_steps)),
                env_device=str(args.distance_eval_env_device).strip().lower(),
            )
            reward_seq = np.asarray(rollout.step_reward_seq, dtype=float)
            if reward_seq.size == 0:
                print(f"[WARN] {snapshot_dir}: empty rollout for mode={mode_lc}, skip mode.")
                continue
            cumulative_reward = np.asarray(rollout.cumulative_reward_seq, dtype=float)
            distance_seq = np.asarray(rollout.ph1_distance_seq, dtype=float)
            episode_step = np.arange(1, reward_seq.shape[0] + 1, dtype=int)

            mode_curves[mode_lc] = {
                "episode_step": episode_step,
                "cumulative_reward": cumulative_reward,
                "distance": distance_seq,
                "total_reward": float(rollout.total_reward),
                "distance_mean": float(rollout.ph1_distance_mean),
            }

            if distance_eval_log_video:
                video_path, _, render_err = _render_video_from_rollout(
                    rollout,
                    env_kwargs=env_kwargs,
                    max_steps=max(1, int(args.distance_eval_max_steps)),
                    force_full_view=force_full_view,
                )
                if video_path is None:
                    print(
                        f"[WARN] {snapshot_dir}: video render failed for mode={mode_lc} "
                        f"({render_err})"
                    )
                else:
                    try:
                        p0 = (-1, -1)
                        p1 = (-1, -1)
                        if tilde_state is not None:
                            arr = np.asarray(tilde_state)
                            if arr.ndim == 3 and arr.shape[0] > 0 and arr.shape[1] > 0 and arr.shape[2] > 0:
                                ch0 = min(max(int(agent0_pos_channel), 0), arr.shape[2] - 1)
                                ch1 = min(max(int(agent1_pos_channel), 0), arr.shape[2] - 1)
                                plane0 = arr[:, :, ch0]
                                plane1 = arr[:, :, ch1]
                                if float(np.max(plane0)) > 0.0:
                                    flat0 = int(np.argmax(plane0))
                                    y0, x0 = np.unravel_index(flat0, plane0.shape)
                                    p0 = (int(y0), int(x0))
                                if float(np.max(plane1)) > 0.0:
                                    flat1 = int(np.argmax(plane1))
                                    y1, x1 = np.unravel_index(flat1, plane1.shape)
                                    p1 = (int(y1), int(x1))

                        reward_str = f"{float(rollout.total_reward):.3f}"
                        filename = (
                            f"{run_name}_run_{run_key}_{ckpt_name}_{mode_lc}_{reward_str}"
                            f"_ego({p0[0]},{p0[1]})_partner({p1[0]},{p1[1]}).gif"
                        )
                        local_path = os.path.join(output_dir, filename)
                        shutil.copyfile(video_path, local_path)
                        saved_videos.append(local_path)
                    except Exception as e:
                        print(
                            f"[WARN] {snapshot_dir}: failed to save video for mode={mode_lc} "
                            f"({e})"
                        )
                    finally:
                        try:
                            os.remove(video_path)
                        except Exception:
                            pass

        if not mode_curves:
            print(f"[WARN] skip snapshot with no plottable mode: {snapshot_dir}")
            continue

        out_name = f"{run_name}_run_{run_key}_{ckpt_name}_episode_reward_distance.png"
        out_path = os.path.join(output_dir, out_name)
        _plot_episode_reward_distance(
            mode_curves=mode_curves,
            modes=distance_eval_modes,
            run_name=run_name,
            run_key=run_key,
            ckpt_name=ckpt_name,
            episode_seed=episode_seed,
            output_path=out_path,
            distance_metric=args.distance_metric,
        )
        saved.append(out_path)

    if not saved:
        raise RuntimeError("No distance-distribution plots were generated.")

    print("=== Distance Distribution Eval Done ===")
    print(f"run_dir: {run_base_dir}")
    print(f"output_dir: {output_dir}")
    print(f"snapshot count (final): {len(snapshot_dirs)}")
    print(f"modes: {list(distance_eval_modes)}")
    print(f"distance metric: {args.distance_metric}")
    print(f"video enabled: {distance_eval_log_video}")
    print("saved:")
    for p in saved:
        print(f" - {p}")
    if saved_videos:
        print("saved videos:")
        for p in saved_videos:
            print(f" - {p}")


def _run_summary(args: argparse.Namespace) -> None:
    if not args.start_run or not args.end_run:
        raise ValueError("start-run/end-run must be provided via CLI or PRESET_* constants.")

    selected_runs = _resolve_runs_in_range(args.runs_dir, args.start_run, args.end_run)
    rows = []
    run_beta_end_cache: Dict[str, float] = {}
    for run_name in selected_runs:
        run_phase = _infer_phase_from_name(run_name)
        if args.phase != "auto" and run_phase != args.phase:
            continue
        csv_path = os.path.join(args.runs_dir, run_name, "eval", "offline_eval_analysis.csv")
        if not os.path.exists(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[WARN] failed to read {csv_path}: {e}")
            continue
        required = {"env", "mode", "env_step", "omega", "sigma", args.metric, args.distance_metric}
        if not required.issubset(df.columns):
            print(f"[WARN] skip {run_name}: missing columns {sorted(required - set(df.columns))}")
            continue
        df = df.copy()
        if run_name not in run_beta_end_cache:
            run_beta_end_cache[run_name] = _resolve_run_beta_end(args.runs_dir, run_name)
        beta_from_ckpt = run_beta_end_cache[run_name]
        if np.isfinite(beta_from_ckpt):
            df["beta_end"] = float(beta_from_ckpt)
        elif "beta_end" in df.columns:
            df["beta_end"] = pd.to_numeric(df["beta_end"], errors="coerce")
            print(
                f"[WARN] {run_name}: checkpoint config beta_end load failed; "
                f"fallback to offline_eval CSV beta_end."
            )
        else:
            df["beta_end"] = float("nan")
            print(
                f"[WARN] {run_name}: beta_end not found in checkpoint config or offline_eval CSV; "
                f"using NaN bucket."
            )
        df["source_run"] = run_name
        df["phase"] = run_phase
        rows.append(df)

    if not rows:
        raise RuntimeError("No usable offline_eval_analysis.csv found in selected range.")

    all_df = pd.concat(rows, ignore_index=True)
    all_df = all_df[all_df["mode"].isin(args.modes)]
    all_df["omega"] = pd.to_numeric(all_df["omega"], errors="coerce")
    all_df["sigma"] = pd.to_numeric(all_df["sigma"], errors="coerce")
    all_df["beta_end"] = pd.to_numeric(all_df["beta_end"], errors="coerce")
    all_df["env_step"] = pd.to_numeric(all_df["env_step"], errors="coerce")
    if args.env:
        all_df = all_df[all_df["env"] == args.env]
    all_df = all_df.dropna(subset=["omega", "sigma", "env_step"])
    if all_df.empty:
        raise RuntimeError("No data left after filtering by phase/env/modes.")

    envs = sorted(all_df["env"].astype(str).unique())
    phases = sorted(all_df["phase"].astype(str).unique())
    if args.phase == "auto":
        phase_name = phases[0] if len(phases) == 1 else "mixed"
    else:
        phase_name = args.phase

    summary_output_dir = (
        str(args.output_dir).strip()
        if str(args.output_dir).strip()
        else os.path.join(args.runs_dir, f"plots/{eval_env}")
    )
    os.makedirs(summary_output_dir, exist_ok=True)
    saved = []
    for env_name in envs:
        env_df = all_df[all_df["env"] == env_name].copy()
        combo_df = (
            env_df[["omega", "sigma", "beta_end"]]
            .drop_duplicates()
            .sort_values(["omega", "sigma", "beta_end"], na_position="last")
            .reset_index(drop=True)
        )
        combos: List[ComboKey] = [
            (
                float(r["omega"]),
                float(r["sigma"]),
                _to_numeric_or_nan(r.get("beta_end", float("nan"))),
            )
            for _, r in combo_df.iterrows()
        ]
        combo_chunks = _split_combos(
            combos=combos,
            per_plot=max(1, int(args.combos_per_plot)),
        )
        rand = secrets.token_hex(4)
        for i, combo_chunk in enumerate(combo_chunks):
            part_suffix = f"_p{i + 1}of{len(combo_chunks)}" if len(combo_chunks) > 1 else ""
            out_name = f"{phase_name}_{env_name}_evaluation_{rand}{part_suffix}.png"
            out_path = os.path.join(summary_output_dir, out_name)
            _plot_env_phase(
                df=env_df,
                env_name=env_name,
                phase_name=phase_name,
                metric=args.metric,
                distance_metric=args.distance_metric,
                combos=combo_chunk,
                part_idx=i,
                part_total=len(combo_chunks),
                xtick_step=max(1, int(args.xtick_step)),
                smooth_window=max(1, int(args.smooth_window)),
                legend_font_size=max(8, int(args.legend_font_size)),
                legend_title_font_size=max(9, int(args.legend_title_font_size)),
                modes=tuple(args.modes),
                output_path=out_path,
            )
            saved.append(out_path)

    print("=== PH1/PH2 Summary Plot Done ===")
    print(f"runs_dir: {args.runs_dir}")
    print(f"selected range: {args.start_run} ~ {args.end_run}")
    print(f"selected runs: {len(selected_runs)}")
    print(f"used rows: {len(all_df)}")
    print(f"phase: {phase_name}")
    print(f"metric: {args.metric}")
    print(f"distance metric: {args.distance_metric}")
    print(f"envs: {envs}")
    print("saved:")
    for p in saved:
        print(f" - {p}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize offline eval CSVs across selected run-folder range and plot "
            "normal/recent/random curves grouped by omega/sigma/beta_end."
        )
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_runs_dir = os.path.join(script_dir, "runs")
    parser.add_argument("--task", choices=["summary", "distance-distribution-eval"], default=PRESET_TASK)
    parser.add_argument("--runs-dir", default=default_runs_dir)
    parser.add_argument("--start-run", default=PRESET_START_RUN)
    parser.add_argument("--end-run", default=PRESET_END_RUN)
    parser.add_argument("--run-dir", default=PRESET_DISTANCE_EVAL_RUN, help="Target run dir for distance-distribution-eval mode")
    parser.add_argument("--phase", choices=["ph1", "ph2", "auto"], default=PRESET_PHASE)
    parser.add_argument("--env", default=PRESET_ENV, help="Optional env filter (e.g. grounded_coord_simple)")
    parser.add_argument("--metric", choices=["total_reward"], default=PRESET_METRIC)
    parser.add_argument("--distance-metric", default=PRESET_DISTANCE_METRIC)
    parser.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES))
    parser.add_argument("--distance-eval-modes", nargs="+", default=list(PRESET_DISTANCE_EVAL_MODES))
    parser.add_argument("--distance-eval-max-steps", type=int, default=PRESET_DISTANCE_EVAL_MAX_STEPS)
    parser.add_argument("--distance-eval-seed", type=int, default=PRESET_DISTANCE_EVAL_SEED)
    parser.add_argument("--distance-eval-env-device", choices=["cpu", "gpu", "cuda"], default=PRESET_DISTANCE_EVAL_ENV_DEVICE)
    parser.add_argument("--distance-eval-log-video", default=str(PRESET_DISTANCE_EVAL_LOG_VIDEO))
    parser.add_argument("--combos-per-plot", type=int, default=PRESET_COMBOS_PER_PLOT)
    parser.add_argument("--xtick-step", type=int, default=PRESET_XTICK_STEP)
    parser.add_argument("--smooth-window", type=int, default=PRESET_SMOOTH_WINDOW)
    parser.add_argument("--legend-font-size", type=int, default=PRESET_LEGEND_FONT_SIZE)
    parser.add_argument(
        "--legend-title-font-size",
        type=int,
        default=PRESET_LEGEND_TITLE_FONT_SIZE,
    )
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()
    if args.task == "distance-distribution-eval":
        _run_distance_distribution_eval(args)
        return
    _run_summary(args)


if __name__ == "__main__":
    main()
