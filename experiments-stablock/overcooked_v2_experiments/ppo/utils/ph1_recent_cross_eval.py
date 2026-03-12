import argparse
import copy
import csv
import itertools
import os
from pathlib import Path
from typing import Dict, List, Optional

import jax
import numpy as np
from PIL import Image

from overcooked_v2_experiments.eval.evaluate import eval_pairing
from overcooked_v2_experiments.eval.policy import PolicyPairing
from overcooked_v2_experiments.eval.utils import (
    make_eval_env,
    resolve_old_overcooked_flags,
)
from overcooked_v2_experiments.helper.plots import visualize_cross_play_matrix
from overcooked_v2_experiments.ppo.ph1_online_eval import (
    _extract_tilde_agent_pos,
    _get_agent_pos_channels_from_env,
)
from overcooked_v2_experiments.ppo.policy import PPOPolicy
from overcooked_v2_experiments.ppo.utils.store import load_all_checkpoints

PH1_DEFAULT_EVAL_STEPS = 400


def _get_stablock(cfg: Dict) -> bool:
    if "alg" in cfg and "STABLOCK_ENABLED" in cfg["alg"]:
        return bool(cfg["alg"]["STABLOCK_ENABLED"])
    return bool(cfg.get("STABLOCK_ENABLED", False))


def _resolve_policy_config(cfg: Dict, policy_source: str) -> Dict:
    out = copy.deepcopy(cfg)
    source = str(policy_source).strip().lower()
    if source not in ("ind", "spec"):
        return out

    key = "PH2_IND_USE_BLOCKED_INPUT" if source == "ind" else "PH2_SPEC_USE_BLOCKED_INPUT"
    use_blocked_input = out.get(key, None)
    if "alg" in out and isinstance(out["alg"], dict):
        use_blocked_input = out["alg"].get(key, use_blocked_input)
    if use_blocked_input is None:
        # Backward-compatible fallback for old PH2 checkpoints:
        # prefer an existing learner flag if present, otherwise
        # default to ind=False / spec=True.
        learner_flag = out.get("LEARNER_USE_BLOCKED_INPUT", None)
        if "alg" in out and isinstance(out["alg"], dict):
            learner_flag = out["alg"].get(
                "LEARNER_USE_BLOCKED_INPUT", learner_flag
            )
        if learner_flag is None:
            learner_flag = (source != "ind")
        use_blocked_input = learner_flag

    out["LEARNER_USE_BLOCKED_INPUT"] = bool(use_blocked_input)
    if "alg" in out and isinstance(out["alg"], dict):
        out["alg"]["LEARNER_USE_BLOCKED_INPUT"] = bool(use_blocked_input)
    return out


def _resolve_alg_label(cfg: Dict) -> str:
    alg = cfg.get("ALG_NAME", "PPO")
    if "alg" in cfg:
        alg = cfg["alg"].get("ALG_NAME", alg)

    ph1_enabled = bool(cfg.get("PH1_ENABLED", False))
    if "alg" in cfg:
        ph1_enabled = ph1_enabled or bool(cfg["alg"].get("PH1_ENABLED", False))

    is_anchor = False
    if "anchor" in cfg:
        is_anchor = cfg["anchor"]
    elif "alg" in cfg and "anchor" in cfg["alg"]:
        is_anchor = cfg["alg"]["anchor"]
    elif "model" in cfg and "anchor" in cfg["model"]:
        is_anchor = cfg["model"]["anchor"]

    if alg == "E3T" and is_anchor:
        alg = "STL"

    if ph1_enabled and "PH1" not in str(alg):
        alg = f"PH1-{alg}"

    return str(alg)


def _resolve_eval_algorithm(algs: List[str]) -> str:
    if any("PH1" in alg for alg in algs):
        if any("E3T" in alg for alg in algs):
            return "PH1-E3T"
        if any("STL" in alg for alg in algs):
            return "PH1-STL"
        return "PH1"
    if any("E3T" in alg for alg in algs):
        return "E3T"
    if any("STL" in alg for alg in algs):
        return "STL"
    return algs[0]


def _snapshot_step_from_path(path: Path) -> int:
    for part in reversed(path.parts):
        if part == "ckpt_final":
            return 10**18
        if part.startswith("ckpt_"):
            try:
                return int(part.split("_", 1)[1])
            except Exception:
                continue
    return -1


def _collect_recent_tildes(run_base_dir: Path) -> List[Dict]:
    eval_root = run_base_dir / "eval"
    if not eval_root.is_dir():
        return []

    entries = []
    for npz_path in sorted(eval_root.rglob("mode_tildes.npz")):
        try:
            with np.load(npz_path) as data:
                has_recent = bool(np.asarray(data.get("has_recent", np.array([0]))).reshape(-1)[0])
                if not has_recent or ("recent_tilde" not in data):
                    continue
                tilde = np.asarray(data["recent_tilde"]).astype(np.float32)
                if tilde.size == 0 or tilde.ndim != 3:
                    continue
                entries.append(
                    {
                        "tilde": tilde,
                        "source": str(npz_path),
                        "step": _snapshot_step_from_path(npz_path.parent),
                    }
                )
        except Exception:
            continue

    return entries


def _sample_recent_tildes(entries: List[Dict], n_samples: int, seed: int) -> List[Dict]:
    if n_samples <= 0:
        raise ValueError("num_recent_tildes must be >= 1")
    if len(entries) == 0:
        raise RuntimeError("No recent tilde states found under run_dir/eval")

    rng = np.random.default_rng(seed)
    replace = len(entries) < n_samples
    picked = rng.choice(len(entries), size=n_samples, replace=replace)
    return [entries[int(i)] for i in picked]


def _build_run_combinations(num_runs: int, num_agents: int, pairing_policy: Optional[int]) -> List[List[int]]:
    if pairing_policy is not None:
        if pairing_policy < 0 or pairing_policy >= num_runs:
            raise ValueError(f"pairing_policy must be in [0, {num_runs - 1}]")
        combos = [[pairing_policy, i] for i in range(num_runs) if i != pairing_policy]
        combos += [[i, pairing_policy] for i in range(num_runs) if i != pairing_policy]
        return combos

    combos = list(itertools.permutations(range(num_runs), num_agents))
    combos = [list(c) for c in combos]
    combos += [[i] * num_agents for i in range(num_runs)]
    return combos


def _annotation_prefix(sample_idx: int, step: int, ego_pos, par_pos) -> str:
    step_text = "final" if step >= 10**18 else str(int(step))
    return (
        f"tilde-{int(sample_idx):03d}_step-{step_text}"
        f"_ego({int(ego_pos[0])},{int(ego_pos[1])})"
        f"_par({int(par_pos[0])},{int(par_pos[1])})"
    )


def _save_gif_with_expected_steps(
    frame_seq: np.ndarray, out_path: Path, expected_steps: int
) -> None:
    """Save GIF and require exactly expected_steps evaluated frames."""
    frames = np.asarray(frame_seq)
    if frames.ndim != 4 or frames.shape[0] == 0:
        raise ValueError(f"Invalid frame_seq shape for gif export: {frames.shape}")

    target_frames = int(expected_steps)
    frame_count = int(frames.shape[0])
    if frame_count != target_frames:
        raise RuntimeError(
            f"PH1 eval frame count must be {target_frames}, got {frame_count} ({out_path})"
        )

    pil_frames: List[Image.Image] = []
    for frame in frames:
        arr = np.asarray(frame)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        pil_frames.append(Image.fromarray(arr))

    # Fixed per-frame duration and no optimization for stable playback.
    pil_frames[0].save(
        str(out_path),
        save_all=True,
        append_images=pil_frames[1:],
        loop=0,
        duration=100,
        optimize=False,
        disposal=2,
    )


def run_ph1_recent_cross_eval(
    run_base_dir: Path,
    seed: int,
    num_seeds: int,
    num_recent_tildes: int,
    no_viz: bool,
    no_reset: bool,
    pairing_policy: Optional[int],
    policy_source: str,
    max_steps: int = PH1_DEFAULT_EVAL_STEPS,
    old_overcooked_override: Optional[bool] = None,
    disable_old_overcooked_auto_override: Optional[bool] = None,
):
    all_params, config, configs = load_all_checkpoints(
        run_base_dir,
        final_only=True,
        policy_source=policy_source,
    )

    run_keys = list(all_params.keys())
    if len(run_keys) == 0:
        raise RuntimeError("No checkpoints found in run directory")

    initial_env_kwargs = copy.deepcopy(config["env"]["ENV_KWARGS"])
    env_kwargs = dict(initial_env_kwargs)
    cfg_old_overcooked, cfg_disable_old_auto = resolve_old_overcooked_flags(config)
    if old_overcooked_override is not None:
        cfg_old_overcooked = bool(old_overcooked_override)
    if disable_old_overcooked_auto_override is not None:
        cfg_disable_old_auto = bool(disable_old_overcooked_auto_override)

    target_steps = int(max_steps)
    if target_steps <= 0:
        raise ValueError("max_steps must be >= 1")

    if int(env_kwargs.get("max_steps", target_steps)) != target_steps:
        print(
            f"[PH1-RECENT-XP] override max_steps {int(env_kwargs.get('max_steps'))} -> {target_steps}",
            flush=True,
        )
    env_kwargs["max_steps"] = target_steps
    if no_reset:
        env_kwargs["random_reset"] = False
        env_kwargs["op_ingredient_permutations"] = False

    env_layout = env_kwargs.get("layout")
    env_kwargs_no_layout = copy.deepcopy(env_kwargs)
    env_kwargs_no_layout.pop("layout", None)
    env, env_name, _resolved_kwargs = make_eval_env(
        env_layout,
        env_kwargs_no_layout,
        old_overcooked=cfg_old_overcooked,
        disable_auto=cfg_disable_old_auto,
    )
    num_actors = env.num_agents
    if num_actors != 2:
        raise RuntimeError(f"PH1 recent-tilde cross eval currently expects 2 agents, got {num_actors}")

    agent0_pos_channel, agent1_pos_channel = _get_agent_pos_channels_from_env(
        env, env_name=env_name
    )

    recent_pool = _collect_recent_tildes(run_base_dir)
    sampled_tildes = _sample_recent_tildes(recent_pool, num_recent_tildes, seed)
    print(
        f"[PH1-RECENT-XP] collected recent tildes={len(recent_pool)} sampled={len(sampled_tildes)}"
    )

    num_runs = len(run_keys)
    run_combinations = _build_run_combinations(num_runs, num_actors, pairing_policy)
    print(f"[PH1-RECENT-XP] run combinations={run_combinations}")

    policy_pairings = [all_params[run_keys[i]]["ckpt_final"] for i in range(num_runs)]

    layout_name = env_layout

    results_structure = {"cross": {}}
    key = jax.random.PRNGKey(seed)
    total_combos = len(run_combinations)
    total_tildes = len(sampled_tildes)
    total_evals = total_combos * total_tildes
    done_evals = 0

    for comb_idx, comb in enumerate(run_combinations, start=1):
        run_ids = [run_keys[i].replace("run_", "") for i in comb]
        run_combination_key = "cross-" + "_".join(run_ids)
        print(
            f"[PH1-RECENT-XP] combo {comb_idx}/{total_combos}: {run_combination_key}",
            flush=True,
        )

        current_run_ids = [run_keys[i] for i in comb]
        current_configs = [
            _resolve_policy_config(configs[rid], policy_source=policy_source)
            for rid in current_run_ids
        ]
        current_algs = [_resolve_alg_label(cfg) for cfg in current_configs]
        alg_arg = _resolve_eval_algorithm(current_algs)
        current_stablock = [_get_stablock(cfg) for cfg in current_configs]

        policies = []
        for actor_idx, run_idx in enumerate(comb):
            agent_params = policy_pairings[run_idx]
            policies.append(PPOPolicy(agent_params.params, current_configs[actor_idx]))
        policy_pairing = PolicyPairing(*policies)

        combo_results = {}
        for sample_idx, tilde_entry in enumerate(sampled_tildes, start=1):
            tilde_state = tilde_entry["tilde"]
            ego_pos = _extract_tilde_agent_pos(tilde_state, agent0_pos_channel)
            par_pos = _extract_tilde_agent_pos(tilde_state, agent1_pos_channel)
            prefix = _annotation_prefix(sample_idx - 1, int(tilde_entry["step"]), ego_pos, par_pos)
            done_evals += 1
            print(
                (
                    f"[PH1-RECENT-XP] eval {done_evals}/{total_evals} "
                    f"(combo {comb_idx}/{total_combos}, tilde {sample_idx}/{total_tildes}) "
                    f"step={int(tilde_entry['step'])}"
                ),
                flush=True,
            )

            key, key_eval = jax.random.split(key)
            eval_runs = eval_pairing(
                policy_pairing,
                layout_name,
                key_eval,
                env_kwargs=env_kwargs_no_layout,
                num_seeds=num_seeds,
                all_recipes=False,
                no_viz=no_viz,
                algorithm=alg_arg,
                stablock_enabled=current_stablock,
                latent_analysis=False,
                value_analysis=False,
                ph1_forced_tilde_state=tilde_state,
                old_overcooked=cfg_old_overcooked,
                disable_old_overcooked_auto=cfg_disable_old_auto,
            )

            for annotation, viz in eval_runs.items():
                combo_results[f"{prefix}_{annotation}"] = viz

        results_structure["cross"][run_combination_key] = combo_results

    rows = []
    for first_level, first_level_runs in results_structure.items():
        for second_level, second_level_runs in first_level_runs.items():
            checkpoint_sum = 0.0
            acc_sum = np.zeros(num_actors, dtype=np.float32)
            acc_count = 0

            print(f"run: {first_level}, policy_labels: {second_level}")
            for annotation, viz in second_level_runs.items():
                frame_seq = viz.frame_seq
                total_reward = float(np.asarray(viz.total_reward))
                pred_acc = viz.prediction_accuracy

                if not no_viz:
                    viz_dir = run_base_dir / first_level / second_level
                    os.makedirs(viz_dir, exist_ok=True)
                    viz_filename = viz_dir / f"{annotation}.gif"
                    _save_gif_with_expected_steps(frame_seq, viz_filename, target_steps)

                checkpoint_sum += total_reward
                row = [first_level, second_level, annotation, total_reward]

                if pred_acc is not None:
                    pred_acc_np = np.asarray(pred_acc)
                    acc_sum += pred_acc_np
                    acc_count += 1
                    for i in range(pred_acc_np.shape[0]):
                        row.append(float(pred_acc_np[i]))
                else:
                    for _ in range(num_actors):
                        row.append(0.0)

                rows.append(row)
                print(f"\t{annotation}:\t{total_reward}")

            reward_mean = checkpoint_sum / max(1, len(second_level_runs))
            print(f"\tMean reward:\t{reward_mean}")

            mean_row = [first_level, second_level, "mean", reward_mean]
            if acc_count > 0:
                acc_mean = acc_sum / float(acc_count)
                print(f"\tMean accuracy:\t{acc_mean}")
                for i in range(acc_mean.shape[0]):
                    mean_row.append(float(acc_mean[i]))
            else:
                for _ in range(num_actors):
                    mean_row.append(0.0)
            rows.append(mean_row)

    summary_file = run_base_dir / "reward_summary_cross.csv"
    with open(summary_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        fieldnames = ["run", "policy_labels", "annotation", "total_reward"]
        for i in range(num_actors):
            fieldnames.append(f"pred_acc_agent_{i}")
        writer.writerow(fieldnames)
        for row in rows:
            writer.writerow(row)

    print(f"Summary written to {summary_file}")
    visualize_cross_play_matrix(summary_file)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "PH1 cross-play evaluation using sampled recent tilde states from run_dir/eval snapshots."
        )
    )
    parser.add_argument("--d", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_seeds", type=int, default=10)
    parser.add_argument("--num_recent_tildes", type=int, default=5)
    parser.add_argument("--no_viz", action="store_true")
    parser.add_argument("--no_reset", action="store_true")
    parser.add_argument("--pairing_policy", type=int)
    parser.add_argument(
        "--policy_source",
        type=str,
        default="params",
        choices=["params", "ind", "spec"],
    )
    parser.add_argument("--old_overcooked", action="store_true")
    parser.add_argument("--disable_old_overcooked_auto", action="store_true")
    parser.add_argument("--max_steps", type=int, default=PH1_DEFAULT_EVAL_STEPS)

    args = parser.parse_args()

    run_base_dir = Path(args.d).resolve()
    if not run_base_dir.exists() or not run_base_dir.is_dir():
        raise FileNotFoundError(f"run directory not found: {run_base_dir}")

    run_ph1_recent_cross_eval(
        run_base_dir=run_base_dir,
        seed=int(args.seed),
        num_seeds=int(args.num_seeds),
        num_recent_tildes=int(args.num_recent_tildes),
        no_viz=bool(args.no_viz),
        no_reset=bool(args.no_reset),
        pairing_policy=args.pairing_policy,
        policy_source=args.policy_source,
        max_steps=int(args.max_steps),
        old_overcooked_override=bool(args.old_overcooked),
        disable_old_overcooked_auto_override=bool(args.disable_old_overcooked_auto),
    )


if __name__ == "__main__":
    main()
