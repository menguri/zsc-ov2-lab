import argparse
import sys
import os
import itertools
import jax.numpy as jnp
import jax
import copy
from datetime import datetime
from pathlib import Path
import chex
import imageio
import csv
from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2


DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(DIR))
sys.path.append(os.path.dirname(os.path.dirname(DIR)))

from overcooked_v2_experiments.ppo.policy import (
    PPOParams,
    PPOPolicy,
    policy_checkoints_to_policy_pairing,
)
from overcooked_v2_experiments.ppo.utils.store import (
    load_all_checkpoints,
)
from overcooked_v2_experiments.helper.plots import visualize_cross_play_matrix
from overcooked_v2_experiments.utils.utils import (
    mini_batch_pmap,          # 지금은 안 쓰지만, 기존 인터페이스 유지용으로 둠
    scanned_mini_batch_map,
)
from overcooked_v2_experiments.eval.evaluate import eval_pairing
from overcooked_v2_experiments.eval.policy import PolicyPairing


def visualize_ppo_policy(
    run_base_dir,
    key,
    final_only=True,
    extra_env_kwargs={},
    num_seeds=None,
    cross=False,
    no_viz=False,
    pairing_policy=None,
):
    # cross-play인데 모든 ckpt를 쓰려고 하면 모순 → 방어 코드
    if cross and not final_only:
        raise ValueError("Cannot run cross play with all checkpoints")

    # 1) PPO 파라미터 로드: all_params: dict[run_id][ckpt_id] -> PPOParams
    all_params, config, configs = load_all_checkpoints(run_base_dir, final_only=final_only)

    # 2) 환경 생성
    initial_env_kwargs = copy.deepcopy(config["env"]["ENV_KWARGS"])
    env_kwargs = initial_env_kwargs | extra_env_kwargs
    env = OvercookedV2(**env_kwargs)

    num_actors = env.num_agents
    run_keys = list(all_params.keys())

    # 3) cross-play 모드: 서로 다른 run 조합으로 PolicyPairing 구성
    if cross:
        num_runs = len(run_keys)

        # 예: num_actors=2면 (0,1), (1,0), (0,0), (1,1) ...
        run_combinations = itertools.permutations(range(num_runs), num_actors)
        run_combinations = list(run_combinations)

        # self-play 조합 추가
        run_combinations += [[i] * num_actors for i in range(num_runs)]

        # 특정 run을 고정 파트너로 쓰고 싶을 때
        if pairing_policy is not None:
            run_combinations = [
                [pairing_policy, i] for i in range(num_runs) if i != pairing_policy
            ]
            run_combinations += [
                [i, pairing_policy] for i in range(num_runs) if i != pairing_policy
            ]

        print("Run combinations: ", run_combinations)

        # 각 run의 ckpt_final만 모아놓기
        policy_pairings = [
            all_params[run_keys[i]]["ckpt_final"] for i in range(num_runs)
        ]

        cross_combinations = {}
        for run_combination in run_combinations:
            run_combination = list(run_combination)

            # run_7 → "7" 이런 식으로 label 만들기
            run_ids = [run_keys[i].replace("run_", "") for i in run_combination]
            run_combination_key = "cross-" + "_".join(run_ids)

            # PolicyPairing(PPOParams_0, PPOParams_1, ...) 형태로 묶음
            policy_combination = PolicyPairing(
                *[policy_pairings[i] for i in run_combination]
            )

            cross_combinations[run_combination_key] = policy_combination

        # cross 모드 구조: {"cross": {comb_key: PolicyPairing}}
        all_params = {"cross": cross_combinations}

    else:
        # self-play 모드: 한 run의 PPOParams를 num_actors개로 복제해서 PolicyPairing 하나로 만들기
        # 구조: dict[run_id][ckpt_id] -> PolicyPairing
        all_params = jax.tree_util.tree_map(
            lambda x: PolicyPairing.from_single_policy(x, num_actors),
            all_params,
            is_leaf=lambda x: type(x) is PPOParams,
        )

    # JIT Cache
    jit_cache = {}
    results_structure = {}

    # Iterate over all pairings
    for first_level, first_level_runs in all_params.items():
        results_structure[first_level] = {}
        for second_level, pairing in first_level_runs.items():
            # Determine algs and configs
            if first_level == "cross":
                # Parse second_level (comb_key) e.g. "cross-0_1"
                run_indices = second_level.replace("cross-", "").split("_")
                
                # Map back to run_ids using run_keys
                # Note: run_keys are "run_0", "run_1" etc.
                # run_indices are "0", "1" etc.
                # We assume run_keys are sorted or consistent with indices used in run_combinations
                # run_combinations used range(num_runs), so indices correspond to run_keys index
                
                # But wait, run_keys is list(all_params.keys()) BEFORE modification.
                # We need to ensure run_keys is consistent.
                # run_keys was created before 'if cross:' block.
                
                # run_indices are indices into run_keys list?
                # In 'run_ids = [run_keys[i].replace("run_", "") for i in run_combination]'
                # Yes, run_combination contains indices into run_keys.
                # But 'run_indices' here comes from 'second_level' string which used 'run_ids'.
                # 'run_ids' are "0", "1" etc.
                # So we need to find run_key that has "run_0".
                
                current_run_ids = [f"run_{idx}" for idx in run_indices]
                current_configs = [configs[rid] for rid in current_run_ids]
                
                # Get alg names
                current_algs = []
                for cfg in current_configs:
                    alg = cfg.get("ALG_NAME", "PPO")
                    if "alg" in cfg:
                        alg = cfg["alg"].get("ALG_NAME", alg)
                    
                    # Check for STL (anchor)
                    # config 구조가 다양할 수 있으므로 여러 경로 확인
                    is_anchor = False
                    if "anchor" in cfg:
                        is_anchor = cfg["anchor"]
                    elif "alg" in cfg and "anchor" in cfg["alg"]:
                        is_anchor = cfg["alg"]["anchor"]
                    # model config 내부에 있을 수도 있음 (예: config.model.anchor)
                    elif "model" in cfg and "anchor" in cfg["model"]:
                        is_anchor = cfg["model"]["anchor"]
                    
                    if alg == "E3T" and is_anchor:
                        alg = "STL"
                        
                    current_algs.append(alg)
                current_algs = tuple(current_algs)
                
            else:
                # Self-play
                # first_level is run_id
                run_id = first_level
                cfg = configs[run_id]
                alg = cfg.get("ALG_NAME", "PPO")
                if "alg" in cfg:
                    alg = cfg["alg"].get("ALG_NAME", alg)
                
                # Check for STL (anchor)
                is_anchor = False
                if "anchor" in cfg:
                    is_anchor = cfg["anchor"]
                elif "alg" in cfg and "anchor" in cfg["alg"]:
                    is_anchor = cfg["alg"]["anchor"]
                elif "model" in cfg and "anchor" in cfg["model"]:
                    is_anchor = cfg["model"]["anchor"]
                
                if alg == "E3T" and is_anchor:
                    alg = "STL"
                
                current_configs = [cfg] * num_actors
                current_algs = tuple([alg] * num_actors)
            
            # Determine alg_arg for eval_pairing
            # Heuristic: if any E3T, use E3T. Else use first.
            if "E3T" in current_algs:
                alg_arg = "E3T"
            elif "STL" in current_algs:
                alg_arg = "STL"
            else:
                alg_arg = current_algs[0]
            
            # JIT Key
            jit_key = current_algs
            
            if jit_key not in jit_cache:
                print(f"Compiling JIT for pair: {jit_key}")
                
                # Define function to JIT
                # We capture current_configs and alg_arg in closure
                
                def _viz_impl(pairing_params):
                    policies = []
                    for i in range(num_actors):
                        # pairing_params is a PolicyPairing object, which is a PyTree.
                        # When flattened/unflattened by JAX, it behaves like a list of policies.
                        # However, here pairing_params is passed directly.
                        # PolicyPairing stores policies in self.policies list.
                        # But wait, pairing_params passed to _viz_impl is likely the PPOParams structure 
                        # if we look at how it was constructed in all_params.
                        
                        # In self-play: PolicyPairing.from_single_policy(PPOParams, num_actors)
                        # In cross-play: PolicyPairing(PPOParams_0, PPOParams_1)
                        
                        # So pairing_params[i] should give us the PPOParams for agent i.
                        agent_params = pairing_params[i]
                        
                        # Create policy with captured config
                        policies.append(PPOPolicy(agent_params.params, current_configs[i]))
                    
                    policy_pairing = PolicyPairing(*policies)
                    
                    env_kwargs_no_layout = copy.deepcopy(env_kwargs)
                    layout_name = env_kwargs_no_layout.pop("layout")
                    
                    return eval_pairing(
                        policy_pairing,
                        layout_name,
                        key,
                        env_kwargs=env_kwargs_no_layout,
                        num_seeds=num_seeds,
                        all_recipes=num_seeds is None,
                        no_viz=no_viz,
                        algorithm=alg_arg,
                    )
                
                if no_viz:
                    jit_cache[jit_key] = jax.jit(_viz_impl)
                else:
                    jit_cache[jit_key] = _viz_impl
            
            # Execute
            print(f"[EVAL] Running {jit_key} for {first_level}/{second_level}")
            viz_result = jit_cache[jit_key](pairing)
            results_structure[first_level][second_level] = viz_result
    
    # Update all_params with results
    all_params = results_structure

    labels = ["run", "checkpoint"]
    if cross:
        labels[1] = "policy_labels"

    # 6) reward summary + gif 저장
    rows = []
    for first_level, first_level_runs in all_params.items():
        for second_level, second_level_runs in first_level_runs.items():
            checkpoint_sum = 0.0
            acc_sum = jnp.zeros(num_actors)
            acc_count = 0

            print(f"{labels[0]}: {first_level}, {labels[1]}: {second_level}")
            for annotation, viz in second_level_runs.items():
                frame_seq = viz.frame_seq
                total_reward = viz.total_reward
                pred_acc = viz.prediction_accuracy

                if not no_viz:
                    viz_dir = run_base_dir / first_level / second_level
                    os.makedirs(viz_dir, exist_ok=True)
                    viz_filename = viz_dir / f"{annotation}.gif"
                    imageio.mimsave(viz_filename, frame_seq, "GIF", duration=0.5)

                checkpoint_sum += total_reward
                row = [first_level, second_level, annotation, total_reward]
                
                if pred_acc is not None:
                    acc_sum += pred_acc
                    acc_count += 1
                    
                    for i in range(pred_acc.shape[0]):
                        row.append(float(pred_acc[i]))
                else:
                    for i in range(num_actors):
                        row.append(0.0)
                
                rows.append(row)
                print(f"\t{annotation}:\t{total_reward}")
            
            reward_mean = checkpoint_sum / len(second_level_runs)
            print(f"\tMean reward:\t{reward_mean}")
            
            mean_row = [first_level, second_level, "mean", reward_mean]
            if acc_count > 0:
                acc_mean = acc_sum / acc_count
                print(f"\tMean accuracy:\t{acc_mean}")
                for i in range(acc_mean.shape[0]):
                    mean_row.append(float(acc_mean[i]))
            else:
                for i in range(num_actors):
                    mean_row.append(0.0)
            
            rows.append(mean_row)

    # 7) CSV로 요약 저장
    summery_name = "reward_summary_cross.csv" if cross else "reward_summary_sp.csv"
    summery_file = run_base_dir / summery_name
    with open(summery_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        fieldnames = [labels[0], labels[1], "annotation", "total_reward"]
        for i in range(num_actors):
            fieldnames.append(f"pred_acc_agent_{i}")
            
        writer.writerow(fieldnames)
        for row in rows:
            writer.writerow(row)

    print(f"Summary written to {summery_file}")

    # 8) cross-play면 교차플레이 매트릭스도 그림
    if cross:
        visualize_cross_play_matrix(summery_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_seeds", type=int)
    parser.add_argument("--all_ckpt", action="store_true")
    parser.add_argument("--cross", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--no_viz", action="store_true")
    parser.add_argument("--no_reset", action="store_true")
    parser.add_argument("--pairing_policy", type=int)

    args = parser.parse_args()

    directory = args.d
    num_seeds = args.num_seeds
    final_only = not args.all_ckpt
    cross = args.cross

    key = jax.random.PRNGKey(args.seed)
    key_sp, key_cross = jax.random.split(key, 2)

    viz_mode = {
        "sp": (not cross) or args.all,
        "cross": cross or args.all,
    }
    modes = [m for m, v in viz_mode.items() if v]

    extra_env_kwargs = {}
    if args.no_reset:
        extra_env_kwargs["random_reset"] = False
        extra_env_kwargs["op_ingredient_permutations"] = False

    for mode in modes:
        fo = final_only or (mode == "cross")
        visualize_ppo_policy(
            Path(directory),
            key_sp if mode == "sp" else key_cross,
            num_seeds=num_seeds,
            final_only=fo,
            cross=mode == "cross",
            no_viz=args.no_viz,
            extra_env_kwargs=extra_env_kwargs,
            pairing_policy=args.pairing_policy,
        )
