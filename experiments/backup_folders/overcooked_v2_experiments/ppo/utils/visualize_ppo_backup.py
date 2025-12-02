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
    policy_checkoints_to_policy_pairing,
)
from overcooked_v2_experiments.ppo.utils.store import (
    load_all_checkpoints,
)
from overcooked_v2_experiments.helper.plots import visualize_cross_play_matrix
from overcooked_v2_experiments.utils.utils import (
    mini_batch_pmap,
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
    if cross and not final_only:
        raise ValueError("Cannot run cross play with all checkpoints")

    all_params, config = load_all_checkpoints(run_base_dir, final_only=final_only)

    print("DEBUG: Model TYPE:", config["model"]["TYPE"])
    print("DEBUG: Config model keys:", list(config["model"].keys()))

    # 첫 번째 체크포인트의 파라미터 shape 확인
    first_run_key = list(all_params.keys())[0]
    first_ckpt_key = list(all_params[first_run_key].keys())[0]
    params = all_params[first_run_key][first_ckpt_key].params
    print("DEBUG: Params keys:", list(params.keys()))
    if "params" in params and "CNN_0" in params["params"]:
        print(
            "DEBUG: CNN_0 Conv_0 kernel shape:",
            params["params"]["CNN_0"]["Conv_0"]["kernel"].shape,
        )
        print(
            "DEBUG: CNN_0 Conv_0 bias shape:",
            params["params"]["CNN_0"]["Conv_0"]["bias"].shape,
        )
    else:
        print("DEBUG: CNN_0 not found in params.params")

    initial_env_kwargs = copy.deepcopy(config["env"]["ENV_KWARGS"])
    env_kwargs = initial_env_kwargs | extra_env_kwargs
    env = OvercookedV2(**env_kwargs)

    num_actors = env.num_agents

    run_keys = list(all_params.keys())

    # restructure if cross play, layout 1. "cross", 2. "run_combinations"
    if cross:
        num_runs = len(run_keys)

        run_combinations = itertools.permutations(range(num_runs), num_actors)
        run_combinations = list(run_combinations)
        # add self play
        run_combinations += [[i] * num_actors for i in range(num_runs)]

        if pairing_policy is not None:
            run_combinations = [
                [pairing_policy, i] for i in range(num_runs) if i != pairing_policy
            ]
            run_combinations += [
                [i, pairing_policy] for i in range(num_runs) if i != pairing_policy
            ]

        print("Run combinations: ", run_combinations)

        policy_pairings = [
            all_params[run_keys[i]]["ckpt_final"] for i in range(num_runs)
        ]

        cross_combinations = {}
        for run_combination in run_combinations:
            run_combination = list(run_combination)

            run_ids = [run_keys[i].replace("run_", "") for i in run_combination]
            run_combination_key = "cross-" + "_".join(run_ids)
            policy_combination = PolicyPairing(
                *[policy_pairings[i] for i in run_combination]
            )

            cross_combinations[run_combination_key] = policy_combination

        all_params = {"cross": cross_combinations}
    else:
        # 각 run의 단일 policy를 num_actors개로 복제한 PolicyPairing 구조로 바꿔줌
        all_params = jax.tree_util.tree_map(
            lambda x: PolicyPairing.from_single_policy(x, num_actors),
            all_params,
            is_leaf=lambda x: type(x) is PPOParams,
        )

    # 여기까지 all_params 구조:
    #   sp 모드: dict[run][ckpt] -> PolicyPairing
    #   cross 모드: {"cross": dict[run_comb_key] -> PolicyPairing}

    def _policy_viz(pairing):
        env_kwargs_no_layout = copy.deepcopy(env_kwargs)
        layout_name = env_kwargs_no_layout.pop("layout")

        pairing = policy_checkoints_to_policy_pairing(pairing, config)

        return eval_pairing(
            pairing,
            layout_name,
            key,
            env_kwargs=env_kwargs_no_layout,
            num_seeds=num_seeds,
            all_recipes=num_seeds is None,
            no_viz=no_viz,
        )

    # PolicyPairing들만 평평하게 펼침
    policy_pairings, treedef = jax.tree_util.tree_flatten(
        all_params, is_leaf=lambda x: type(x) is PolicyPairing
    )

    # ⚠️ 파라미터를 stack해서 (10, ...) 만들지 말고,
    #    각 PolicyPairing을 개별로 _policy_viz에 넣어서 순차로 평가
    results = []
    for idx, pairing in enumerate(policy_pairings):
        print(f"[EVAL] Running _policy_viz for policy_pairing {idx+1}/{len(policy_pairings)}")
        viz_result = _policy_viz(pairing)
        results.append(viz_result)

    # viz 결과를 원래 트리 구조(dict[run][ckpt] -> dict[annotation -> viz])로 복원
    all_params = jax.tree_util.tree_unflatten(treedef, results)

    labels = ["run", "checkpoint"]
    if cross:
        labels[1] = "policy_labels"

    rows = []
    for first_level, first_level_runs in all_params.items():
        for second_level, second_level_runs in first_level_runs.items():
            checkpoint_sum = 0.0

            print(f"{labels[0]}: {first_level}, {labels[1]}: {second_level}")
            for annotation, viz in second_level_runs.items():
                frame_seq = viz.frame_seq
                total_reward = viz.total_reward

                if not no_viz:
                    viz_dir = run_base_dir / first_level / second_level
                    os.makedirs(viz_dir, exist_ok=True)
                    viz_filename = viz_dir / f"{annotation}.gif"

                    imageio.mimsave(viz_filename, frame_seq, "GIF", duration=0.5)

                checkpoint_sum += total_reward
                rows.append([first_level, second_level, annotation, total_reward])
                print(f"\t{annotation}:\t{total_reward}")
            reward_mean = checkpoint_sum / len(second_level_runs)
            print(f"\tMean reward:\t{reward_mean}")

    summery_name = "reward_summary_cross.csv" if cross else "reward_summary_sp.csv"
    summery_file = run_base_dir / summery_name
    with open(summery_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        fieldnames = [labels[0], labels[1], "annotation", "total_reward"]

        writer.writerow(fieldnames)

        for row in rows:
            writer.writerow(row)

    print(f"Summary written to {summery_file}")

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
