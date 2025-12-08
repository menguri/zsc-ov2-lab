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
    all_params, config = load_all_checkpoints(run_base_dir, final_only=final_only)

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

    # 여기까지 구조 요약:
    #   - self-play: all_params[run_id][ckpt_id] = PolicyPairing
    #   - cross-play: all_params["cross"][comb_key] = PolicyPairing

    def _policy_viz(pairing):
        """
        PolicyPairing 하나에 대해:
          1) 체크포인트 → 실제 policy 함수로 변환
          2) rollout 실행
          3) gif + total_reward 반환
        """
        env_kwargs_no_layout = copy.deepcopy(env_kwargs)
        layout_name = env_kwargs_no_layout.pop("layout")

        # PPOParams → 실제 callable policy 구조로 변환
        pairing = policy_checkoints_to_policy_pairing(pairing, config)

        # eval_pairing이 rollout 돌리고, frame_seq/total_reward 등을 들고있는 객체 반환
        # ALG_NAME은 config의 최상위 레벨에 위치함 (rnn-e3t.yaml 등에서 @package _global_ 사용)
        alg_name = config.get("ALG_NAME", "PPO")
        # 만약 alg 키 아래에 있다면 (구버전 호환)
        if "alg" in config:
            alg_name = config["alg"].get("ALG_NAME", alg_name)

        return eval_pairing(
            pairing,
            layout_name,
            key,
            env_kwargs=env_kwargs_no_layout,
            num_seeds=num_seeds,
            all_recipes=num_seeds is None,
            no_viz=no_viz,
            algorithm=alg_name,
        )

    # JIT 한 번 감싸두면, 여러 policy에 대해 돌릴 때도 컴파일 한 번만 사용됨
    # _policy_viz_jit = jax.jit(_policy_viz, static_argnames=())

    # ❗ 여기 포인트: no_viz에 따라 jit 여부를 바꾼다
    if no_viz:
        # 프레임 안 만들고 리워드만 계산 → 순수 JAX → jit 가능
        _policy_viz = jax.jit(_policy_viz, static_argnames=())

    # 4) PolicyPairing만 쭉 뽑아서 flat 리스트와 treedef로 변환
    policy_pairings, treedef = jax.tree_util.tree_flatten(
        all_params, is_leaf=lambda x: type(x) is PolicyPairing
    )

    # ❌ 여기서 파라미터를 stack해서 (B, ...)로 만드는 건 금지
    # policy_pairings = jax.tree_util.tree_map(lambda *v: jnp.stack(v), *policy_pairings)
    #
    # → 이렇게 하면 CNN kernel이 (1,1,38,128) → (B,1,1,38,128)이 되어
    #   Flax Conv 초기화 shape (10,1,1,38,128) 같은 expectation과 충돌함.

    # ✅ 대신 PolicyPairing을 하나씩 순차적으로 평가 (파라미터는 항상 scalar tree)
    results = []
    for idx, pairing in enumerate(policy_pairings):
        print(f"[EVAL] Running _policy_viz for policy_pairing {idx+1}/{len(policy_pairings)}")
        viz_result = _policy_viz(pairing)
        results.append(viz_result)

    # 5) 결과들을 원래 트리 구조(dict[run][ckpt] → dict[annotation → viz])로 복원
    all_params = jax.tree_util.tree_unflatten(treedef, results)

    labels = ["run", "checkpoint"]
    if cross:
        labels[1] = "policy_labels"

    # 6) reward summary + gif 저장
    rows = []
    max_agents = 0
    for first_level, first_level_runs in all_params.items():
        for second_level, second_level_runs in first_level_runs.items():
            checkpoint_sum = 0.0
            acc_sum = None
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
                    # pred_acc is (num_agents,)
                    if acc_sum is None:
                        acc_sum = jnp.zeros_like(pred_acc)
                    acc_sum += pred_acc
                    acc_count += 1
                    
                    for i in range(pred_acc.shape[0]):
                        row.append(float(pred_acc[i]))
                    max_agents = max(max_agents, pred_acc.shape[0])
                
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
            
            rows.append(mean_row)

    # 7) CSV로 요약 저장
    summery_name = "reward_summary_cross.csv" if cross else "reward_summary_sp.csv"
    summery_file = run_base_dir / summery_name
    with open(summery_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        fieldnames = [labels[0], labels[1], "annotation", "total_reward"]
        for i in range(max_agents):
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
