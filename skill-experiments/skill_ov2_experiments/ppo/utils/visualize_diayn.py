
import argparse
import copy
import csv
import os
import sys
from pathlib import Path

import imageio
import jax
import jax.numpy as jnp
import numpy as np
import yaml
from flax.training.train_state import TrainState
from jaxmarl.viz.overcooked_v2_visualizer import OvercookedV2Visualizer

from skill_ov2_experiments.eval.evaluate import eval_pairing
from skill_ov2_experiments.eval.rollout import PolicyPairing
from skill_ov2_experiments.ppo.policy import DIAYNPPOPolicy, PPOPolicy, PPOParams
from skill_ov2_experiments.ppo.utils.utils import get_config_file, get_run_dirs, load_train_state
from skill_ov2_experiments.ppo.utils.store import load_all_checkpoints

def visualize_diayn_policy(
    run_base_dir: Path,
    key: jax.random.PRNGKey,
    num_seeds: int = 1,
    final_only: bool = True,
    no_viz: bool = False,
    extra_env_kwargs: dict = None,
    specific_skill: int = None,
):
    """
    DIAYN 정책을 시각화합니다. 각 스킬별로 동작을 확인합니다.
    """
    print(f"Visualizing DIAYN policies in {run_base_dir}")
    
    # 1) PPO 파라미터 로드: all_params: dict[run_id][ckpt_id] -> PPOParams
    # load_all_checkpoints 내부에서 config.yaml을 찾아서 로드함
    # 만약 config.yaml이 없으면 에러가 발생할 수 있지만, 
    # visualize_ppo.py와 동일한 방식으로 로드 시도
    
    try:
        all_params, config, configs = load_all_checkpoints(run_base_dir, final_only=final_only)
    except Exception as e:
        print(f"Failed to load checkpoints using load_all_checkpoints: {e}")
        print("Trying fallback method (manual loading)...")
        # Fallback: 기존 방식 (get_run_dirs 사용)
        # 하지만 config.yaml이 없으면 이 방법도 실패함.
        # 사용자가 config.yaml이 없다고 했으므로, 
        # load_all_checkpoints가 실패하면 정말 방법이 없음.
        # 단, load_all_checkpoints는 .hydra/config.yaml 등도 찾을 수 있음.
        return

    if not all_params:
        print("No valid parameters loaded.")
        return

    # 3) 환경 설정 준비
    # 첫 번째 유효한 config 사용 (load_all_checkpoints가 반환한 config 사용)
    # config는 첫 번째 run의 config임.
    
    base_config = config
    
    print(f"[DEBUG] Config keys: {base_config.keys()}")
    if "env" in base_config:
        print(f"[DEBUG] Config['env'] keys: {base_config['env'].keys()}")

    env_kwargs = base_config.get("ENV_KWARGS", {})
    if not env_kwargs and "env" in base_config:
        env_kwargs = base_config["env"].get("ENV_KWARGS", {})
    
    print(f"[DEBUG] Loaded env_kwargs: {env_kwargs}")

    if extra_env_kwargs:
        env_kwargs.update(extra_env_kwargs)
        
    num_actors = base_config.get("NUM_ACTORS", 2)
    
    # DIAYN 설정 확인
    diayn_config = base_config.get("DIAYN", {})
    num_skills = diayn_config.get("NUM_SKILLS", 0)
    
    if num_skills <= 0:
        print(f"Warning: NUM_SKILLS is {num_skills}. Is this a DIAYN run?")
        # 일반 PPO로 처리하거나 종료
        # 여기서는 강제로 진행하되 skill 0만 있다고 가정할 수도 있지만, 
        # DIAYNPPOPolicy는 num_skills > 0을 기대함.
        if num_skills == 0:
             num_skills = 1 # Fallback
    
    skills_to_viz = range(num_skills)
    if specific_skill is not None:
        if 0 <= specific_skill < num_skills:
            skills_to_viz = [specific_skill]
        else:
            print(f"Invalid specific skill {specific_skill} for num_skills {num_skills}")
            return

    # 4) 시각화 루프
    # 구조: Run -> Checkpoint -> Skill
    
    rows = []
    
    for run_id, run_ckpts in all_params.items():
        # configs 딕셔너리에서 해당 run의 config 가져오기
        # load_all_checkpoints는 configs[run_id]를 반환함
        current_config = configs.get(run_id, base_config)

        # [FIX] Config 구조에 따라 ENV_KWARGS 위치가 다를 수 있음
        # ippo.py 등에서는 config["env"]["ENV_KWARGS"]에 저장됨
        run_env_kwargs = current_config.get("ENV_KWARGS", {})
        if not run_env_kwargs and "env" in current_config:
            run_env_kwargs = current_config["env"].get("ENV_KWARGS", {})
        
        print(f"[DEBUG] Run {run_id} env_kwargs: {run_env_kwargs}")
        
        for ckpt_id, ckpt_data in run_ckpts.items():
            # [FIX] load_all_checkpoints 반환 구조 변경 대응
            if isinstance(ckpt_data, dict) and "policy" in ckpt_data:
                ppo_params = ckpt_data["policy"]
                diayn_state = ckpt_data.get("diayn_state")
            else:
                ppo_params = ckpt_data
                diayn_state = None
            
            for skill_idx in skills_to_viz:
                print(f"Visualizing {run_id}/{ckpt_id} - Skill {skill_idx}")
                
                # PolicyPairing 생성
                # Self-play: 두 에이전트 모두 같은 스킬을 사용하도록 설정
                
                # 환경 설정 (Layout 등)
                env_kwargs_viz = copy.deepcopy(run_env_kwargs)
                if extra_env_kwargs:
                    env_kwargs_viz.update(extra_env_kwargs)

                layout_name = env_kwargs_viz.pop("layout", "cramped_room") # Default fallback
                print(f"[DEBUG] Using layout: {layout_name}")
                
                # 평가 실행
                # JIT 컴파일을 매번 하면 느리므로, skill_idx가 static argument가 되도록 주의해야 함.
                # DIAYNPPOPolicy 내부에 skill_idx가 저장되어 있고, compute_action은 이를 사용.
                # eval_pairing은 policy 객체를 받음.
                # policy 객체는 PyTree가 아닐 수 있음 (DIAYNPPOPolicy는 일반 클래스).
                # 하지만 eval_pairing 내부에서 policy.compute_action을 호출함.
                # JAX 변환을 위해 policy 객체가 JIT 호환되어야 하거나, 
                # eval_pairing이 policy를 static으로 취급해야 함.
                # 기존 eval_pairing 구현을 보면 policy_pairing을 인자로 받음.
                # PolicyPairing은 PyTree로 등록되어 있을 것임 (flax struct 등).
                # 하지만 DIAYNPPOPolicy는 단순 Python 클래스일 가능성이 높음.
                # visualize_ppo.py에서는 closure로 policy를 캡처해서 JIT함.
                
                # 여기서도 closure 방식 사용
                def _viz_impl(pairing_params):
                    # pairing_params는 PPOParams (train_state.params)
                    # 여기서 다시 Policy 객체를 재구성해야 함.
                    # 하지만 eval_pairing은 PolicyPairing 객체를 받음.
                    
                    # visualize_ppo.py의 방식을 따름:
                    # _viz_impl 내부에서 Policy 객체를 생성.
                    
                    inner_policies = []
                    for _ in range(num_actors):
                        inner_policies.append(DIAYNPPOPolicy(
                            params=pairing_params.params, # PPOParams.params
                            config=current_config,
                            skill_idx=skill_idx,
                            num_skills=num_skills
                        ))
                    
                    inner_pairing = PolicyPairing(*inner_policies)
                    
                    return eval_pairing(
                        inner_pairing,
                        layout_name,
                        key,
                        env_kwargs=env_kwargs_viz,
                        num_seeds=1, # [FIX] all_recipes와 충돌 방지를 위해 1로 고정
                        all_recipes=True, # [FIX] OV2 특성상 all_recipes 필수
                        no_viz=True, # [FIX] JIT 내부 렌더링 방지 (OOM 해결)
                        algorithm="PPO" # DIAYN도 PPO 기반
                    )
                
                # JIT 실행
                # ppo_params는 PPOParams 객체.
                viz_result = jax.jit(_viz_impl)(ppo_params)
                
                # 결과 저장
                # viz_result는 dict {annotation: PolicyVizualization} 입니다.
                
                viz_dir = run_base_dir / run_id / ckpt_id / f"skill_{skill_idx}"
                
                if not no_viz:
                    os.makedirs(viz_dir, exist_ok=True)
                    viz = OvercookedV2Visualizer()
                    agent_view_size = env_kwargs_viz.get("agent_view_size", None)
                    
                    for annotation, res in viz_result.items():
                        viz_filename = viz_dir / f"{annotation}.gif"
                        
                        # JIT 밖에서 렌더링 수행
                        if res.state_seq is not None:
                            print(f"\tRendering {annotation}...")
                            # state_seq는 JAX array이므로 CPU로 이동하거나 그대로 사용
                            # animate 메서드는 내부적으로 vmap을 사용하지만, 
                            # 여기서는 JIT 밖이므로 메모리 관리가 더 유연할 수 있음.
                            # 하지만 안전을 위해 순차적으로 렌더링하거나 배치 처리를 하는 것이 좋음.
                            # OvercookedV2Visualizer.animate는 vmap을 사용함.
                            # 메모리 부족이 여전하다면 순차 처리로 변경해야 함.
                            
                            # 순차 렌더링으로 구현
                            frames = []
                            state_seq = res.state_seq
                            # state_seq는 PyTree. 길이를 구해야 함.
                            seq_len = jax.tree_util.tree_leaves(state_seq)[0].shape[0]
                            
                            for t in range(seq_len):
                                state_t = jax.tree_util.tree_map(lambda x: x[t], state_seq)
                                frame = viz._render_state(state_t, agent_view_size)
                                frames.append(np.array(frame))
                            
                            imageio.mimsave(viz_filename, frames, "GIF", duration=0.5)
                        
                        row = [run_id, ckpt_id, skill_idx, annotation, res.total_reward]
                        rows.append(row)
                        print(f"\tSkill {skill_idx} - {annotation}: {res.total_reward}")
                else:
                    # no_viz라도 reward는 기록
                    for annotation, res in viz_result.items():
                        row = [run_id, ckpt_id, skill_idx, annotation, res.total_reward]
                        rows.append(row)
                        print(f"\tSkill {skill_idx} - {annotation}: {res.total_reward}")

    # CSV 저장
    summary_file = run_base_dir / "diayn_reward_summary.csv"
    with open(summary_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["run_id", "ckpt_id", "skill_idx", "annotation", "total_reward"])
        for row in rows:
            writer.writerow(row)
            
    print(f"Summary written to {summary_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=str, required=True, help="Base directory of the run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--all_ckpt", action="store_true", help="Visualize all checkpoints")
    parser.add_argument("--no_viz", action="store_true", help="Do not generate GIFs")
    parser.add_argument("--skill", type=int, default=None, help="Specific skill index to visualize")
    parser.add_argument("--no_reset", action="store_true")

    args = parser.parse_args()

    directory = Path(args.d)
    final_only = not args.all_ckpt
    
    key = jax.random.PRNGKey(args.seed)
    
    extra_env_kwargs = {}
    if args.no_reset:
        extra_env_kwargs["random_reset"] = False
        extra_env_kwargs["op_ingredient_permutations"] = False

    visualize_diayn_policy(
        directory,
        key,
        num_seeds=args.num_seeds,
        final_only=final_only,
        no_viz=args.no_viz,
        extra_env_kwargs=extra_env_kwargs,
        specific_skill=args.skill
    )
