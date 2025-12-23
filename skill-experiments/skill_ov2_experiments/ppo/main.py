from pathlib import Path
import hydra
import sys
import os
import jax
import jax.numpy as jnp
from omegaconf import OmegaConf
import wandb

from skill_ov2_experiments.ppo.policy import PPOParams
from skill_ov2_experiments.ppo.utils.store import load_all_checkpoints, store_checkpoint
from skill_ov2_experiments.ppo.state_sample_run import state_sample_run
from skill_ov2_experiments.ppo.run import single_run
from skill_ov2_experiments.ppo.tune import tune
from skill_ov2_experiments.ppo.utils.utils import get_run_base_dir
from skill_ov2_experiments.ppo.utils.visualize_ppo import visualize_ppo_policy

# ----------------------------------------------------------------------
# 모듈 import 시점 디버그
# ----------------------------------------------------------------------
print("[MAINDBG] =========================")
print("[MAINDBG] skill_ov2_experiments.ppo.main IMPORTED")
print(f"[MAINDBG] __file__ = {__file__}")
print("[MAINDBG] =========================")

DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(DIR))
# 현재 디렉토리의 상위 디렉토리(skill-experiments)를 sys.path의 맨 앞에 추가하여
# 설치된 패키지보다 현재 폴더의 코드를 우선적으로 로드하도록 함
sys.path.insert(0, os.path.dirname(DIR))

# NaN 디버그 옵션
jax.config.update("jax_debug_nans", True)


def single_run_with_viz(config):
    # Hydra Config -> 일반 dict
    print("[RUNDBG] ===== single_run_with_viz CALLED =====")
    print(f"[RUNDBG] raw config type = {type(config)}")
    config = OmegaConf.to_container(config)
    print("[RUNDBG] OmegaConf.to_container 완료")
    print(f"[RUNDBG] top-level config keys = {list(config.keys())}")

    num_checkpoints = config.get("NUM_CHECKPOINTS", 0)
    print(f"[RUNDBG] NUM_CHECKPOINTS = {num_checkpoints}")

    # 이름 구성: 모델/레이아웃 정보는 항상 준비하고, CLI에서 wandb.name을 넘기면 이를 우선 사용 (예: rnn-sp-uc)
    model_name = config["model"]["TYPE"]
    layout_name = config["env"]["ENV_KWARGS"]["layout"]
    agent_view_size = config["env"]["ENV_KWARGS"].get("agent_view_size", None)
    avs_str = f"avs-{agent_view_size}" if agent_view_size is not None else "avs-full"

    cli_run_name = config.get("wandb", {}).get("name")
    if cli_run_name:
        run_name = cli_run_name
    else:
        # utils.py의 _infer_run_suffix 함수를 사용해서 일관성 있게 suffix 결정
        from skill_ov2_experiments.ppo.utils.utils import _infer_run_suffix
        suffix = _infer_run_suffix(config)
        run_name = f"{suffix}_{layout_name}_{model_name.lower()}_{avs_str}"

    if "FCP" in config:
        population_dir = Path(config["FCP"])
        run_name = f"fcp_{population_dir.name}_seed_{config['SEED']}"

    print(f"[RUNDBG] run_name = {run_name}")
    print("[RUNDBG] >>> wandb.init 진입 직전")

    with wandb.init(
        entity=config["wandb"]["ENTITY"],
        project=config["wandb"]["PROJECT"],
        tags=["IPPO", model_name, "OvercookedV2"],
        config=config,
        mode=config["wandb"]["WANDB_MODE"],
        name=run_name,
    ) as run:
        print("[RUNDBG] <<< wandb.init 성공 / context 진입")
        run_id = run.id
        print(f"[RUNDBG] wandb run_id = {run_id}")

        run_base_dir = get_run_base_dir(run_id, config)
        print(f"[RUNDBG] run_base_dir = {run_base_dir}")
        config["RUN_BASE_DIR"] = run_base_dir

        # NUM_CHECKPOINTS 최종 값 재확인
        print(f"[RUNDBG] (wandb.init 블록 내부) NUM_CHECKPOINTS={config.get('NUM_CHECKPOINTS')}")

        # 실제 학습 함수 진입
        print("[RUNDBG] >>> single_run(config) 호출")
        out = single_run(config)
        print("[RUNDBG] <<< single_run(config) 종료")

    # ---------------------------
    # 체크포인트 디버그/저장
    # ---------------------------
    print(f"[CKPTDBG] NUM_CHECKPOINTS after run = {config.get('NUM_CHECKPOINTS')}")
    if config.get("NUM_CHECKPOINTS", 0) > 0:
        print("[CKPTDBG] 체크포인트 버퍼에서 파라미터 추출 시작")
        checkpoints = out["runner_state"][1]
        sample_leaf = jax.tree_util.tree_leaves(checkpoints)[0]

        num_checkpoints = config["NUM_CHECKPOINTS"]
        # 러프하게 run 축 크기 추정: 첫 leaf가 (num_runs, num_checkpoints, ... ) 또는 (num_checkpoints, ...)
        if sample_leaf.shape[0] == num_checkpoints and not (
            sample_leaf.ndim >= 2 and sample_leaf.shape[1] == num_checkpoints
        ):
            num_runs = 1
        else:
            # 다차원인 경우 첫 차원을 run 갯수로 가정
            num_runs = sample_leaf.shape[0]

        print(
            f"[CKPTDBG] 전체 체크포인트 버퍼 구조 예시 leaf shape={sample_leaf.shape}; "
            f"num_runs={num_runs}; num_checkpoints={num_checkpoints}"
        )

        def params_for(run_num, ck_idx):
            def _sel(x):
                arr = x
                # (num_runs, num_checkpoints, ...)
                if arr.ndim >= 2 and arr.shape[1] == num_checkpoints:
                    return arr[run_num][ck_idx]
                # (num_checkpoints, ...)
                if arr.shape[0] == num_checkpoints:
                    return arr[ck_idx]
                # fallback (희귀 케이스): 동일 가정
                return arr[run_num][ck_idx]

            return jax.tree_util.tree_map(_sel, checkpoints)

        def summarize_params(pytree):
            leaves = jax.tree_util.tree_leaves(pytree)
            if not leaves:
                return {"num_leaves": 0}
            first = leaves[0]
            stats = {
                "num_leaves": len(leaves),
                "first_shape": first.shape,
                "first_mean": float(jnp.mean(first)),
                "first_std": float(jnp.std(first)),
                "first_abs_mean": float(jnp.mean(jnp.abs(first))),
                "first_head": first.flatten()[:5].tolist(),
                "all_zero_first": bool(jnp.all(first == 0.0)),
            }
            return stats

        # 디버그: 각 체크포인트 슬롯 내용 요약 출력
        for run_num in range(num_runs):
            print(f"[CKPTDBG] ==== run {run_num} 체크포인트 메모리 요약 ====")
            for ck_idx in range(num_checkpoints):
                params_ck = params_for(run_num, ck_idx)
                stats = summarize_params(params_ck)
                print(
                    f"[CKPTDBG] run={run_num} ckpt_index={ck_idx} stats="
                    f"num_leaves={stats['num_leaves']} first_shape={stats.get('first_shape')} "
                    f"mean={stats.get('first_mean'):.4e} std={stats.get('first_std'):.4e} "
                    f"abs_mean={stats.get('first_abs_mean'):.4e} "
                    f"all_zero_first={stats.get('all_zero_first')} "
                    f"head={stats.get('first_head')}"
                )

        # 디스크 저장 (마지막 숫자 체크포인트는 중복이라 생략 -> ckpt_final만 유지)
        for run_num in range(num_runs):
            run_dir = config["RUN_BASE_DIR"] / f"run_{run_num}"
            print(f"[CKPTDBG] ==== 디스크 저장 시작: run {run_num} ====")

            # 마지막 인덱스를 제외한 숫자 체크포인트 저장
            save_count = max(num_checkpoints - 1, 0)
            for ck_idx in range(save_count):
                params_ck = params_for(run_num, ck_idx)
                print(f"[CKPTDBG] store ckpt_{ck_idx} (index {ck_idx})")
                
                # [DIAYN] params_ck 구조 분해 (ippo.py 수정에 따라 dict 형태 예상)
                if isinstance(params_ck, dict) and "params" in params_ck:
                    p_params = params_ck["params"]
                    p_diayn = params_ck.get("diayn_state", None)
                else:
                    p_params = params_ck
                    p_diayn = None
                
                store_checkpoint(config, p_params, run_num, ck_idx, final=False, diayn_state=p_diayn)

            # 마지막 인덱스 파라미터를 최종본으로만 저장 (예: 기존 ckpt_2 제거 목적)
            last_idx = num_checkpoints - 1
            if num_checkpoints > 0:
                params_last = params_for(run_num, last_idx)
                print(f"[CKPTDBG] store ckpt_final (from index {last_idx})")
                
                if isinstance(params_last, dict) and "params" in params_last:
                    p_params = params_last["params"]
                    p_diayn = params_last.get("diayn_state", None)
                else:
                    p_params = params_last
                    p_diayn = None
                
                store_checkpoint(config, p_params, run_num, last_idx, final=True, diayn_state=p_diayn)

            # 저장 후 실제 생성된 디렉토리 나열
            if run_dir.exists():
                produced = sorted(
                    [p.name for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("ckpt_")]
                )
                print(f"[CKPTDBG] run={run_num} 생성된 디렉토리: {produced}")
    else:
        print("[CKPTDBG] NUM_CHECKPOINTS == 0, 체크포인트 저장 스킵")

    # ---------------------------
    # 시각화
    # ---------------------------
    if config.get("VISUALIZE", False):
        print("[VIZDBG] VISUALIZE=True → visualize_ppo_policy 호출 (final_only=True, num_seeds=2)")
        visualize_ppo_policy(
            run_base_dir,
            key=jax.random.PRNGKey(config["SEED"]),
            final_only=True,
            num_seeds=2,
        )

        print("[VIZDBG] cross-play 평가 호출 (final_only=True, num_seeds=500, cross=True, no_viz=True)")
        visualize_ppo_policy(
            run_base_dir,
            key=jax.random.PRNGKey(config["SEED"]),
            final_only=True,
            num_seeds=500,
            cross=True,
            no_viz=True,
        )

    print("[RUNDBG] ===== single_run_with_viz 종료 =====")


@hydra.main(version_base=None, config_path="config", config_name="base")
def main(config):
    print("===================================================")
    print("[MAINDBG] main() 진입")
    print(f"[MAINDBG] config type = {type(config)}")
    # top-level 키만 간단히
    try:
        print(f"[MAINDBG] top-level config keys = {list(config.keys())}")
    except Exception as e:
        print(f"[MAINDBG] config keys 출력 중 에러: {e}")

    # 분기 로직 확인
    is_tune = bool(config.get("TUNE", False))
    has_num_iterations = "NUM_ITERATIONS" in config

    print(f"[MAINDBG] TUNE = {is_tune}")
    print(f"[MAINDBG] 'NUM_ITERATIONS' in config = {has_num_iterations}")

    if is_tune:
        print("[MAINDBG] → tune(config) 분기 선택")
        tune(config)
    elif has_num_iterations:
        print("[MAINDBG] → state_sample_run(config) 분기 선택")
        state_sample_run(config)
    else:
        print("[MAINDBG] → single_run_with_viz(config) 분기 선택")
        single_run_with_viz(config)

    print("[MAINDBG] main() 종료")
    print("===================================================")


if __name__ == "__main__":
    print("[MAINDBG] __main__ 진입 (python -m / 직접 실행)")
    main()
