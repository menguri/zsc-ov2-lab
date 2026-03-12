from pathlib import Path
import hydra
import sys
import os
import json
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import wandb

from overcooked_v2_experiments.ppo.policy import PPOParams
from overcooked_v2_experiments.ppo.utils.store import load_all_checkpoints, store_checkpoint
from overcooked_v2_experiments.ppo.state_sample_run import state_sample_run
from overcooked_v2_experiments.ppo.run import single_run
from overcooked_v2_experiments.ppo.tune import tune
from overcooked_v2_experiments.ppo.utils.utils import get_run_base_dir
from overcooked_v2_experiments.ppo.utils.visualize_ppo import visualize_ppo_policy

# ----------------------------------------------------------------------
# 모듈 import 시점 디버그
# ----------------------------------------------------------------------
print("[MAINDBG] =========================")
print("[MAINDBG] overcooked_v2_experiments.ppo.main IMPORTED")
print(f"[MAINDBG] __file__ = {__file__}")
print("[MAINDBG] =========================")

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(DIR))

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
        from overcooked_v2_experiments.ppo.utils.utils import _infer_run_suffix
        suffix = _infer_run_suffix(config)
        run_name = f"{suffix}_{layout_name}_{model_name.lower()}_{avs_str}"

    if "FCP" in config:
        population_dir = Path(config["FCP"])
        run_name = f"fcp_{population_dir.name}_seed_{config['SEED']}"

    wandb_cfg = config.get("wandb", {})
    extra_tags = wandb_cfg.get("tags", [])
    if isinstance(extra_tags, str):
        extra_tags = [extra_tags]
    elif not isinstance(extra_tags, (list, tuple)):
        extra_tags = []

    old_overcooked_enabled = bool(config.get("OLD_OVERCOOKED", False))
    if isinstance(config.get("alg"), dict):
        old_overcooked_enabled = old_overcooked_enabled or bool(
            config["alg"].get("OLD_OVERCOOKED", False)
        )
    if old_overcooked_enabled and "old_overcooked" not in str(run_name).lower():
        run_name = f"{run_name}_old_overcooked"
    if old_overcooked_enabled:
        extra_tags = list(extra_tags) + ["old_overcooked"]

    wandb_tags = []
    for tag in ["IPPO", model_name, "OvercookedV2", *extra_tags]:
        tag_str = str(tag).strip()
        if tag_str and tag_str not in wandb_tags:
            wandb_tags.append(tag_str)

    print(f"[RUNDBG] run_name = {run_name}")
    print(f"[RUNDBG] wandb_tags = {wandb_tags}")
    print("[RUNDBG] >>> wandb.init 진입 직전")

    with wandb.init(
        entity=config["wandb"]["ENTITY"],
        project=config["wandb"]["PROJECT"],
        tags=wandb_tags,
        config=config,
        mode=config["wandb"]["WANDB_MODE"],
        name=run_name,
    ) as run:
        print("[RUNDBG] <<< wandb.init 성공 / context 진입")
        run_id = run.id
        print(f"[RUNDBG] wandb run_id = {run_id}")

        run_base_dir = Path(get_run_base_dir(run_id, config)).resolve()
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
    alg_name = str(config.get("ALG_NAME", "")).upper()
    print(f"[CKPTDBG] NUM_CHECKPOINTS after run(cfg) = {config.get('NUM_CHECKPOINTS')}")
    checkpoints = out["runner_state"][1]
    sample_leaf = np.asarray(jax.device_get(jax.tree_util.tree_leaves(checkpoints)[0]))
    if sample_leaf.ndim >= 2:
        num_runs = int(sample_leaf.shape[0])
        num_checkpoints = int(sample_leaf.shape[1])
    elif sample_leaf.ndim == 1:
        num_runs = 1
        num_checkpoints = int(sample_leaf.shape[0])
    else:
        num_runs = 1
        num_checkpoints = 0
    is_ph2_flag = out.get("is_ph2_dual", None)
    if is_ph2_flag is None:
        is_ph2 = ("PH2" in alg_name) and (len(out["runner_state"]) >= 5)
    else:
        is_ph2 = bool(np.asarray(jax.device_get(is_ph2_flag)).reshape(-1)[0])
    ind_checkpoints = out["runner_state"][-4] if is_ph2 else None

    if num_checkpoints > 0:
        print("[CKPTDBG] 체크포인트 버퍼에서 파라미터 추출 시작")

        print(
            f"[CKPTDBG] 전체 체크포인트 버퍼 구조 예시 leaf shape={sample_leaf.shape}; "
            f"num_runs={num_runs}; num_checkpoints={num_checkpoints}"
        )

        def params_for(run_num, ck_idx):
            def _sel(x):
                arr = np.asarray(jax.device_get(x))
                # (num_runs, num_checkpoints, ...)
                if arr.ndim >= 2 and arr.shape[1] == num_checkpoints:
                    return arr[run_num, ck_idx]
                # (num_checkpoints, ...)
                if arr.shape[0] == num_checkpoints:
                    return arr[ck_idx]
                # fallback (희귀 케이스): 동일 가정
                return arr[run_num, ck_idx]

            return jax.tree_util.tree_map(_sel, checkpoints)

        def ind_params_for(run_num, ck_idx):
            if (not is_ph2) or (ind_checkpoints is None):
                return None

            def _sel(x):
                arr = np.asarray(jax.device_get(x))
                if arr.ndim >= 2 and arr.shape[1] == num_checkpoints:
                    return arr[run_num, ck_idx]
                if arr.shape[0] == num_checkpoints:
                    return arr[ck_idx]
                return arr[run_num, ck_idx]

            return jax.tree_util.tree_map(_sel, ind_checkpoints)

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

        # 디스크 저장(run_x): 기본은 ckpt_0 + ckpt_final만 유지.
        # eval 스냅샷은 별도(eval/)로 전체 cadence를 보존한다.
        keep_first_final_only = bool(config.get("RUN_KEEP_FIRST_FINAL_ONLY", True))
        if num_checkpoints <= 0:
            disk_ckpt_src_indices = []
        elif keep_first_final_only:
            # 체크포인트가 1개뿐이어도 ckpt_0과 ckpt_final을 둘 다 남길 수 있게 같은 src를 사용한다.
            if int(num_checkpoints) == 1:
                disk_ckpt_src_indices = [0, 0]
            else:
                disk_ckpt_src_indices = [0, int(num_checkpoints - 1)]
        else:
            # Backward-compatible path: legacy budget-based saving.
            disk_ckpt_budget = int(config.get("NUM_CHECKPOINTS", num_checkpoints))
            if disk_ckpt_budget <= 0:
                disk_ckpt_src_indices = [int(num_checkpoints - 1)]
            elif num_checkpoints <= disk_ckpt_budget:
                disk_ckpt_src_indices = list(range(int(num_checkpoints)))
            else:
                disk_ckpt_src_indices = np.linspace(
                    0,
                    int(num_checkpoints - 1),
                    int(disk_ckpt_budget),
                    endpoint=True,
                    dtype=np.int32,
                ).tolist()

        print(
            f"[CKPTDBG] run_x disk policy: keep_first_final_only={keep_first_final_only} "
            f"src_indices={disk_ckpt_src_indices}",
            flush=True,
        )

        for run_num in range(num_runs):
            run_dir = Path(config["RUN_BASE_DIR"]).resolve() / f"run_{run_num}"
            print(f"[CKPTDBG] ==== 디스크 저장 시작: run {run_num} ====")

            # 마지막 인덱스를 제외한 숫자 체크포인트 저장 (ckpt_0, ckpt_1, ...)
            save_count = max(len(disk_ckpt_src_indices) - 1, 0)
            for out_ck_idx, src_ck_idx in enumerate(disk_ckpt_src_indices[:save_count]):
                params_ck = params_for(run_num, int(src_ck_idx))
                print(f"[CKPTDBG] store ckpt_{out_ck_idx} (src index {int(src_ck_idx)})")
                if is_ph2:
                    ind_params_ck = ind_params_for(run_num, int(src_ck_idx))
                    store_checkpoint(
                        config,
                        params_ck,
                        run_num,
                        int(out_ck_idx),
                        final=False,
                        params_spec=params_ck,
                        params_ind=ind_params_ck,
                    )
                else:
                    store_checkpoint(config, params_ck, run_num, int(out_ck_idx), final=False)

            # 마지막 선택 인덱스 파라미터를 최종본으로 저장
            if len(disk_ckpt_src_indices) > 0:
                last_src_idx = int(disk_ckpt_src_indices[-1])
                params_last = params_for(run_num, last_src_idx)
                print(f"[CKPTDBG] store ckpt_final (src index {last_src_idx})")
                if is_ph2:
                    ind_params_last = ind_params_for(run_num, last_src_idx)
                    store_checkpoint(
                        config,
                        params_last,
                        run_num,
                        last_src_idx,
                        final=True,
                        params_spec=params_last,
                        params_ind=ind_params_last,
                    )
                else:
                    store_checkpoint(config, params_last, run_num, last_src_idx, final=True)

            # 저장 후 실제 생성된 디렉토리 나열
            if run_dir.exists():
                produced = sorted(
                    [p.name for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("ckpt_")]
                )
                print(f"[CKPTDBG] run={run_num} 생성된 디렉토리: {produced}")

        # Offline eval snapshots are exported once after training from in-memory checkpoint buffers.
        try:
            from flax.training import orbax_utils
            import orbax.checkpoint as ocp
        except Exception:
            orbax_utils = None
            ocp = None

        if orbax_utils is not None and ocp is not None:
            try:
                num_steps = int(config["model"]["NUM_STEPS"])
                num_envs = int(config["model"]["NUM_ENVS"])
                steps_per_update = num_steps * num_envs
                total_timesteps = int(config["model"]["TOTAL_TIMESTEPS"])
                if "NUM_UPDATES" in config["model"]:
                    num_updates = int(config["model"]["NUM_UPDATES"])
                else:
                    num_updates = max(1, total_timesteps // max(1, steps_per_update))
                ckpt_every_env_steps = int(config.get("PH1_EVAL_EVERY_ENV_STEPS", 0))
                cfg_num_checkpoints = int(config.get("NUM_CHECKPOINTS", num_checkpoints))
                use_env_step_ckpt_schedule = (
                    ckpt_every_env_steps > 0 and int(num_checkpoints) != int(cfg_num_checkpoints)
                )
                if use_env_step_ckpt_schedule:
                    target_env_steps = np.arange(
                        0,
                        total_timesteps + ckpt_every_env_steps,
                        ckpt_every_env_steps,
                        dtype=np.int64,
                    )
                    if target_env_steps[-1] != total_timesteps:
                        target_env_steps = np.append(target_env_steps, total_timesteps)
                    pair_updates = []
                    pair_env_steps = []
                    seen = set()
                    for env_step_t in target_env_steps:
                        upd_t = int(
                            np.clip(
                                np.ceil(env_step_t / max(1, steps_per_update)),
                                0,
                                num_updates,
                            )
                        )
                        if upd_t in seen:
                            continue
                        seen.add(upd_t)
                        pair_updates.append(upd_t)
                        pair_env_steps.append(int(env_step_t))
                    ckpt_updates = np.asarray(pair_updates, dtype=np.int32)
                    ckpt_env_steps = np.asarray(pair_env_steps, dtype=np.int64)
                else:
                    ckpt_updates = np.linspace(
                        0,
                        num_updates,
                        int(num_checkpoints),
                        endpoint=True,
                        dtype=np.int32,
                    )
                    if int(num_checkpoints) > 0:
                        ckpt_updates[-1] = np.int32(num_updates)
                    ckpt_env_steps = ckpt_updates.astype(np.int64) * int(steps_per_update)

                if ckpt_updates.shape[0] < int(num_checkpoints):
                    pad_n = int(num_checkpoints) - int(ckpt_updates.shape[0])
                    ckpt_updates = np.concatenate(
                        [ckpt_updates, np.full((pad_n,), int(num_updates), dtype=np.int32)],
                        axis=0,
                    )
                    ckpt_env_steps = np.concatenate(
                        [ckpt_env_steps, np.full((pad_n,), int(total_timesteps), dtype=np.int64)],
                        axis=0,
                    )
                elif ckpt_updates.shape[0] > int(num_checkpoints):
                    ckpt_updates = ckpt_updates[: int(num_checkpoints)]
                    ckpt_env_steps = ckpt_env_steps[: int(num_checkpoints)]

                eval_root = Path(config["RUN_BASE_DIR"]).resolve() / "eval"
                eval_root.mkdir(parents=True, exist_ok=True)
                orbax_checkpointer = ocp.PyTreeCheckpointer()

                def _plain(v):
                    if isinstance(v, dict):
                        return {str(k): _plain(x) for k, x in v.items()}
                    if isinstance(v, (list, tuple)):
                        return [_plain(x) for x in v]
                    if isinstance(v, np.ndarray):
                        return v.tolist()
                    if hasattr(v, "item"):
                        try:
                            return v.item()
                        except Exception:
                            pass
                    if isinstance(v, (str, int, float, bool)) or v is None:
                        return v
                    try:
                        return os.fspath(v)
                    except Exception:
                        return str(v)

                is_ph2_flag = out.get("is_ph2_dual", None)
                if is_ph2_flag is None:
                    is_ph2 = ("PH2" in alg_name) and (len(out["runner_state"]) >= 5)
                else:
                    is_ph2 = bool(np.asarray(jax.device_get(is_ph2_flag)).reshape(-1)[0])
                ind_checkpoints = out["runner_state"][-4] if is_ph2 else None
                recent_ckpts = np.asarray(out["runner_state"][-2]) if is_ph2 else None
                recent_has = np.asarray(out["runner_state"][-1]).astype(np.bool_) if is_ph2 else None
                ph1_pool_final = None
                if not is_ph2 and len(out["runner_state"]) >= 25:
                    try:
                        recent_ckpts = np.asarray(jax.device_get(out["runner_state"][23]))
                        recent_has = np.asarray(jax.device_get(out["runner_state"][24])).astype(np.bool_)
                    except Exception:
                        recent_ckpts = None
                        recent_has = None

                def ind_params_for(run_num, ck_idx):
                    def _sel(x):
                        arr = np.asarray(jax.device_get(x))
                        if arr.ndim >= 2 and arr.shape[1] == num_checkpoints:
                            return arr[run_num, ck_idx]
                        if arr.shape[0] == num_checkpoints:
                            return arr[ck_idx]
                        return arr[run_num, ck_idx]
                    return jax.tree_util.tree_map(_sel, ind_checkpoints)

                def recent_for(run_num, ck_idx):
                    if (recent_ckpts is None) or (recent_has is None):
                        nonlocal ph1_pool_final
                        if ph1_pool_final is None:
                            try:
                                ph1_pool_final = np.asarray(jax.device_get(out["runner_state"][20]))
                            except Exception:
                                ph1_pool_final = None
                        if ph1_pool_final is None:
                            return np.zeros((0,), dtype=np.float32), False
                        pf = ph1_pool_final
                        if pf.ndim >= 6:
                            pool_run = pf[run_num]
                        elif pf.ndim >= 5:
                            pool_run = pf
                        else:
                            return np.zeros((0,), dtype=np.float32), False
                        if pool_run.shape[0] <= 0:
                            return np.zeros((0,), dtype=np.float32), False
                        env0 = pool_run[0]  # (pool, H, W, C)
                        if env0.ndim < 4 or env0.shape[0] <= 0:
                            return np.zeros((0,), dtype=np.float32), False
                        mask = np.any(env0 >= 0.0, axis=(1, 2, 3))
                        if not bool(np.any(mask)):
                            return np.zeros((0,), dtype=np.float32), False
                        idx = int(np.where(mask)[0][-1])
                        return np.asarray(env0[idx]).astype(np.float32), True
                    rc = recent_ckpts
                    rh = recent_has
                    if rc.ndim >= 5 and rc.shape[1] == num_checkpoints:
                        rcv = rc[run_num][ck_idx]
                    elif rc.ndim >= 4 and rc.shape[0] == num_checkpoints:
                        rcv = rc[ck_idx]
                    else:
                        rcv = rc[run_num][ck_idx]
                    if rh.ndim >= 2 and rh.shape[1] == num_checkpoints:
                        rhv = bool(np.asarray(rh[run_num][ck_idx]).reshape(-1)[0])
                    elif rh.ndim >= 1 and rh.shape[0] == num_checkpoints:
                        rhv = bool(np.asarray(rh[ck_idx]).reshape(-1)[0])
                    else:
                        rhv = bool(np.asarray(rh[run_num][ck_idx]).reshape(-1)[0])
                    return np.asarray(rcv).astype(np.float32), rhv

                def existing_recent_for(step_dir: Path):
                    tildes_path = step_dir / "mode_tildes.npz"
                    if not tildes_path.exists():
                        return np.zeros((0,), dtype=np.float32), False
                    try:
                        with np.load(str(tildes_path)) as data:
                            has_recent = bool(np.asarray(data.get("has_recent", np.array([0]))).reshape(-1)[0])
                            if not has_recent or ("recent_tilde" not in data):
                                return np.zeros((0,), dtype=np.float32), False
                            return np.asarray(data["recent_tilde"]).astype(np.float32), True
                    except Exception:
                        return np.zeros((0,), dtype=np.float32), False

                for run_num in range(num_runs):
                    seed_local = int(config["SEED"]) + int(run_num)
                    seed_dir = eval_root / f"run_{run_num}"
                    seed_dir.mkdir(parents=True, exist_ok=True)
                    last_item = None
                    last_meta = None
                    last_step_env = 0
                    for ck_idx in range(int(num_checkpoints)):
                        spec_params = params_for(run_num, ck_idx)
                        upd = int(ckpt_updates[ck_idx])
                        step_env = int(ckpt_env_steps[ck_idx])
                        ckpt_name = f"ckpt_{step_env}"
                        step_dir = seed_dir / ckpt_name
                        step_dir.mkdir(parents=True, exist_ok=True)

                        recent_tilde, has_recent = recent_for(run_num, ck_idx)
                        if not has_recent:
                            existing_recent, existing_has = existing_recent_for(step_dir)
                            if existing_has:
                                recent_tilde, has_recent = existing_recent, True
                        np.savez_compressed(
                            step_dir / "mode_tildes.npz",
                            has_recent=np.array([1 if has_recent else 0], dtype=np.uint8),
                            has_random=np.array([0], dtype=np.uint8),
                            recent_tilde=recent_tilde if has_recent else np.zeros((0,), dtype=np.float32),
                            random_tilde=np.zeros((0,), dtype=np.float32),
                        )

                        meta = {
                            "run_num": int(run_num),
                            "seed": int(seed_local),
                            "update_step": int(upd),
                            "env_step": int(step_env),
                            "wandb_run_id": str(run_id),
                            "wandb_run_name": str(run_name),
                            "wandb_project": str(config["wandb"]["PROJECT"]),
                            "wandb_entity": str(config["wandb"]["ENTITY"]),
                        }
                        with open(step_dir / "metadata.json", "w", encoding="utf-8") as f:
                            json.dump(meta, f, ensure_ascii=False, indent=2)

                        item = {
                            "params": spec_params,
                            "config": _plain(config),
                            "update_step": np.int32(upd),
                            "seed": np.int32(seed_local),
                        }
                        if is_ph2:
                            ind_params = ind_params_for(run_num, ck_idx)
                            item["params_spec"] = spec_params
                            item["params_ind"] = ind_params

                        ckpt_dir = step_dir / "model_ckpt"
                        if ckpt_dir.exists():
                            import shutil as _shutil
                            _shutil.rmtree(ckpt_dir, ignore_errors=True)
                        save_args = orbax_utils.save_args_from_target(item)
                        orbax_checkpointer.save(str(ckpt_dir.resolve()), item, save_args=save_args)
                        last_item = item
                        last_meta = meta
                        last_step_env = step_env

                    if last_item is not None:
                        final_dir = seed_dir / "ckpt_final"
                        final_dir.mkdir(parents=True, exist_ok=True)
                        with open(final_dir / "metadata.json", "w", encoding="utf-8") as f:
                            json.dump(last_meta, f, ensure_ascii=False, indent=2)
                        if (seed_dir / f"ckpt_{last_step_env}" / "mode_tildes.npz").exists():
                            import shutil as _shutil
                            _shutil.copy2(
                                seed_dir / f"ckpt_{last_step_env}" / "mode_tildes.npz",
                                final_dir / "mode_tildes.npz",
                            )
                        final_ckpt_dir = final_dir / "model_ckpt"
                        if final_ckpt_dir.exists():
                            import shutil as _shutil
                            _shutil.rmtree(final_ckpt_dir, ignore_errors=True)
                        save_args = orbax_utils.save_args_from_target(last_item)
                        orbax_checkpointer.save(str(final_ckpt_dir.resolve()), last_item, save_args=save_args)
            except Exception as e:
                print(f"[CKPTDBG][ERROR] eval snapshot export failed: {type(e).__name__}: {e}")
                raise RuntimeError(f"eval snapshot export failed: {type(e).__name__}: {e}") from e
    else:
        print("[CKPTDBG] NUM_CHECKPOINTS == 0, 체크포인트 저장 스킵")
        if "PH2" in alg_name and len(out["runner_state"]) >= 5:
            try:
                from flax.training import orbax_utils
                import orbax.checkpoint as ocp

                eval_root = Path(config["RUN_BASE_DIR"]).resolve() / "eval"
                eval_root.mkdir(parents=True, exist_ok=True)
                steps_per_update = int(config["model"]["NUM_STEPS"]) * int(config["model"]["NUM_ENVS"])
                if "NUM_UPDATES" in config["model"]:
                    upd = int(config["model"]["NUM_UPDATES"])
                else:
                    upd = max(1, int(config["model"]["TOTAL_TIMESTEPS"]) // max(1, steps_per_update))
                step_env = int(upd * steps_per_update)
                seed_local = int(config["SEED"])
                step_dir = eval_root / "run_0" / "ckpt_final"
                step_dir.mkdir(parents=True, exist_ok=True)

                meta = {
                    "run_num": 0,
                    "seed": int(seed_local),
                    "update_step": int(upd),
                    "env_step": int(step_env),
                    "wandb_run_id": str(run_id),
                    "wandb_run_name": str(run_name),
                    "wandb_project": str(config["wandb"]["PROJECT"]),
                    "wandb_entity": str(config["wandb"]["ENTITY"]),
                }
                with open(step_dir / "metadata.json", "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)

                # Preserve recent tilde for PH2 final eval snapshot even when NUM_CHECKPOINTS=0.
                recent_tilde = np.zeros((0,), dtype=np.float32)
                has_recent = False
                try:
                    recent_ckpts = np.asarray(jax.device_get(out["runner_state"][-2]))
                    recent_has = np.asarray(jax.device_get(out["runner_state"][-1])).astype(np.bool_)
                    if (recent_ckpts is not None) and (recent_has is not None):
                        if recent_ckpts.ndim >= 2 and recent_ckpts.shape[1] > 0:
                            # shape: (num_runs, num_ckpt, ...)
                            rh = bool(np.asarray(recent_has[0, -1]).reshape(-1)[0])
                            if rh:
                                recent_tilde = np.asarray(recent_ckpts[0, -1]).astype(np.float32)
                                has_recent = True
                        elif recent_ckpts.ndim >= 1 and recent_ckpts.shape[0] > 0:
                            # shape: (num_ckpt, ...)
                            rh = bool(np.asarray(recent_has[-1]).reshape(-1)[0])
                            if rh:
                                recent_tilde = np.asarray(recent_ckpts[-1]).astype(np.float32)
                                has_recent = True
                except Exception:
                    has_recent = False
                    recent_tilde = np.zeros((0,), dtype=np.float32)

                np.savez_compressed(
                    step_dir / "mode_tildes.npz",
                    has_recent=np.array([1 if has_recent else 0], dtype=np.uint8),
                    has_random=np.array([0], dtype=np.uint8),
                    recent_tilde=recent_tilde if has_recent else np.zeros((0,), dtype=np.float32),
                    random_tilde=np.zeros((0,), dtype=np.float32),
                )

                def _plain_final(v):
                    if isinstance(v, dict):
                        return {str(k): _plain_final(x) for k, x in v.items()}
                    if isinstance(v, (list, tuple)):
                        return [_plain_final(x) for x in v]
                    if isinstance(v, np.ndarray):
                        return v.tolist()
                    if hasattr(v, "item"):
                        try:
                            return v.item()
                        except Exception:
                            pass
                    if isinstance(v, (str, int, float, bool)) or v is None:
                        return v
                    try:
                        return os.fspath(v)
                    except Exception:
                        return str(v)

                item = {
                    "params": out["runner_state"][0].params,
                    "params_spec": out["runner_state"][0].params,
                    "params_ind": out["runner_state"][-5].params,
                    "config": _plain_final(config),
                    "update_step": np.int32(upd),
                    "seed": np.int32(seed_local),
                }
                ckpt_dir = step_dir / "model_ckpt"
                if ckpt_dir.exists():
                    import shutil as _shutil
                    _shutil.rmtree(ckpt_dir, ignore_errors=True)
                orbax_checkpointer = ocp.PyTreeCheckpointer()
                save_args = orbax_utils.save_args_from_target(item)
                orbax_checkpointer.save(str(ckpt_dir.resolve()), item, save_args=save_args)
            except Exception as e:
                print(f"[CKPTDBG][PH2][ERROR] final eval snapshot export failed: {type(e).__name__}: {e}")
                raise RuntimeError(
                    f"final eval snapshot export failed: {type(e).__name__}: {e}"
                ) from e

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
