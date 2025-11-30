import copy
from functools import partial
from pathlib import Path
from omegaconf import OmegaConf
import wandb
import jax
import os
from datetime import datetime
import jax.numpy as jnp

from overcooked_v2_experiments.human_rl.imitation.bc_policy import BCPolicy
from overcooked_v2_experiments.ppo.policy import PPOParams
from overcooked_v2_experiments.ppo.utils.fcp import FCPWrapperPolicy
from .ippo import make_train
from .utils.store import (
    load_all_checkpoints,
    store_checkpoint,
)
from .utils.utils import get_num_devices, get_run_base_dir
from overcooked_v2_experiments.utils.utils import (
    mini_batch_pmap,
    scanned_mini_batch_map,
)
from overcooked_v2_experiments.ppo.utils.visualize_ppo import visualize_ppo_policy

jax.config.update("jax_debug_nans", True)


# def load_fcp_populations(population_dir):
#     def _load_fcp_population(dir):
#         all_checkpoints, fcp_config = load_all_checkpoints(
#             dir, final_only=False, skip_initial=False
#         )
#         ppo_params_list, _ = jax.tree_util.tree_flatten(
#             all_checkpoints, is_leaf=lambda x: type(x) is PPOParams
#         )
#         print(
#             f"Loaded FCP population params for {len(ppo_params_list)} policies from {dir}"
#         )
#         return ppo_params_list, fcp_config

#     all_policies = []
#     first_fcp_config = None

#     # 🔧 (통하지 않음) leaf-level에서 Conv 커널 모양을 (1,1,Cin,Cout) -> (1,Cin,Cout)로 맞춰주는 함수
#     # def fix_conv_kernel_leaf(x):
#     #     import jax.numpy as jnp
#     #     if isinstance(x, jnp.ndarray):
#     #         # Checkpoint has (1, 1, Cin, Cout), we want (1, Cin, Cout)
#     #         if x.ndim == 4 and x.shape[0] == 1 and x.shape[1] == 1:
#     #             return x.reshape((1, x.shape[2], x.shape[3]))
#     #     return x

#     for dir in population_dir.iterdir():
#         if not dir.is_dir() or "fcp_" not in dir.name:
#             continue

#         print(f"Loading FCP population from {dir}")
#         ppo_params_list, fcp_config = _load_fcp_population(dir)  # list[PPOParams]

#         for ppo_params in ppo_params_list:
#             # ✅ 여기서 그냥 PPOParams 안에 있는 params만 꺼내고,
            
#             params_dict = ppo_params.params   # 보통 {'params': {...}} 형태일 것

#             # 여기에서만 커널 모양을 한 번 수정해준다.
#             # params_dict = jax.tree_util.tree_map(fix_conv_kernel_leaf, params_dict)

#             all_policies.append(params_dict)

#         if first_fcp_config is None:
#             first_fcp_config = fcp_config

#     print(f"Successfully loaded {len(all_policies)} FCP policies")

#     if len(all_policies) == 0:
#         raise ValueError(f"No FCP populations found in {population_dir}")

#     stacked_populations = jax.tree_util.tree_map(
#         lambda *xs: jnp.stack(xs), *all_policies
#     )
#     # leaf: (num_policies, ...)  ex) (30, 1, 1, 38, 128) 이런 식

#     return stacked_populations, first_fcp_config
# def load_fcp_populations(population_dir):
#     def _load_fcp_population(dir):
#         all_checkpoints, fcp_config = load_all_checkpoints(
#             dir, final_only=False, skip_initial=True
#         )
#         all_population_params, _ = jax.tree_util.tree_flatten(
#             all_checkpoints, is_leaf=lambda x: type(x) is PPOParams
#         )
#         print(
#             f"Loaded FCP population params for {len(all_population_params)} policies from {dir}"
#         )
#         all_population_params = jax.tree_util.tree_map(
#             lambda *v: jnp.stack(v), *all_population_params
#         )
#         return all_population_params, fcp_config

#     all_populations = []
#     first_fcp_config = None
#     for dir in population_dir.iterdir():
#         if not dir.is_dir() or "fcp_" not in dir.name:
#             continue

#         print(f"Loading FCP population from {dir}")
#         population, fcp_config = _load_fcp_population(dir)
#         all_populations.append(population)
#         if first_fcp_config is None:
#             first_fcp_config = fcp_config

#     print(f"Successfully loaded {len(all_populations)} FCP populations")
#     all_populations = jax.tree_util.tree_map(lambda *v: jnp.stack(v), *all_populations)
#     return all_populations, first_fcp_config
def load_fcp_populations(population_dir: Path):
    """
    FCP population 디렉토리 아래 모든 fcp_* 폴더에서
    PPOParams 체크포인트를 전부 모아 하나의 population으로 만든다.

    - 폴더마다 policy 개수(pop_size)는 달라도 상관 없음.
    - 단, 각 policy의 params 트리 구조와 leaf shape는 동일해야 함.

    Returns
    -------
    stacked_populations : PyTree of JAX arrays
        각 leaf shape: (num_policies_total, ...)  # 모든 폴더에서 모은 policy 수
    first_fcp_config : DictConfig
        첫 번째로 발견된 fcp_config (대부분 동일할 것)
    """

    def _load_policies_from_dir(dir: Path):
        """
        하나의 fcp_* 디렉토리에서 PPOParams들을 전부 꺼내서
        [params_tree, params_tree, ...] 리스트로 반환.
        """
        all_checkpoints, fcp_config = load_all_checkpoints(
            dir,
            final_only=False,
            skip_initial=False,   # 원본과 동일 동작. 필요하면 False로 바꿔도 됨.
        )

        # all_checkpoints 안에서 PPOParams만 leaf로 취급해서 리스트로 뽑기
        ppo_params_list, _ = jax.tree_util.tree_flatten(
            all_checkpoints,
            is_leaf=lambda x: isinstance(x, PPOParams),
        )

        print(
            f"Loaded FCP population params for {len(ppo_params_list)} policies from {dir}"
        )

        # 각 PPOParams에서 .params만 꺼내서 순수 params 트리로 변환
        params_list = [p.params for p in ppo_params_list]

        # 디버그: 첫 번째 policy의 shape 한 번 찍어보기
        if params_list:
            shapes = jax.tree_util.tree_map(lambda x: x.shape, params_list[0])
            print(f"[DEBUG] Example policy shapes in {dir.name}: {shapes}")

        return params_list, fcp_config

    # ----------------------------------------------------------------------
    # 1. population_dir 아래 모든 fcp_* 폴더에서 policy params 모으기
    # ----------------------------------------------------------------------
    all_policy_params = []   # 모든 폴더의 params를 여기로 평탄하게 모음
    first_fcp_config = None

    population_dir = Path(population_dir)
    if not population_dir.exists():
        raise ValueError(f"Population dir does not exist: {population_dir}")

    for dir in sorted(population_dir.iterdir()):
        if not dir.is_dir() or "fcp_" not in dir.name:
            continue

        print(f"Loading FCP population from {dir}")
        params_list, fcp_config = _load_policies_from_dir(dir)

        # 이 폴더의 policy들을 전체 리스트에 추가
        all_policy_params.extend(params_list)

        if first_fcp_config is None:
            first_fcp_config = fcp_config

    if len(all_policy_params) == 0:
        raise ValueError(f"No PPOParams found under {population_dir}")

    print(f"Successfully collected {len(all_policy_params)} FCP policies in total.")

    # ----------------------------------------------------------------------
    # 2. 모든 policy params를 한 번에 stack → (num_policies_total, ...)
    # ----------------------------------------------------------------------
    stacked_populations = jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs), *all_policy_params
    )

    # 디버그: 최종 population shape
    fcp_params_shape = jax.tree_util.tree_map(lambda x: x.shape, stacked_populations)
    print("[DEBUG] Final stacked FCP params shape:", fcp_params_shape)

    return stacked_populations, first_fcp_config


def single_run(config):
    num_seeds = config["NUM_SEEDS"]
    num_runs = num_seeds

    all_populations = None
    if "FCP" in config:
        print("Training FCP")
        # assert num_seeds == 1
        print("Loading population from", config["FCP"])
        population_dir = Path(config["FCP"])

        all_populations, fcp_population_config = load_fcp_populations(population_dir)
        # all_populations = all_populations.params

        # fcp population은 num_runs와 별도의 축 
        pop_size = jax.tree_util.tree_flatten(all_populations)[0][0].shape[0]

        print(f"Loaded FCP population with {num_runs} runs")

        fcp_params_shape = jax.tree_util.tree_map(lambda x: x.shape, all_populations)
        print("FCP params shape", fcp_params_shape)

    bc_policy = None
    if "BC" in config:
        print("Training with BC")
        layout_name = config["env"]["ENV_KWARGS"]["layout"]
        split = "all"
        run_id = 1
        print(f"Loading BC policy from {layout_name}-{split}-{run_id}")
        bc_policy = BCPolicy.from_pretrained(layout_name, split, run_id)

    with jax.disable_jit(False):
        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, num_runs)

        config_copy = copy.deepcopy(config)
        if bc_policy is not None:
            config_copy["env"]["ENV_KWARGS"]["force_path_planning"] = True

        population_config = None
        if all_populations is not None:
            population_config = fcp_population_config

        train_func = make_train(
            config_copy,
            population_config=population_config,
        )

        # num_devices = len(jax.devices("gpu"))
        num_devices = get_num_devices()
        print("Using", num_devices, "devices")

        # ---- FCP일 때: population은 클로저로 고정 ----
        if all_populations is not None:
            print("Training with FCP")

            def train_with_pop(rng):
                # 여기서 population을 고정 파라미터로 넣어줌
                return train_func(rng, population=all_populations)

            train_with_pop_jit = jax.jit(train_with_pop)

            # ✅ 여기서 mini_batch_pmap 재사용
            out = mini_batch_pmap(train_with_pop_jit, num_devices)(rngs)
            return out


        # 시드 하나 단위로 학습을 돌리기 위해, 그 학습을 병렬로 처리
        train_jit = jax.jit(train_func)

        train_extra_args = {}
        if all_populations is not None:
            print("Training with FCP")
            train_extra_args["population"] = all_populations
        elif bc_policy is not None:
            print("Training with BC")
            print("Using BC policy", bc_policy)
            train_extra_args["population"] = bc_policy

        out = mini_batch_pmap(train_jit, num_devices)(rngs, **train_extra_args)

        return out
    


# def single_run(config):
    # num_seeds = config["NUM_SEEDS"]
    # num_runs = num_seeds

    # all_populations = None
    # if "FCP" in config:
    #     print("Training FCP")
    #     # NOTE: FCP에서도 여러 시드 지원. 각 시드에서 같은 population을 사용해 ego agent 훈련.
    #     # population은 시드별로 다시 로드됨 (비효율적일 수 있음).
    #     print("Loading population from", config["FCP"])
    #     population_dir = Path(config["FCP"])

    #     all_populations, fcp_population_config = load_fcp_populations(population_dir)
    #     fcp_policy = FCPWrapperPolicy(config, *all_populations)

    #     print(f"Loaded FCP population with {len(all_populations)} policies")

    # bc_policy = None
    # if "BC" in config:
    #     print("Training with BC")
    #     layout_name = config["env"]["ENV_KWARGS"]["layout"]
    #     split = "all"
    #     run_id = 1
    #     print(f"Loading BC policy from {layout_name}-{split}-{run_id}")
    #     bc_policy = BCPolicy.from_pretrained(layout_name, split, run_id)

    # with jax.disable_jit(False):
    #     rng = jax.random.PRNGKey(config["SEED"])
    #     rngs = jax.random.split(rng, num_runs)

    #     config_copy = copy.deepcopy(config)
    #     if bc_policy is not None:
    #         config_copy["env"]["ENV_KWARGS"]["force_path_planning"] = True

    #     population_config = None

    #     rngs = jax.random.split(jax.random.PRNGKey(config["SEED"]), num_runs)

    #     train_func = make_train(
    #         config_copy,
    #         population_config=population_config,
    #     )

    #     # num_devices = len(jax.devices("gpu"))
    #     # num_devices = 1
    #     num_devices = get_num_devices()
    #     print("Using", num_devices, "devices")

    #     train_jit = jax.jit(train_func, static_argnames=['population'])

    #     # 한국어 주석: population(또는 BC policy) 같은 큰 PyTree/비배열 인자를 pmap 인자로 넘기면
    #     # in_axes=0 기본 규칙 때문에 축 매핑 대상이 되어 rank 0 오류가 날 수 있습니다.
    #     # 따라서 이러한 인자는 pmap 인자에서 제거하고, 클로저로 캡처하여 브로드캐스트(모든 디바이스에서 동일 사용)합니다.
    #     train_extra_args = {}
    #     if all_populations is not None:
    #         print("Training with FCP")
    #         train_extra_args["population"] = fcp_policy
    #     elif bc_policy is not None:
    #         print("Training with BC")
    #         print("Using BC policy", bc_policy)
    #         train_extra_args["population"] = bc_policy

    #     # 한국어 주석: pmap/vmap에 넘길 함수는 rng 하나만을 인자로 받도록 래핑합니다.
    #     def train_with_pop(rng):
    #         return train_jit(rng, **train_extra_args)

    #     # 한국어 주석: 디바이스 수에 따라 안전하게 분기합니다.
    #     # - GPU 1장: pmap 대신 직접 호출 또는 vmap 사용 (pmap 축 오류 방지)
    #     # - GPU 여러장: 시드 축을 디바이스 축으로 매핑. 시드 수가 디바이스 수의 배수인 경우 2D로 reshape 후 pmap(jax.vmap) 사용
    #     seed_n = rngs.shape[0]
    #     if num_devices <= 1:
    #         if seed_n == 1:
    #             out = train_with_pop(rngs[0])
    #         else:
    #             out = jax.vmap(train_with_pop)(rngs)
    #     else:
    #         if seed_n == num_devices:
    #             # (num_devices, ...) 형태로 바로 pmap
    #             out = jax.pmap(train_with_pop)(rngs)
    #         elif seed_n % num_devices == 0:
    #             # (num_devices, seeds_per_device, ...) 형태로 나누어 각 디바이스에서 vmap
    #             seeds_per_device = seed_n // num_devices
    #             rngs_2d = rngs.reshape((num_devices, seeds_per_device, *rngs.shape[1:]))
    #             out = jax.pmap(jax.vmap(train_with_pop))(rngs_2d)
    #             # pmap(jax.vmap) 결과를 다시 (seed_n, ...)로 평탄화
    #             out = jax.tree_util.tree_map(lambda x: x.reshape((seed_n, *x.shape[2:])), out)
    #         else:
    #             # 시드 수가 디바이스 수의 배수가 아니면, 단일 디바이스 경로로 안전 처리
    #             print(
    #                 f"[warn] num_seeds({seed_n}) % num_devices({num_devices}) != 0; falling back to single-device vmap"
    #             )
    #             out = jax.vmap(train_with_pop)(rngs)

    #     return out
