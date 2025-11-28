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


def load_fcp_populations(population_dir):
    def _load_fcp_population(dir):
        # ckpt_0 (initial weights)도 population 다양성/분석에 포함시키기 위해 skip_initial=False로 설정
        all_checkpoints, fcp_config = load_all_checkpoints(
            dir, final_only=False, skip_initial=False
        )
        all_population_params, _ = jax.tree_util.tree_flatten(
            all_checkpoints, is_leaf=lambda x: type(x) is PPOParams
        )
        print(
            f"Loaded FCP population params for {len(all_population_params)} policies from {dir}"
        )
        all_population_params = jax.tree_util.tree_map(
            lambda *v: jnp.stack(v), *all_population_params
        )
        return all_population_params, fcp_config

    all_populations = []
    first_fcp_config = None
    for dir in population_dir.iterdir():
        if not dir.is_dir() or "fcp_" not in dir.name:
            continue

        print(f"Loading FCP population from {dir}")
        population, fcp_config = _load_fcp_population(dir)
        # population is stacked (num_checkpoints, ...)
        num = jax.tree_util.tree_leaves(population)[0].shape[0]
        def reshape_kernel(x):
            if isinstance(x, dict) and 'kernel' in x:
                shape = x['kernel'].shape
                if len(shape) == 4 and shape[0] == 1 and shape[1] == 1:  # (1, 1, 38, 128) -> squeeze axis 1
                    x['kernel'] = x['kernel'].squeeze(1)
            return x
        # population = jax.tree_util.tree_map(reshape_kernel, population)  # Wrong: applied to stacked, len=5
        for i in range(num):
            params_i = jax.tree_util.tree_map(lambda x: x[i], population)
            params_i = jax.tree_util.tree_map(reshape_kernel, params_i)  # Apply reshape after unstacking
            all_populations.append(params_i)
        if first_fcp_config is None:
            first_fcp_config = fcp_config

    print(f"Successfully loaded {len(all_populations)} FCP policies")
    return all_populations, first_fcp_config


def single_run(config):
    num_seeds = config["NUM_SEEDS"]
    num_runs = num_seeds

    all_populations = None
    if "FCP" in config:
        print("Training FCP")
        assert num_seeds == 1
        print("Loading population from", config["FCP"])
        population_dir = Path(config["FCP"])

        all_populations, fcp_population_config = load_fcp_populations(population_dir)
        all_populations = all_populations.params

        num_runs = jax.tree_util.tree_flatten(all_populations)[0][0].shape[0]

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
        # num_devices = 1
        num_devices = get_num_devices()
        print("Using", num_devices, "devices")

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
        # out = scanned_mini_batch_map(train_jit, 4, use_pmap=True)(
        #     rngs, **train_extra_args
        # )
        # out = scanned_mini_batch_map(train_jit, 2, num_devices=num_devices)(
        #     rngs, **train_extra_args
        # )
        # out = jax.vmap(train_jit)(rngs, **train_extra_args)

        return out
# def single_run(config):
    num_seeds = config["NUM_SEEDS"]
    num_runs = num_seeds

    all_populations = None
    if "FCP" in config:
        print("Training FCP")
        # NOTE: FCP에서도 여러 시드 지원. 각 시드에서 같은 population을 사용해 ego agent 훈련.
        # population은 시드별로 다시 로드됨 (비효율적일 수 있음).
        print("Loading population from", config["FCP"])
        population_dir = Path(config["FCP"])

        all_populations, fcp_population_config = load_fcp_populations(population_dir)
        fcp_policy = FCPWrapperPolicy(config, *all_populations)

        print(f"Loaded FCP population with {len(all_populations)} policies")

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

        rngs = jax.random.split(jax.random.PRNGKey(config["SEED"]), num_runs)

        train_func = make_train(
            config_copy,
            population_config=population_config,
        )

        # num_devices = len(jax.devices("gpu"))
        # num_devices = 1
        num_devices = get_num_devices()
        print("Using", num_devices, "devices")

        train_jit = jax.jit(train_func, static_argnames=['population'])

        # 한국어 주석: population(또는 BC policy) 같은 큰 PyTree/비배열 인자를 pmap 인자로 넘기면
        # in_axes=0 기본 규칙 때문에 축 매핑 대상이 되어 rank 0 오류가 날 수 있습니다.
        # 따라서 이러한 인자는 pmap 인자에서 제거하고, 클로저로 캡처하여 브로드캐스트(모든 디바이스에서 동일 사용)합니다.
        train_extra_args = {}
        if all_populations is not None:
            print("Training with FCP")
            train_extra_args["population"] = fcp_policy
        elif bc_policy is not None:
            print("Training with BC")
            print("Using BC policy", bc_policy)
            train_extra_args["population"] = bc_policy

        # 한국어 주석: pmap/vmap에 넘길 함수는 rng 하나만을 인자로 받도록 래핑합니다.
        def train_with_pop(rng):
            return train_jit(rng, **train_extra_args)

        # 한국어 주석: 디바이스 수에 따라 안전하게 분기합니다.
        # - GPU 1장: pmap 대신 직접 호출 또는 vmap 사용 (pmap 축 오류 방지)
        # - GPU 여러장: 시드 축을 디바이스 축으로 매핑. 시드 수가 디바이스 수의 배수인 경우 2D로 reshape 후 pmap(jax.vmap) 사용
        seed_n = rngs.shape[0]
        if num_devices <= 1:
            if seed_n == 1:
                out = train_with_pop(rngs[0])
            else:
                out = jax.vmap(train_with_pop)(rngs)
        else:
            if seed_n == num_devices:
                # (num_devices, ...) 형태로 바로 pmap
                out = jax.pmap(train_with_pop)(rngs)
            elif seed_n % num_devices == 0:
                # (num_devices, seeds_per_device, ...) 형태로 나누어 각 디바이스에서 vmap
                seeds_per_device = seed_n // num_devices
                rngs_2d = rngs.reshape((num_devices, seeds_per_device, *rngs.shape[1:]))
                out = jax.pmap(jax.vmap(train_with_pop))(rngs_2d)
                # pmap(jax.vmap) 결과를 다시 (seed_n, ...)로 평탄화
                out = jax.tree_util.tree_map(lambda x: x.reshape((seed_n, *x.shape[2:])), out)
            else:
                # 시드 수가 디바이스 수의 배수가 아니면, 단일 디바이스 경로로 안전 처리
                print(
                    f"[warn] num_seeds({seed_n}) % num_devices({num_devices}) != 0; falling back to single-device vmap"
                )
                out = jax.vmap(train_with_pop)(rngs)

        return out
