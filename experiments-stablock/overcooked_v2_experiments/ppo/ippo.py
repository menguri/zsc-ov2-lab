""" 
Based on PureJaxRL Implementation of PPO
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Optional, Sequence, NamedTuple, Any, Dict, Union
from flax.training.train_state import TrainState
import distrax
import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper, OvercookedV2LogWrapper
import hydra
from omegaconf import OmegaConf
from datetime import datetime
import os
import wandb
import functools
import math
import pickle
import threading
from models.rnn import ScannedRNN
import matplotlib.pyplot as plt
from jaxmarl.environments.overcooked_v2.overcooked import ObservationType
from jaxmarl.environments.overcooked.layouts import overcooked_layouts
from overcooked_v2_experiments.ppo.utils.stablock import (
    expand_blocked_states,
    initialize_blocked_states,
    resample_blocked_states,
    enumerate_reachable_positions,
)
from overcooked_v2_experiments.ppo.ph1_utils import (
    compute_ph1_probs,
    sample_target_idx,
    sample_multi_targets_from_pool,
)
from overcooked_v2_experiments.eval.policy import AbstractPolicy
from overcooked_v2_experiments.ppo.models.abstract import ActorCriticBase
from .models.model import get_actor_critic, initialize_carry
from overcooked_v2_experiments.eval.policy import AbstractPolicy
from flax import core
from .utils.stablock import enumerate_reachable_positions


# Host-side guard for PH1 eval callback.
# In pmap runs, callback can be invoked per replica.
# Deduplicate by (seed, env_step) to avoid concurrent same-path snapshot writes.
_PH1_EVAL_LOCK = threading.Lock()
_PH1_EVAL_SEEN_KEYS = set()


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    train_mask: jnp.ndarray
    obs_history: jnp.ndarray
    act_history: jnp.ndarray
    partner_action: jnp.ndarray
    is_ego: jnp.ndarray
    z_state: jnp.ndarray
    blocked_states: jnp.ndarray
    prev_action: jnp.ndarray
    partner_prediction: jnp.ndarray
    hstate: any


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def _should_use_old_overcooked(config: Dict[str, Any], env_config: Dict[str, Any]):
    # SA(state augmentation)는 기존 OvercookedV2 경로를 유지한다.
    if "NUM_ITERATIONS" in config:
        return False, "state_augmentation"

    explicit_old = bool(config.get("OLD_OVERCOOKED", False))
    if explicit_old:
        return True, "explicit_flag"

    if bool(config.get("DISABLE_OLD_OVERCOOKED_AUTO", False)):
        return False, "auto_disabled"

    layout_name = env_config.get("ENV_KWARGS", {}).get("layout", None)
    if isinstance(layout_name, str) and layout_name in overcooked_layouts:
        return True, "auto_layout_match"

    return False, "default_v2"


def _prepare_env_spec(config: Dict[str, Any], env_config: Dict[str, Any]):
    env_name = str(env_config.get("ENV_NAME", "overcooked_v2"))
    env_kwargs = dict(env_config.get("ENV_KWARGS", {}))

    use_old, reason = _should_use_old_overcooked(config, env_config)
    if use_old:
        if env_name != "overcooked":
            print(
                f"[ENV] Routing to overcooked(v1) ({reason}); "
                f"ignoring ENV_NAME='{env_name}'."
            )
        env_name = "overcooked"

    if env_name == "overcooked":
        allowed = {"layout", "random_reset", "max_steps"}
        dropped = sorted(set(env_kwargs.keys()) - allowed)
        if dropped:
            print(
                "[ENV] overcooked(v1) selected; dropping OV2-only kwargs:",
                ", ".join(dropped),
            )
            env_kwargs = {k: v for k, v in env_kwargs.items() if k in allowed}

        layout_name = env_kwargs.get("layout", "cramped_room")
        if isinstance(layout_name, str):
            if layout_name not in overcooked_layouts:
                raise ValueError(
                    f"Unknown overcooked(v1) layout '{layout_name}'. "
                    f"Available: {sorted(overcooked_layouts.keys())}"
                )
            env_kwargs["layout"] = overcooked_layouts[layout_name]

    return env_name, env_kwargs


def make_train(
    config,
    update_step_offset=None,
    update_step_num_overwrite=None,
    population_config=None,
):
    env_config = config["env"]
    model_config = config["model"]

    ph1_enabled = bool(config.get("PH1_ENABLED", False))
    # Execution policy input is always the environment-provided observation.
    # PH1 only injects the blocked target (tilde{s}) via `blocked_states`.
    use_ph1_partner_pred = bool(config.get("PH1_USE_PARTNER_PRED", ph1_enabled))
    ph1_epsilon = float(config.get("PH1_EPSILON", 0.0))
    ph1_pool_size = int(config.get("PH1_POOL_SIZE", 100))
    ph1_multi_penalty_enabled = bool(config.get("PH1_MULTI_PENALTY_ENABLED", False))
    ph1_max_penalty_count = int(config.get("PH1_MAX_PENALTY_COUNT", 1))
    ph1_max_penalty_count = max(1, ph1_max_penalty_count)
    ph1_penalty_slots = ph1_max_penalty_count if ph1_multi_penalty_enabled else 1
    ph1_multi_penalty_single_weight = float(
        config.get("PH1_MULTI_PENALTY_SINGLE_WEIGHT", 2.0)
    )
    ph1_multi_penalty_other_weight = float(
        config.get("PH1_MULTI_PENALTY_OTHER_WEIGHT", 1.0)
    )
    ph1_warmup_steps = int(config.get("PH1_WARMUP_STEPS", 0))
    ph1_beta_base = float(config.get("PH1_BETA", 1.0))
    ph1_beta_schedule_enabled = bool(config.get("PH1_BETA_SCHEDULE_ENABLED", False))
    ph1_beta_start = float(config.get("PH1_BETA_START", 0.0))
    ph1_beta_end = float(config.get("PH1_BETA_END", 2.0))
    ph1_beta_schedule_horizon_cfg = int(
        config.get("PH1_BETA_SCHEDULE_HORIZON_ENV_STEPS", -1)
    )
    ph1_eval_enabled = bool(config.get("PH1_EVAL_ENABLED", False)) and ph1_enabled
    ph1_eval_every_updates = int(config.get("PH1_EVAL_EVERY_UPDATES", 50))
    ph1_eval_every_env_steps = int(config.get("PH1_EVAL_EVERY_ENV_STEPS", 0))
    ph1_eval_video_every_env_steps = int(config.get("PH1_EVAL_VIDEO_EVERY_ENV_STEPS", 0))
    ph1_eval_defer_video = bool(config.get("PH1_EVAL_DEFER_VIDEO", False))
    abort_on_nan = bool(config.get("ABORT_ON_NAN", True))

    # E3T 설정 로드
    alg_name = config.get("ALG_NAME", "SP")
    alg_name_u = str(alg_name).upper()
    e3t_enabled = ("E3T" in alg_name_u)
    e3t_epsilon = config.get("E3T_EPSILON", 0.05)
    use_partner_modeling = config.get("USE_PARTNER_MODELING", True)
    # [Stablock] 알고리즘 활성화 및 패널티 설정
    stablock_enabled = bool(config.get("STABLOCK_ENABLED", False))
    stablock_heavy_penalty = float(config.get("STABLOCK_HEAVY_PENALTY", 10.0))
    stablock_no_block_prob = config.get("STABLOCK_NO_BLOCK_PROB", None)
    env_name, env_kwargs = _prepare_env_spec(config, env_config)
    is_overcooked_v1 = env_name == "overcooked"

    if is_overcooked_v1 and stablock_enabled:
        raise ValueError(
            "Stablock is implemented for overcooked_v2 state APIs. "
            "Use ENV_NAME=overcooked_v2 for Stablock runs."
        )
    if is_overcooked_v1 and ph1_eval_enabled:
        print(
            "[PH1-EVAL] overcooked(v1) mode detected; disabling PH1 online snapshot callback."
        )
        ph1_eval_enabled = False

    # Optional device selection for environment to mitigate GPU OOM.
    # Usage via Hydra override: +ENV_DEVICE=cpu  (default: gpu / auto)
    env_device = config.get("ENV_DEVICE", None)  # None -> default placement
    if env_device is not None and env_device not in ("cpu", "gpu"):
        raise ValueError(f"ENV_DEVICE must be one of 'cpu','gpu' (or unset). Got: {env_device}")

    # Create env under a device context so large static buffers (layout grids etc.) live on CPU when requested.
    if env_device == "cpu":
        with jax.default_device(jax.devices("cpu")[0]):
            env = jaxmarl.make(env_name, **env_kwargs)
    else:
        env = jaxmarl.make(env_name, **env_kwargs)

    ACTION_DIM = env.action_space(env.agents[0]).n

    model_config["NUM_ACTORS"] = env.num_agents * model_config["NUM_ENVS"]
    model_config["NUM_UPDATES"] = (
        model_config["TOTAL_TIMESTEPS"]
        // model_config["NUM_STEPS"]
        // model_config["NUM_ENVS"]
    )
    model_config["MINIBATCH_SIZE"] = (
        model_config["NUM_ACTORS"]
        * model_config["NUM_STEPS"]
        // model_config["NUM_MINIBATCHES"]
    )

    steps_per_update = model_config["NUM_STEPS"] * model_config["NUM_ENVS"]
    total_timesteps = int(model_config["TOTAL_TIMESTEPS"])
    if ph1_beta_schedule_horizon_cfg <= 0:
        ph1_beta_schedule_horizon_env_steps = max(1, total_timesteps - ph1_warmup_steps)
    else:
        ph1_beta_schedule_horizon_env_steps = max(1, ph1_beta_schedule_horizon_cfg)

    def _current_ph1_beta(update_step):
        if not ph1_beta_schedule_enabled:
            return jnp.float32(ph1_beta_base), jnp.float32(0.0)

        env_step = (
            jnp.asarray(update_step, dtype=jnp.float32)
            * jnp.float32(steps_per_update)
        )
        progress = (
            (env_step - jnp.float32(ph1_warmup_steps))
            / jnp.float32(ph1_beta_schedule_horizon_env_steps)
        )
        progress = jnp.clip(progress, 0.0, 1.0)
        beta_t = (
            jnp.float32(ph1_beta_start)
            + progress * jnp.float32(ph1_beta_end - ph1_beta_start)
        )
        return beta_t, progress

    ckpt_every_env_steps = int(config.get("PH1_EVAL_EVERY_ENV_STEPS", 0))
    if ckpt_every_env_steps > 0:
        target_env_steps = np.arange(
            0,
            total_timesteps + ckpt_every_env_steps,
            ckpt_every_env_steps,
            dtype=np.int64,
        )
        if target_env_steps[-1] != total_timesteps:
            target_env_steps = np.append(target_env_steps, total_timesteps)
        checkpoint_steps_np = np.unique(
            np.clip(
                np.ceil(target_env_steps / max(1, steps_per_update)).astype(np.int32),
                0,
                int(model_config["NUM_UPDATES"]),
            )
        )
        num_checkpoints = int(checkpoint_steps_np.shape[0])
        checkpoint_steps = jnp.array(checkpoint_steps_np, dtype=jnp.int32)
    else:
        num_checkpoints = int(config["NUM_CHECKPOINTS"])
        checkpoint_steps = jnp.linspace(
            0,
            model_config["NUM_UPDATES"],
            num_checkpoints,
            endpoint=True,
            dtype=jnp.int32,
        )
        if num_checkpoints > 0:
            # make sure the last checkpoint is the last update step
            checkpoint_steps = checkpoint_steps.at[-1].set(model_config["NUM_UPDATES"])

    print("Checkpoint steps: ", checkpoint_steps)

    def _update_checkpoint(checkpoint_states, params, i):
        jax.debug.print("Saving checkpointing {i}", i=i)
        return jax.tree_util.tree_map(
            lambda x, y: x.at[i].set(y),
            checkpoint_states,
            params,
        )

    def _extract_recent_tilde_from_pool(pool_states):
        # pool_states: (Envs, Pool, ...)
        if pool_states.ndim < 2:
            return jnp.zeros((1,), dtype=jnp.float32), jnp.array(False)
        env0 = pool_states[0]
        if env0.ndim < 1 or env0.shape[0] <= 0:
            return jnp.zeros((1,), dtype=jnp.float32), jnp.array(False)
        reduce_axes = tuple(range(1, int(env0.ndim)))
        if len(reduce_axes) == 0:
            valid = env0 >= 0.0
        else:
            valid = jnp.any(env0 >= 0.0, axis=reduce_axes)
        has_recent = jnp.any(valid)
        last_idx = jnp.maximum(jnp.sum(valid.astype(jnp.int32)) - 1, 0)
        recent = env0[last_idx]
        recent = jnp.where(has_recent, recent, jnp.zeros_like(env0[0]))
        return recent, has_recent

    if env_name == "overcooked_v2":
        env = OvercookedV2LogWrapper(env, replace_info=False)
    else:
        env = LogWrapper(env, replace_info=False)

    def _extract_pos_axes(log_env_state):
        if is_overcooked_v1:
            # overcooked(v1) batched state: (E, A, 2) -> convert to (A, E)
            agent_pos = log_env_state.env_state.agent_pos
            pos_x = jnp.swapaxes(agent_pos[..., 0], 0, 1)
            pos_y = jnp.swapaxes(agent_pos[..., 1], 0, 1)
            return pos_y, pos_x
        return log_env_state.env_state.agents.pos.y, log_env_state.env_state.agents.pos.x

    def _extract_global_full_obs(log_env_state):
        if is_overcooked_v1:
            full = jax.vmap(env.get_obs)(log_env_state.env_state)
            return full[env.agents[0]].astype(jnp.float32)
        full = jax.vmap(env.get_obs_default)(log_env_state.env_state)
        return full[:, 0].astype(jnp.float32)

    # Wrap reset/step with backend-specific jits if ENV_DEVICE explicitly set.
    reset_fn = env.reset
    step_fn = env.step
    if env_device == "cpu":
        reset_fn = jax.jit(env.reset, backend="cpu")
        step_fn = jax.jit(env.step, backend="cpu")
    elif env_device == "gpu":
        reset_fn = jax.jit(env.reset, backend="gpu")
        step_fn = jax.jit(env.step, backend="gpu")

    # Optional mixed precision for stored observations to reduce memory of trajectory buffers.
    cast_obs_bf16 = bool(config.get("CAST_OBS_BF16", False))

    def create_learning_rate_fn():
        base_learning_rate = model_config["LR"]

        lr_warmup = model_config["LR_WARMUP"]
        update_steps = model_config["NUM_UPDATES"]
        warmup_steps = int(lr_warmup * update_steps)

        steps_per_epoch = (
            model_config["NUM_MINIBATCHES"] * model_config["UPDATE_EPOCHS"]
        )

        warmup_fn = optax.linear_schedule(
            init_value=0.0,
            end_value=base_learning_rate,
            transition_steps=warmup_steps * steps_per_epoch,
        )
        cosine_epochs = max(update_steps - warmup_steps, 1)

        print("Update steps: ", update_steps)
        print("Warmup epochs: ", warmup_steps)
        print("Cosine epochs: ", cosine_epochs)

        cosine_fn = optax.cosine_decay_schedule(
            init_value=base_learning_rate, decay_steps=cosine_epochs * steps_per_epoch
        )
        schedule_fn = optax.join_schedules(
            schedules=[warmup_fn, cosine_fn],
            boundaries=[warmup_steps * steps_per_epoch],
        )
        return schedule_fn

    rew_shaping_anneal = optax.linear_schedule(
        init_value=1.0,
        end_value=0.0,
        transition_steps=model_config["REW_SHAPING_HORIZON"],
    )

    train_idxs = jnp.linspace(
        0,
        env.num_agents,
        model_config["NUM_ENVS"],
        dtype=jnp.int32,
        endpoint=False,
    )
    train_mask_dict = {a: train_idxs == i for i, a in enumerate(env.agents)}
    train_mask_flat = batchify(
        train_mask_dict, env.agents, model_config["NUM_ACTORS"]
    ).squeeze()

    print("train_mask_flat", train_mask_flat.shape)
    print("train_mask_flat sum", train_mask_flat.sum())

    # NUM_ACTORS = (env.num_agents * NUM_ENVS)이므로, 각 슬롯이 어떤 에이전트인지 구분하기 위해
    # 반복되는 인덱스 벡터를 만들어 ego/partner 마스크를 구성한다.
    # batchify 결과는 Agent Major 순서 ([Ag0_Env0, ..., Ag0_EnvN, Ag1_Env0, ...])이므로
    # repeat를 사용하여 [0, 0, ..., 1, 1, ...] 형태로 만들어야 한다.
    actor_indices = jnp.repeat(
        jnp.arange(env.num_agents, dtype=jnp.int32), model_config["NUM_ENVS"]
    )
    ego_actor_mask = actor_indices == 0
    partner_actor_mask = actor_indices == (env.num_agents - 1)

    use_population_annealing = False
    if "POPULATION_ANNEAL_HORIZON" in config:
        print("Using population annealing")
        use_population_annealing = True
        transition_begin = 0
        if "POPULATION_ANNEAL_BEGIN" in config:
            transition_begin = config["POPULATION_ANNEAL_BEGIN"]

        anneal_horizon = config["POPULATION_ANNEAL_HORIZON"]
        if anneal_horizon == 0:
            population_annealing_schedule = optax.constant_schedule(1.0)
        else:
            population_annealing_schedule = optax.linear_schedule(
                init_value=0.0,
                end_value=1.0,
                transition_steps=config["POPULATION_ANNEAL_HORIZON"] - transition_begin,
                transition_begin=transition_begin,
            )

    def train(
        rng,
        population: Optional[Union[AbstractPolicy, core.FrozenDict[str, Any]]] = None,
        initial_train_state=None,
    ):
        original_seed = rng[0]

        jax.debug.print("original_seed {s}", s=rng)

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, model_config["NUM_ENVS"])
        obsv, env_state = jax.vmap(reset_fn)(reset_rng)

        # Execution input shape: environment-provided observation
        # (NUM_ENVS, H, W, C_obs)
        state_shape = obsv[env.agents[0]].shape[1:]

        # PH1 blocked target shape: global full state with agent info.
        if ph1_enabled:
            global_full_env0 = _extract_global_full_obs(env_state)  # (E, H, W, C_full)
            ph1_block_shape = global_full_env0.shape[1:]
        else:
            ph1_block_shape = None

        # INIT NETWORK
        network = get_actor_critic(config)

        rng, _rng = jax.random.split(rng)

        init_x = (
            jnp.zeros(
                (1, model_config["NUM_ENVS"], *state_shape),
            ),
            jnp.zeros((1, model_config["NUM_ENVS"])),
        )
        init_hstate = initialize_carry(config, model_config["NUM_ENVS"])

        if init_hstate is not None:
            if isinstance(init_hstate, tuple):
                print("init_hstate (tuple)", [x.shape for x in init_hstate])
            else:
                print("init_hstate", init_hstate.shape)
        # jax.debug.print("check1 {x}", x=init_hstate.flatten()[0])

        print("init_x", init_x[0].shape, init_x[1].shape)

        # E3T: Prepare dummy inputs for Single Initialization
        dummy_partner_prediction = None
        dummy_obs_hist = None
        dummy_act_hist = None
        dummy_blocked_states = None
        # NOTE: agent_idx conditioning removed

        if (e3t_enabled or (ph1_enabled and use_ph1_partner_pred)) and use_partner_modeling:
            # Shape: (Time=1, Batch=NUM_ENVS, ActionDim=6)
            dummy_partner_prediction = jnp.zeros((1, model_config["NUM_ENVS"], ACTION_DIM))
            # Shape: (1, Batch, Context=5, H, W, C)
            dummy_obs_hist = jnp.zeros((1, model_config["NUM_ENVS"], 5, *state_shape))
            dummy_act_hist = jnp.zeros((1, model_config["NUM_ENVS"], 5), dtype=jnp.int32)

        # Keep init-time blocked input rule consistent with runtime apply rule.
        # PH2 phase2(ind) learner path uses population partner and should not consume blocked input.
        use_blocked_input_learner = (ph1_enabled or stablock_enabled)
        if use_blocked_input_learner:
            if stablock_enabled:
                # [Stablock] blocked_states 더미 입력 (좌표 그대로 전달)
                dummy_blocked_states = jnp.zeros((1, model_config["NUM_ENVS"], 2), dtype=jnp.int32)
            elif config.get("PH1_ENABLED", False):
                # PH1 always uses global full state as blocked target
                if ph1_multi_penalty_enabled:
                    init_shape = (
                        1,
                        model_config["NUM_ENVS"],
                        ph1_penalty_slots,
                    ) + ph1_block_shape
                else:
                    init_shape = (1, model_config["NUM_ENVS"]) + ph1_block_shape
                dummy_blocked_states = jnp.zeros(init_shape, dtype=jnp.float32)

        # Single Init: Initialize all parameters at once to avoid collision
        # 단일 초기화: 파라미터 충돌 방지를 위해 모든 모듈을 한 번에 초기화
        network_params = network.init(
            _rng, 
            init_hstate, 
            init_x, 
            partner_prediction=dummy_partner_prediction,
            obs_history=dummy_obs_hist,
            act_history=dummy_act_hist,
            blocked_states=dummy_blocked_states,
        )

        if model_config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(model_config["MAX_GRAD_NORM"]),
                optax.adam(create_learning_rate_fn(), eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(model_config["MAX_GRAD_NORM"]),
                optax.adam(model_config["LR"], eps=1e-5),
            )

        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        if initial_train_state is not None:
            train_state = initial_train_state

        # INIT ENV state already created above
        init_hstate = initialize_carry(config, model_config["NUM_ACTORS"])
        # jax.debug.print("check2 {x}", x=init_hstate.flatten()[0])

        init_population_hstate = None
        init_population_annealing_mask = None
        if population is not None:
            is_policy_population = False
            if isinstance(population, AbstractPolicy):
                is_policy_population = True
                rng, _rng = jax.random.split(rng)
                init_population_hstate = population.init_hstate(
                    model_config["NUM_ACTORS"], key=_rng
                )
            else:
                assert (
                    population_config is not None
                ), "population_config cannot be None if population is not a policy"
                population_network = get_actor_critic(population_config)
                init_population_hstate = initialize_carry(
                    population_config, model_config["NUM_ACTORS"]
                )

                fcp_population_size = jax.tree_util.tree_flatten(population)[0][0].shape[0]
                print("FCP population size", fcp_population_size)

                # print(f"normal hstate {init_hstate.shape}")
                # print(f"population hstate {init_population_hstate.shape}")

            if use_population_annealing:

                def _sample_population_annealing_mask(step, rng):
                    return jax.random.uniform(
                        rng, (model_config["NUM_ENVS"],)
                    ) < population_annealing_schedule(step)

                def _make_train_mask(annealing_mask):
                    full_anneal_mask = jnp.tile(annealing_mask, env.num_agents)
                    return jnp.where(full_anneal_mask, train_mask_flat, True)

                rng, _rng = jax.random.split(rng)
                init_population_annealing_mask = _sample_population_annealing_mask(
                    0, _rng
                )

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            (
                train_state,
                checkpoint_states,
                env_state,
                last_obs,
                last_done,
                update_step,
                initial_hstate,
                initial_population_hstate,
                last_population_annealing_mask,
                initial_fcp_pop_agent_idxs,
                obs_history,
                act_history,
                last_partner_action,
                last_action,
                ego_idxs,
                blocked_states_env,
                episode_returns_penalized,
                returned_episode_returns_penalized,
                episode_hit_counts,
                returned_episode_hit_counts,
                ph1_pool_states,
                ph1_probs,
                ph1_pool_ready,
                ph1_recent_ckpts,
                ph1_recent_has,
                rng,
            ) = runner_state

            # jax.debug.print("check3 {x}", x=initial_hstate.flatten()[0])
            ph1_beta_current, ph1_beta_progress = _current_ph1_beta(update_step)

            # COLLECT TRAJECTORIES
            def _env_step(env_step_state, unused):
                (
                    train_state,
                    env_state,
                    last_obs,
                    last_done,
                    update_step,
                    hstate,
                    population_hstate,
                    population_annealing_mask,
                    fcp_pop_agent_idxs,
                    obs_history,
                    act_history,
                    last_partner_action,
                    last_action,
                    ego_idxs,
                    blocked_states_env,
                    ph1_pool_states,
                    episode_returns_penalized,
                    returned_episode_returns_penalized,
                    episode_hit_counts,
                    returned_episode_hit_counts,
                    rng,
                ) = env_step_state

                # [E3T] Dynamic Ego Assignment Logic
                # Check episode completion using last_done
                # last_done is (NUM_ACTORS,), reshaped to check env-wise done
                # OvercookedV2 agents terminate simultaneously
                last_done_reshaped = last_done.reshape(env.num_agents, model_config["NUM_ENVS"])
                episode_done = last_done_reshaped[0] # (NUM_ENVS,)

                rng, _rng = jax.random.split(rng)
                new_random_idxs = jax.random.randint(_rng, (model_config["NUM_ENVS"],), 0, env.num_agents)

                # Update ego_idxs only for environments that just finished
                ego_idxs = jnp.where(episode_done, new_random_idxs, ego_idxs)

                # Calculate actor indices and is_ego
                actor_indices = jnp.repeat(
                    jnp.arange(env.num_agents, dtype=jnp.int32), model_config["NUM_ENVS"]
                )
                
                # Expand ego_idxs (NUM_ENVS,) to match actor_indices (NUM_ACTORS,)
                # actor_indices is [0...0, 1...1] (Agent-Major)
                # target_ego_idxs should be [e0...en, e0...en]
                target_ego_idxs = jnp.tile(ego_idxs, env.num_agents)
                
                is_ego = (actor_indices == target_ego_idxs)
                partner_actor_mask = ~is_ego

                partner_pos = jnp.zeros((model_config["NUM_ENVS"], 2), dtype=jnp.int32)
                if ph1_enabled or stablock_enabled:
                    pos_y, pos_x = _extract_pos_axes(env_state)
                    partner_idxs = (ego_idxs + 1) % env.num_agents
                    # 환경 인덱스를 생성하여 정확히 매칭 (128개의 환경에서 각각 지정된 파트너의 값 추출)
                    env_range = jnp.arange(model_config["NUM_ENVS"])
                    partner_y = pos_y[partner_idxs, env_range]
                    partner_x = pos_x[partner_idxs, env_range]
                    partner_pos = jnp.stack([partner_y, partner_x], axis=-1)

                # [Stablock] 에피소드 종료 시 차단 좌표 재샘플링 (partner만 적용)
                if config.get("PH1_ENABLED", False):
                    # [STA-PH1] env별 pool에서 타겟 샘플링
                    # ph1_pool_states: (Envs, Pool, H, W, C_full)
                    pool_size = ph1_pool_states.shape[1]

                    rng, _rng = jax.random.split(rng)
                    rng_envs = jax.random.split(_rng, model_config["NUM_ENVS"])

                    if ph1_multi_penalty_enabled:

                        def _sample_multi_target(r, pool_env, probs_env):
                            none_targets = jnp.full(
                                (ph1_penalty_slots,) + pool_env.shape[1:],
                                -1,
                                dtype=pool_env.dtype,
                            )

                            def _do_sample(_):
                                targets, _mask, _count = sample_multi_targets_from_pool(
                                    r,
                                    pool_env,
                                    probs_env,
                                    max_penalty_count=ph1_penalty_slots,
                                    single_weight=ph1_multi_penalty_single_weight,
                                    other_weight=ph1_multi_penalty_other_weight,
                                )
                                return targets

                            return jax.lax.cond(
                                ph1_pool_ready,
                                _do_sample,
                                lambda _: none_targets,
                                operand=None,
                            )

                        new_targets = jax.vmap(_sample_multi_target)(
                            rng_envs, ph1_pool_states, ph1_probs
                        )
                    else:

                        def _sample_target(r, probs_env):
                            # Warmup 동안에도 tilde target 샘플링은 유지한다.
                            # 단, pool이 아직 준비되지 않은 경우에만 normal(index=pool_size)로 강제.
                            return jax.lax.cond(
                                ~ph1_pool_ready,
                                lambda _: jnp.int32(pool_size),
                                lambda _: sample_target_idx(r, probs_env),
                                operand=None,
                            )

                        target_idxs = jax.vmap(_sample_target)(rng_envs, ph1_probs)

                        def _pick_candidate(idx, pool_env):
                            is_none = (idx == pool_size)
                            safe_idx = jnp.clip(idx, 0, pool_size - 1)
                            cand = pool_env[safe_idx]
                            none_val = jnp.full_like(cand, -1)
                            return jax.lax.select(is_none, none_val, cand)

                        new_targets = jax.vmap(_pick_candidate)(
                            target_idxs, ph1_pool_states
                        )

                    # done인 환경만 업데이트
                    # blocked_states_env shape: (Envs, ...)
                    # episode_done: (Envs,)
                    # Reshape done to (Envs, 1...)
                    done_expanded = episode_done.reshape(
                        (model_config["NUM_ENVS"],)
                        + (1,) * (blocked_states_env.ndim - 1)
                    )

                    blocked_states_env = jnp.where(
                        done_expanded,
                        new_targets,
                        blocked_states_env,
                    )
                elif stablock_enabled:
                    blocked_states_env = resample_blocked_states(
                        rng,
                        env_state,
                        episode_done,
                        blocked_states_env,
                        stablock_enabled,
                        partner_pos,
                        no_block_prob=stablock_no_block_prob,
                    )
                else:
                    blocked_states_env = blocked_states_env
                if config.get("PH1_ENABLED", False):
                    # PH1 Expansion Logic
                    # blocked_states_env: (Envs, ...)  <-- Same target for both agents
                    # Agent order is [Ag0_E0...Ag0_En, Ag1_E0...Ag1_En] (Agent-Major)

                    blocked_states_actor = jnp.concatenate([blocked_states_env, blocked_states_env], axis=0)
                    
                elif stablock_enabled:
                    blocked_states_actor = expand_blocked_states(
                        blocked_states_env, env.num_agents, is_ego
                    )
                else:
                    blocked_states_actor = jnp.full(
                        (model_config["NUM_ACTORS"], 2), -1, dtype=jnp.int32
                    )

                # Capture z_state from input hstate (before update) for storage
                z_state_in = jnp.zeros((model_config["NUM_ACTORS"], ACTION_DIM))
                if isinstance(hstate, tuple):
                    _, z_state_in = hstate

                # jax.debug.print("check4 {x}", x=hstate.flatten()[0])

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)

                obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape(
                    -1, *state_shape
                )
                if cast_obs_bf16:
                    obs_batch = obs_batch.astype(jnp.bfloat16)

                # --------------------------------------------------------------
                # E3T History Update & Partner Prediction
                # --------------------------------------------------------------
                # 1. History Update
                if (e3t_enabled or (ph1_enabled and use_ph1_partner_pred)) and use_partner_modeling:
                    # last_done이 True이면 히스토리 초기화 (Masking)
                    reset_mask = (1 - last_done.astype(jnp.int32))[:, None]
                    obs_history = obs_history * reset_mask[..., None, None, None].astype(jnp.float32)
                    act_history = act_history * reset_mask
                    
                    # Shift (Left)
                    obs_history = jnp.roll(obs_history, shift=-1, axis=1)
                    act_history = jnp.roll(act_history, shift=-1, axis=1)
                    
                    # Update last element
                    # obs_history[-1] <- current obs (obs_batch)
                    obs_history = obs_history.at[:, -1].set(obs_batch)
                    
                    # act_history[-1] <- last partner action
                    # 에피소드 시작 시점(last_done=True)이면 0으로 처리
                    # last_partner_action은 이전 스텝의 '내' 행동이므로, 파트너의 행동을 얻기 위해 swap해야 함
                    # (NUM_ACTORS = 2 * NUM_ENVS 가정)
                    prev_partner_action = jnp.roll(last_partner_action, shift=model_config["NUM_ENVS"], axis=0)
                    last_partner_action_masked = jnp.where(last_done, 0, prev_partner_action)
                    act_history = act_history.at[:, -1].set(last_partner_action_masked)
                
                # 2. Partner Prediction
                partner_prediction = None
                combined_prediction = jnp.zeros((model_config["NUM_ACTORS"], ACTION_DIM))
                if (e3t_enabled or (ph1_enabled and use_ph1_partner_pred)) and use_partner_modeling:
                    # Extract z_state if available
                    z_in = None
                    if isinstance(hstate, tuple):
                        _, z_in = hstate
                        # z_in is (Batch, Dim)
                    
                    # anchor removed

                    # (1) 실제 파트너 예측 (Ego용)
                    real_prediction = network.apply(train_state.params, obs_history, act_history, z_state=z_in, method='predict_partner')
                    
                    # (2) 무작위 행동 임베딩 (Partner용)
                    # 파트너는 Ego를 예측하지 않고, 무작위 행동(또는 노이즈)을 조건으로 받아 자신의 정책을 수행한다고 가정
                    rng, rng_rand_input = jax.random.split(rng)
                    rand_act_input = jax.random.randint(rng_rand_input, (model_config["NUM_ACTORS"],), 0, ACTION_DIM)
                    rand_pred_input = jax.nn.one_hot(rand_act_input, num_classes=ACTION_DIM)
                    
                    # L2 Normalize (PartnerPredictionModule 출력 스케일과 맞춤)
                    norm = jnp.linalg.norm(rand_pred_input, axis=-1, keepdims=True)
                    rand_pred_input = rand_pred_input / (norm + 1e-6)
                    
                    # (3) Ego vs Partner 구분하여 입력 선택
                    # Ego(Agent 0)는 예측값 사용, Partner(Agent 1)는 무작위 입력 사용 (기본)
                    # [E3T] PH1_ENABLED인 경우에도 모델 훈련 및 V-spec 계산을 위해 모든 에이전트가 Real Prediction 사용
                    use_real_pred = (ph1_enabled and use_ph1_partner_pred) | (stablock_enabled == True) | is_ego
                    combined_prediction = jnp.where(use_real_pred[:, None], real_prediction, rand_pred_input)
                    
                    # (Batch, ActionDim) -> (1, Batch, ActionDim) for Actor input
                    partner_prediction = combined_prediction[jnp.newaxis, ...]

                ac_in = (
                    obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                )

                # [STA-PH1] Handle Extra Return (Latent Embeddings)
                net_out = network.apply(
                    train_state.params,
                    hstate,
                    ac_in,
                    partner_prediction=partner_prediction,
                    blocked_states=(
                        blocked_states_actor
                        if (ph1_enabled or stablock_enabled)
                        else None
                    ),
                )
                
                ph1_extras = {}
                if len(net_out) == 4:
                    hstate, pi, value, ph1_extras = net_out
                else:
                    hstate, pi, value = net_out

                # jax.debug.print("check5 {x}", x=hstate.flatten()[0])

                num_action_choices = pi.logits.shape[-1]
                # policy 정규화 진행해야 하나? => 이미 함수 안에서 되어 있음.
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                # policy_action(action)과 환경에 전달할 action을 분리해서 관리한다.
                #  - trajectory/log_prob 계산에는 action(정책 샘플) 그대로 사용
                #  - 환경 실행 시에는 이후 mask 로직을 거친 action_for_env를 사용한다.
                action_for_env = action
                raw_policy_entropy = pi.entropy().reshape(
                    (model_config["NUM_ACTORS"],)
                )
                log_action_dim = jnp.log(jnp.array(num_action_choices, dtype=jnp.float32))
                safe_log_action_dim = jnp.maximum(log_action_dim, 1e-6)
                policy_entropy = raw_policy_entropy / safe_log_action_dim

                action_pick_mask = jnp.ones(
                    (model_config["NUM_ACTORS"],), dtype=jnp.bool_
                )
                if population is not None:
                    print("Using population")

                    obs_population = obs_batch
                    if isinstance(population, AbstractPolicy):
                        # obs_featurized = jax.vmap(
                        #     env.get_obs_for_type, in_axes=(0, None)
                        # )(env_state.env_state, ObservationType.FEATURIZED)
                        # obs_population = batchify(
                        #     obs_featurized, env.agents, model_config["NUM_ACTORS"]
                        # )
                        pass  # NOTE: Use grid-based observations for FCP agents as well

                    if is_policy_population:
                        rng, _rng = jax.random.split(rng)
                        pop_actions, population_hstate = population.compute_action(
                            obs_population,
                            last_done,
                            population_hstate,
                            _rng,
                        )
                    else:

                        def _compute_population_actions(
                            policy_idx, obs_pop, obs_ld, fcp_h_state
                        ):
                            current_p = jax.tree.map(
                                lambda x: x[policy_idx], population
                            )
                            current_ac_in = (
                                obs_pop[np.newaxis, np.newaxis, :],
                                jnp.array([obs_ld])[np.newaxis, :],
                            )
                            new_fcp_h_state, fcp_pi, _ = population_network.apply(
                                current_p,
                                jax.tree.map(lambda x: x[np.newaxis, :], fcp_h_state),
                                current_ac_in,
                            )
                            fcp_action = fcp_pi.sample(seed=_rng)
                            return fcp_action.squeeze(), jax.tree.map(
                                lambda x: x.squeeze(axis=0), new_fcp_h_state
                            )

                        pop_actions, population_hstate = jax.vmap(
                            _compute_population_actions
                        )(
                            fcp_pop_agent_idxs,
                            obs_population,
                            last_done,
                            population_hstate,
                        )

                    action_pick_mask = train_mask_flat
                    if use_population_annealing:
                        action_pick_mask = _make_train_mask(population_annealing_mask)

                    # use action_pick_mask to select the action from the population or the network
                    action_for_env = jnp.where(action_pick_mask, action_for_env, pop_actions)

                # E3T 알고리즘: 파트너 정책 혼합 (Mixture Partner Policy)
                # [E3T] 파트너 행동 변경 로직 (Mixture Policy)
                # 파트너는 (1-epsilon) 확률로 자신의 정책을 따르고, epsilon 확률로 무작위 행동을 수행함
                if e3t_enabled:
                    # 1. rng 분리 (혼합 정책용)
                    rng, rng_mix = jax.random.split(rng)
                    
                    # 2. 베르누이 마스크 생성 (p=E3T_EPSILON)
                    # True일 경우 무작위 행동을 선택
                    mix_mask = jax.random.bernoulli(rng_mix, p=e3t_epsilon, shape=(model_config["NUM_ACTORS"],))
                    
                    # [중요] 파트너 에이전트(partner_actor_mask)인 경우에만 무작위 행동 혼합을 적용
                    # Ego 에이전트는 자신의 정책을 그대로 따름
                    mix_mask = mix_mask & partner_actor_mask
                    
                    # 3. 완전 무작위 행동 샘플링
                    rng, rng_rand = jax.random.split(rng)
                    # Fix: Ensure shape is (NUM_ACTORS,) not (1, NUM_ACTORS)
                    rand_action = jax.random.randint(
                        rng_rand,
                        (model_config["NUM_ACTORS"],),
                        0,
                        num_action_choices,
                        dtype=action_for_env.dtype,
                    )
                    
                    # 4. 실제 파트너 행동 결정
                    # 마스크가 True이면 무작위 행동, False이면 기존 정책 행동(ego_policy_action) 사용
                    actual_partner_action = jax.lax.select(mix_mask, rand_action, action_for_env.squeeze())
                    
                    # 5. 환경에 전달할 행동 업데이트
                    # Fix: Force int32 type
                    action_for_env = actual_partner_action.astype(jnp.int32)

                # [PH1] epsilon: env마다 확률적으로 한 에이전트를 랜덤 행동으로 교체
                if ph1_enabled and ph1_epsilon > 0.0:
                    rng, rng_eps = jax.random.split(rng)
                    env_mask = jax.random.bernoulli(
                        rng_eps, p=ph1_epsilon, shape=(model_config["NUM_ENVS"],)
                    )

                    rng, rng_pick = jax.random.split(rng)
                    rand_agent = jax.random.randint(
                        rng_pick, (model_config["NUM_ENVS"],), 0, env.num_agents
                    )

                    selected_actor = actor_indices == jnp.tile(rand_agent, env.num_agents)
                    selected_actor = selected_actor & jnp.tile(env_mask, env.num_agents)

                    rng, rng_rand = jax.random.split(rng)
                    rand_action = jax.random.randint(
                        rng_rand,
                        (model_config["NUM_ACTORS"],),
                        0,
                        num_action_choices,
                        dtype=action_for_env.dtype,
                    )

                    action_for_env = jnp.where(selected_actor, rand_action, action_for_env)

                # Ensure action_for_env is squeezed (NUM_ACTORS,) to match carry shape
                # This fixes the scan carry shape mismatch error (int32[512] vs int32[1,512])
                if len(action_for_env.shape) > 1:
                    action_for_env = action_for_env.squeeze()

                # Update last_partner_action for next step history
                last_partner_action = action_for_env

                env_act = unbatchify(
                    action_for_env,
                    env.agents,
                    model_config["NUM_ENVS"],
                    env.num_agents,
                )
                env_act = {k: v.flatten() for k, v in env_act.items()}

                # STEP ENV: 환경 스텝 실행
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, model_config["NUM_ENVS"])

                obsv, env_state, reward, done, info = jax.vmap(
                    step_fn, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)
                
                # original_reward: 환경에서 직접 반환한 원본 보상 (예: 배달 성공 시 +20, 실패 시 -10 등)
                original_reward = jnp.array([reward[a] for a in env.agents])
                reward_no_penalty = {a: reward[a] for a in env.agents}

                # [PH1] Build global full state (with agent info) for this next env_state.
                # This is used for:
                #  - env-specific target pool (recent visited full states)
                #  - latent distance penalty computation
                global_full_next_env0 = None
                if ph1_enabled:
                    global_full_next_env0 = _extract_global_full_obs(env_state)  # (E, H, W, C_full)

                    # Update env-specific ring buffer pool: (E, Pool, H, W, C_full)
                    ph1_pool_states = jnp.roll(ph1_pool_states, shift=-1, axis=1)
                    ph1_pool_states = ph1_pool_states.at[:, -1].set(global_full_next_env0)
                
                # [STA-PH1] Apply Latent Distance Penalty
                # Calculate Distance: || z(next_obs) - z(blocked) ||
                # We need z(next_obs). We can run the network encoder on 'obsv' (next observation).
                
                dist_penalty = jnp.zeros((model_config["NUM_ACTORS"],), dtype=jnp.float32)
                if (
                    ph1_enabled
                    and (global_full_next_env0 is not None)
                    and ("blocked_emb" in ph1_extras)
                    and (ph1_extras["blocked_emb"] is not None)
                ):
                    # Latent distance is computed in the *blocked/global* encoder space.
                    # z_next comes from global full state, not from execution observation.
                    global_full_next_actor = jnp.concatenate(
                        [global_full_next_env0, global_full_next_env0], axis=0
                    )  # (Actors, H, W, C_full)

                    z_next = network.apply(
                        train_state.params,
                        global_full_next_actor,
                        method=network.encode_blocked,
                    ).squeeze()  # (Actors, D or D_proj)
                    z_tilde_slots_src = ph1_extras.get("blocked_emb_slots", None)
                    if z_tilde_slots_src is not None:
                        z_tilde_slots = (
                            z_tilde_slots_src[0]
                            if z_tilde_slots_src.ndim >= 4
                            else z_tilde_slots_src
                        )
                        if z_tilde_slots.ndim == 2:
                            z_tilde_slots = z_tilde_slots[:, None, :]
                    else:
                        z_tilde_src = ph1_extras["blocked_emb"]
                        z_tilde = (
                            z_tilde_src[0] if z_tilde_src.ndim >= 3 else z_tilde_src
                        )
                        if z_tilde.ndim == 1:
                            z_tilde = z_tilde[None, :]
                        z_tilde_slots = z_tilde[:, None, :]

                    num_slots = z_tilde_slots.shape[1]
                    z_next_slots = jnp.expand_dims(z_next, axis=1)
                    lat_dist_slots = jnp.sqrt(
                        jnp.sum((z_next_slots - z_tilde_slots) ** 2, axis=-1)
                    )
                    if blocked_states_actor is not None:
                        if blocked_states_actor.ndim >= 5:
                            flat_blocks = blocked_states_actor.reshape(
                                blocked_states_actor.shape[0],
                                blocked_states_actor.shape[1],
                                -1,
                            )
                            valid_slots = ~jnp.all(flat_blocks == -1, axis=-1)
                        else:
                            flat_blocks = blocked_states_actor.reshape(
                                blocked_states_actor.shape[0], -1
                            )
                            valid_slots = ~jnp.all(flat_blocks == -1, axis=-1)
                            valid_slots = valid_slots[:, None]
                    else:
                        valid_slots = jnp.ones(
                            (model_config["NUM_ACTORS"], num_slots),
                            dtype=jnp.bool_,
                        )
                    if valid_slots.shape[1] == 1 and num_slots > 1:
                        valid_slots = jnp.tile(valid_slots, (1, num_slots))
                    valid_slots = valid_slots[:, :num_slots]
                    lat_dist_slots = jnp.where(valid_slots, lat_dist_slots, 0.0)

                    ph1_omega = config.get("PH1_OMEGA", 1.0)
                    ph1_sigma = config.get("PH1_SIGMA", 1.0)
                    penalty_slots = ph1_omega * jnp.exp(-ph1_sigma * lat_dist_slots)
                    penalty_slots = jnp.where(valid_slots, penalty_slots, 0.0)

                    # [Warmup] Disable penalty if warming up
                    current_global_step = update_step * model_config["NUM_STEPS"] * model_config["NUM_ENVS"]
                    is_ph1_warmup = current_global_step < ph1_warmup_steps
                    penalty_slots = jnp.where(is_ph1_warmup, 0.0, penalty_slots)
                    dist_penalty = jnp.sum(penalty_slots, axis=-1)
                    lat_dist = jnp.sum(lat_dist_slots, axis=-1)

                    # Shared reward penalty (env-level mean)
                    dist_penalty_env = dist_penalty.reshape(
                        env.num_agents, model_config["NUM_ENVS"]
                    ).mean(axis=0)
                    lat_dist_env = lat_dist.reshape(
                        env.num_agents, model_config["NUM_ENVS"]
                    ).mean(axis=0)
                    dist_penalty_env_slots = penalty_slots.reshape(
                        env.num_agents, model_config["NUM_ENVS"], num_slots
                    ).mean(axis=0)
                    lat_dist_env_slots = lat_dist_slots.reshape(
                        env.num_agents, model_config["NUM_ENVS"], num_slots
                    ).mean(axis=0)

                    reward = {k: v - dist_penalty_env for k, v in reward.items()}
                    info["ph1_penalty_env"] = dist_penalty_env
                    info["ph1_dist_env"] = lat_dist_env
                    info["ph1_penalty_env_slots"] = dist_penalty_env_slots
                    info["ph1_dist_env_slots"] = lat_dist_env_slots
                
                # [Stablock] 차단 좌표 도달 시 큰 음의 보상 적용 (partner만 유효)
                if stablock_enabled:
                    pos_y, pos_x = _extract_pos_axes(env_state)
                    pos = jnp.stack([pos_y, pos_x], axis=-1)
                    pos = jnp.swapaxes(pos, 0, 1).reshape(
                        model_config["NUM_ACTORS"], 2
                    )

                    hit_mask = jnp.all(pos == blocked_states_actor, axis=-1)
                    hit_mask_swapped = jnp.all(pos[:, ::-1] == blocked_states_actor, axis=-1)
                    penalty_by_actor = (
                        hit_mask.astype(jnp.float32) * stablock_heavy_penalty
                    )
                    # Debug: print positions, blocked states, and hit mask (first few)
                    # jax.debug.print("[STABLOCK] pos.shape={}", pos.shape)
                    # jax.debug.print("[STABLOCK] pos[0:8]={}", pos[0:8])
                    # jax.debug.print("[STABLOCK] blocked_states_actor[0:8]={}", blocked_states_actor[0:8])
                    # jax.debug.print("[STABLOCK] hit_mask[0:8]={}", hit_mask[0:8])
                    penalty_by_agent = penalty_by_actor.reshape(
                        env.num_agents, model_config["NUM_ENVS"]
                    )
                    reward = {
                        a: reward[a] - penalty_by_agent[i]
                        for i, a in enumerate(env.agents)
                    }
                    info["stablock_hit"] = penalty_by_agent
                    info["stablock_hit_count"] = jnp.full((model_config["NUM_ACTORS"],), hit_mask.sum(), dtype=jnp.int32)
                    info["stablock_hit_count_swapped"] = jnp.full((model_config["NUM_ACTORS"],), hit_mask_swapped.sum(), dtype=jnp.int32)

                    hit_by_env = jnp.sum(
                        (hit_mask & partner_actor_mask).reshape(env.num_agents, model_config["NUM_ENVS"]),
                        axis=0,
                    )
                    new_episode_hit_counts = episode_hit_counts + hit_by_env
                    episode_hit_counts = new_episode_hit_counts * (1 - done["__all__"])
                    returned_episode_hit_counts = jax.lax.select(
                        done["__all__"],
                        new_episode_hit_counts,
                        returned_episode_hit_counts,
                    )
                    returned_episode_hit_counts_actor = jnp.where(
                        is_ego,
                        0,
                        jnp.tile(returned_episode_hit_counts, env.num_agents),
                    )
                    info["returned_episode_hit_count"] = returned_episode_hit_counts_actor

                # anneal_factor: 보상 쉐이핑 감쇠 계수 (학습 초반 1.0 → REW_SHAPING_HORIZON에 걸쳐 0.0으로 선형 감소)
                current_timestep = (
                    update_step * model_config["NUM_STEPS"] * model_config["NUM_ENVS"]
                )
                anneal_factor = rew_shaping_anneal(current_timestep)
                
                # combined_reward: original_reward + (shaped_reward * anneal_factor)
                # - shaped_reward는 환경이 제공하는 중간 단계 보상 (예: 재료 집기, 냄비에 넣기 등)
                # - 학습 초반에는 쉐이핑 보상이 크게 반영되어 학습 신호가 풍부하고,
                # - 학습 후반에는 anneal_factor가 0에 가까워져 원본 보상만 사용 (과최적화 방지)
                reward = jax.tree_util.tree_map(
                    lambda x, y: x + y * anneal_factor, reward, info["shaped_reward"]
                )

                # penalty 포함/미포함 reward 모두 로깅
                shaped_reward = jnp.array(
                    [info["shaped_reward"][a] for a in env.agents]
                )
                combined_reward = jnp.array([reward[a] for a in env.agents])
                combined_reward_no_penalty = (
                    jnp.array([reward_no_penalty[a] for a in env.agents])
                    + shaped_reward * anneal_factor
                )

                info["original_reward"] = original_reward
                info["combined_reward"] = combined_reward
                info["combined_reward_no_penalty"] = combined_reward_no_penalty

                # penalty 포함 episode return (combined_reward의 env-level mean 누적)
                ep_done = done["__all__"]
                combined_reward_env = combined_reward.mean(axis=0)
                new_episode_returns_penalized = (
                    episode_returns_penalized + combined_reward_env
                )
                episode_returns_penalized = new_episode_returns_penalized * (1 - ep_done)
                returned_episode_returns_penalized = jax.lax.select(
                    ep_done,
                    new_episode_returns_penalized,
                    returned_episode_returns_penalized,
                )
                returned_episode_returns_penalized_actor = jnp.tile(
                    returned_episode_returns_penalized, env.num_agents
                )
                info["returned_episode_returns_penalized_env"] = (
                    returned_episode_returns_penalized
                )
                # Deprecated alias: use returned_episode_returns_penalized_env instead.
                info["returned_episode_returns_penalized"] = (
                    returned_episode_returns_penalized_actor
                )

                # --------------------------------------------------------------
                # entropy 디버깅 정보를 info에 추가
                #   - policy_entropy: 각 슬롯별 정책 엔트로피 (항상 기록)
                # --------------------------------------------------------------
                info["policy_entropy"] = policy_entropy

                def _reshape_info_leaf(x):
                    if not hasattr(x, "shape"):
                        return x
                    if (
                        x.ndim >= 2
                        and x.shape[0] == env.num_agents
                        and x.shape[1] == model_config["NUM_ENVS"]
                    ):
                        return x.reshape((model_config["NUM_ACTORS"],) + x.shape[2:])
                    if x.shape == (model_config["NUM_ENVS"],):
                        x = jnp.tile(x, env.num_agents)
                    elif x.ndim >= 2 and x.shape[0] == model_config["NUM_ENVS"]:
                        x = jnp.tile(
                            x,
                            (env.num_agents,) + (1,) * (x.ndim - 1),
                        )
                    if x.shape == ():
                        x = jnp.full((model_config["NUM_ACTORS"],), x)
                    if x.ndim >= 1 and x.shape[0] == model_config["NUM_ACTORS"]:
                        return x.reshape((model_config["NUM_ACTORS"],) + x.shape[1:])
                    return x

                info = {
                    k: (
                        batchify(v, env.agents, model_config["NUM_ACTORS"]).squeeze()
                        if isinstance(v, dict) and all(a in v for a in env.agents)
                        else jax.tree_util.tree_map(_reshape_info_leaf, v)
                    )
                    for k, v in info.items()
                }

                done_batch = batchify(
                    done, env.agents, model_config["NUM_ACTORS"]
                ).squeeze()

                if use_population_annealing:
                    env_steps = (
                        update_step
                        * model_config["NUM_STEPS"]
                        * model_config["NUM_ENVS"]
                    )
                    rng, _rng = jax.random.split(rng)
                    new_population_annealing_mask = jnp.where(
                        done["__all__"],
                        _sample_population_annealing_mask(env_steps, _rng),
                        population_annealing_mask,
                    )
                else:
                    new_population_annealing_mask = population_annealing_mask

                if population is not None and not is_policy_population:
                    new_fcp_pop_agent_idxs = jnp.where(
                        jnp.tile(done["__all__"], env.num_agents),
                        jax.random.randint(
                            _rng, (model_config["NUM_ACTORS"],), 0, fcp_population_size
                        ),
                        fcp_pop_agent_idxs,
                    )
                else:
                    new_fcp_pop_agent_idxs = fcp_pop_agent_idxs

                # E3T: 현재 스텝의 파트너 행동 (Target Label)
                # action_for_env는 현재 스텝의 '내' 행동이므로, 파트너의 행동을 얻기 위해 swap
                current_partner_action = jnp.roll(action_for_env, shift=model_config["NUM_ENVS"], axis=0)

                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    action.squeeze(),
                    value.squeeze(),
                    batchify(reward, env.agents, model_config["NUM_ACTORS"]).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    info,
                    action_pick_mask,
                    obs_history,
                    act_history,
                    current_partner_action,
                    is_ego,
                    z_state_in,
                    blocked_states_actor,
                    last_action,
                    combined_prediction,
                    hstate,
                )

                env_step_state = (
                    train_state,
                    env_state,
                    obsv,
                    done_batch,
                    update_step,
                    hstate,
                    population_hstate,
                    new_population_annealing_mask,
                    new_fcp_pop_agent_idxs,
                    obs_history,
                    act_history,
                    last_partner_action,
                    action.squeeze(),
                    ego_idxs,
                    blocked_states_env,
                    ph1_pool_states,
                    episode_returns_penalized,
                    returned_episode_returns_penalized,
                    episode_hit_counts,
                    returned_episode_hit_counts,
                    rng,
                )
                return env_step_state, transition

            env_step_state = (
                train_state,
                env_state,
                last_obs,
                last_done,
                update_step,
                initial_hstate,
                initial_population_hstate,
                last_population_annealing_mask,
                initial_fcp_pop_agent_idxs,
                obs_history,
                act_history,
                last_partner_action,
                last_action,
                ego_idxs,
                blocked_states_env,
                ph1_pool_states,
                episode_returns_penalized,
                returned_episode_returns_penalized,
                episode_hit_counts,
                returned_episode_hit_counts,
                rng,
            )
            env_step_state, traj_batch = jax.lax.scan(
                _env_step, env_step_state, None, model_config["NUM_STEPS"]
            )
            (
                train_state,
                env_state,
                last_obs,
                last_done,
                update_step,
                next_initial_hstate,
                next_population_hstate,
                last_population_annealing_mask,
                next_fcp_pop_agent_idxs,
                obs_history,
                act_history,
                last_partner_action,
                next_last_action,
                next_ego_idxs,
                blocked_states_env,
                ph1_pool_states,
                episode_returns_penalized,
                returned_episode_returns_penalized,
                episode_hit_counts,
                returned_episode_hit_counts,
                rng,
            ) = env_step_state

            # jax.debug.print("check7 {x}", x=next_initial_hstate)

            # print("Hilfeeeee", traj_batch.done.shape, traj_batch.action.shape)

            # CALCULATE ADVANTAGE
            last_obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape(
                -1, *state_shape
            )
            if cast_obs_bf16:
                last_obs_batch = last_obs_batch.astype(jnp.bfloat16)
            ac_in = (
                last_obs_batch[np.newaxis, :],
                last_done[np.newaxis, :],
            )

            # [E3T] Calculate partner_prediction for last_val (Value Function)
            partner_prediction = None
            # [Stablock/E3T] last_val 계산 시점의 ego 마스크
            target_ego_idxs = jnp.tile(next_ego_idxs, env.num_agents)
            is_ego_last = (actor_indices == target_ego_idxs)
            if (e3t_enabled or (ph1_enabled and use_ph1_partner_pred)) and use_partner_modeling:
                # 1. Predict
                pred_logits = network.apply(train_state.params, obs_history, act_history, method='predict_partner')
                
                # 2. Random Input for Partner
                rng, rng_rand_input = jax.random.split(rng)
                rand_act_input = jax.random.randint(rng_rand_input, (model_config["NUM_ACTORS"],), 0, ACTION_DIM)
                rand_pred_input = jax.nn.one_hot(rand_act_input, num_classes=ACTION_DIM)
                norm = jnp.linalg.norm(rand_pred_input, axis=-1, keepdims=True)
                rand_pred_input = rand_pred_input / (norm + 1e-6)
                
                # 3. Combine (dynamic ego idx 기준)
                # PH1 활성화 시 모든 에이전트가 Real Prediction 사용 (V-gap 정확도를 위해)
                use_real_pred_last = (ph1_enabled and use_ph1_partner_pred) | (stablock_enabled == True) | is_ego_last
                combined_prediction = jnp.where(use_real_pred_last[:, None], pred_logits, rand_pred_input)
                
                # 4. Add Time Dimension (1, Batch, ActionDim)
                partner_prediction = combined_prediction[jnp.newaxis, ...]

            # [Stablock] last_val 계산에도 blocked_states를 전달
            blocked_states_actor_last = None
            if config.get("PH1_ENABLED", False):
                # Same target for both agents in state mode
                blocked_states_actor_last = jnp.concatenate([blocked_states_env, blocked_states_env], axis=0)
            elif stablock_enabled:
                blocked_states_actor_last = expand_blocked_states(
                    blocked_states_env, env.num_agents, is_ego_last
                )

            net_out = network.apply(
                train_state.params,
                next_initial_hstate,
                ac_in,
                partner_prediction=partner_prediction,
                blocked_states=(
                    blocked_states_actor_last
                    if (ph1_enabled or stablock_enabled)
                    else None
                ),
            )
            
            if len(net_out) == 4:
                _, _, last_val, _ = net_out
            else:
                _, _, last_val = net_out

            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = (
                        reward + model_config["GAMMA"] * next_value * (1 - done) - value
                    )
                    gae = (
                        delta
                        + model_config["GAMMA"]
                        * model_config["GAE_LAMBDA"]
                        * (1 - done)
                        * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state_and_rng, batch_info):
                    train_state, mb_rng = train_state_and_rng
                    mb_rng, loss_rng = jax.random.split(mb_rng)
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets, loss_rng):
                        hstate = init_hstate
                        if hstate is not None:
                            if isinstance(hstate, tuple):
                                hstate = tuple(x.squeeze(axis=0) for x in hstate)
                            else:
                                hstate = hstate.squeeze(axis=0)
                            # hstate = jax.lax.stop_gradient(hstate)

                        train_mask = True
                        if population is not None:
                            train_mask = jax.lax.stop_gradient(traj_batch.train_mask)

                        # --------------------------------------------------------------
                        # E3T Partner Prediction Loss & Gradient Blocking
                        # --------------------------------------------------------------
                        pred_loss = 0.0
                        pred_accuracy = 0.0
                        partner_prediction = None
                        
                        use_ph1 = config.get("PH1_ENABLED", False)

                        if (e3t_enabled or use_ph1) and use_partner_modeling:
                            # 1. Forward Pass for Prediction
                            # PPO 미니배치는 셔플되어 시간 연속성이 없으므로 Scan을 사용할 수 없습니다.
                            # 따라서 단일 스텝 예측(predict_partner)을 사용합니다.
                            
                            # traj_batch.obs_history: (Minibatch, 5, H, W, C)
                            # traj_batch.act_history: (Minibatch, 5)
                            obs_hist_flat = traj_batch.obs_history
                            act_hist_flat = traj_batch.act_history
                            
                            # --------------------------------------------------------------
                            # [E3T Implementation] Partner Prediction Loss
                            # --------------------------------------------------------------
                            # 인코더는 파트너의 다음 행동을 예측하도록 학습합니다.
                            # z_state는 Rollout 시점의 값을 사용하여 일관성을 유지합니다 (Carry state matching).
                            
                            pred_logits = network.apply(params, obs_hist_flat, act_hist_flat, z_state=traj_batch.z_state, method='predict_partner')
                            
                            # Policy Input (Actor Training) uses the same prediction
                            pred_logits_policy = pred_logits

                            # 2. Label Matching & Loss Calculation
                            # 예측값: (Minibatch, ActionDim)
                            pred_logits_flat = pred_logits
                            
                            # 정답값: (Minibatch,)
                            target_labels_flat = traj_batch.partner_action.astype(jnp.int32)
                            
                            # 3. Compute Cross-Entropy Loss (예측 손실 계산 - Raw 기준)
                            pred_loss_vec = optax.softmax_cross_entropy_with_integer_labels(logits=pred_logits_flat, labels=target_labels_flat)
                            
                            # [E3T] Ego/Partner 모두 prediction loss에 기여
                            pred_loss = pred_loss_vec.mean()
                            
                            # Apply Coefficient
                            pred_loss = pred_loss * config.get("PRED_LOSS_COEF", 1.0)
                            
                            # Calculate Accuracy
                            pred_labels = jnp.argmax(pred_logits_flat, axis=-1)
                            pred_accuracy = jnp.mean(pred_labels == target_labels_flat)
                            
                            # 4. Stop Gradient for Ego Policy (Actor 입력용)
                            # Actor-Critic 네트워크에 입력으로 주기 위해 gradient를 차단합니다.
                            partner_prediction_flat = jax.lax.stop_gradient(pred_logits_policy)
                            # (Minibatch, ActionDim) -> (Minibatch, 1, ActionDim) for Actor Input (if needed)
                            # 하지만 Actor는 (Batch, ...)를 기대하므로 (Minibatch, 1, ActionDim)으로 확장
                            partner_prediction = partner_prediction_flat[:, jnp.newaxis, :]

                        # [PH1/E3T Fix] If PH1 enables partner modeling but condition above failed or partner_prediction still None?
                        # Actually 'use_partner_modeling' is True in config.
                        # Wait, the condition `if alg_name == "E3T" and use_partner_modeling:` skipped PH1.
                        # I changed it to `if (alg_name == "E3T" or use_ph1) and use_partner_modeling:` above.
                        
                        # Fallback for shape correctness if prediction is still None but network expects it
                        # The network expects it if it was initialized with it.
                        # Initialization uses dummy_partner_prediction if PH1 is enabled.
                        # So we MUST provide partner_prediction here too.
                        
                        if partner_prediction is None and ((e3t_enabled or use_ph1) and use_partner_modeling):
                             # Should not happen with the fix above, but as a safety measure for correct shape:
                             # Create dummy zeros (Minibatch, 1, ActionDim)
                             # But this means we skip prediction loss calculation? Yes, if logic falls here.
                             # But with my fix above, it should enter the block.
                             pass

                        # RERUN NETWORK
                        # Actor-Critic 네트워크 재실행 (Value, Policy 계산)
                        # partner_prediction is used only when partner modeling is enabled.
                        net_out = network.apply(
                            params,
                            hstate,
                            (traj_batch.obs, traj_batch.done),
                            partner_prediction=partner_prediction,
                            blocked_states=(
                                traj_batch.blocked_states
                                if (ph1_enabled or stablock_enabled)
                                else None
                            ),
                        )
                        
                        if len(net_out) == 4:
                            _, pi, value, _ = net_out
                        else:
                            _, pi, value = net_out

                        # print("value shape", value.shape)
                        # print("targets shape", targets.shape)
                        # print("pi shape", pi.logits.shape)

                        log_prob = pi.log_prob(traj_batch.action)

                        # def safe_mean(x, mask):
                        #     x_safe = jnp.where(mask, x, 0.0)
                        #     total = jnp.sum(x_safe)
                        #     count = jnp.sum(mask)
                        #     return total / count

                        # def safe_std(x, mask, eps=1e-8):
                        #     m = safe_mean(x, mask)
                        #     diff_sq = (x - m) ** 2
                        #     variance = safe_mean(diff_sq, mask)
                        #     return jnp.sqrt(variance + eps)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-model_config["CLIP_EPS"], model_config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(
                            value_losses, value_losses_clipped
                        ).mean(where=train_mask)
                        # value_loss = safe_mean(value_loss, train_mask)

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean(where=train_mask)) / (
                            gae.std(where=train_mask) + 1e-8
                        )
                        # gae_mean = safe_mean(gae, train_mask)
                        # gae_std = safe_std(gae, train_mask)
                        # gae = (gae - gae_mean) / (gae_std + 1e-8)

                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - model_config["CLIP_EPS"],
                                1.0 + model_config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean(where=train_mask)
                        # loss_actor = safe_mean(loss_actor, train_mask)
                        entropy = pi.entropy().mean(where=train_mask)
                        # entropy = safe_mean(pi.entropy(), train_mask)
                        ratio = ratio.mean(where=train_mask)
                        # ratio = safe_mean(ratio, train_mask)

                        total_loss = (
                            loss_actor
                            + model_config["VF_COEF"] * value_loss
                            - model_config["ENT_COEF"] * entropy
                            + pred_loss
                        )

                        return total_loss, (
                            value_loss,
                            loss_actor,
                            entropy,
                            ratio,
                            pred_loss,
                            pred_accuracy,
                        )

                    def _perform_update():
                        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                        total_loss, grads = grad_fn(
                            train_state.params,
                            init_hstate,
                            traj_batch,
                            advantages,
                            targets,
                            loss_rng,
                        )

                        # jax.debug.print(
                        #     "grads {x}, hstate {y}, mask {z}",
                        #     x=jax.tree_util.tree_flatten(grads)[0][0][0],
                        #     y=init_hstate.flatten()[0],
                        #     z=traj_batch.train_mask.sum(),
                        # )

                        new_train_state = train_state.apply_gradients(grads=grads)
                        return (new_train_state, mb_rng), total_loss

                    def _no_op():
                        # jax.debug.print("No update")
                        return (train_state, mb_rng), (
                            0.0,
                            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                        )

                    # jax.debug.print(
                    #     "train_mask {x}, {y}",
                    #     x=traj_batch.train_mask.sum(),
                    #     y=traj_batch.train_mask.any(),
                    # )

                    (train_state, mb_rng), total_loss = jax.lax.cond(
                        traj_batch.train_mask.any(),
                        _perform_update,
                        _no_op,
                    )
                    return (train_state, mb_rng), total_loss

                train_state, init_hstate, traj_batch, advantages, targets, rng = (
                    update_state
                )
                rng, _rng = jax.random.split(rng)

                num_actors = model_config["NUM_ACTORS"]

                hstate = init_hstate
                if hstate is not None:
                    if isinstance(hstate, tuple):
                        # print("hstate shape (tuple)", [x.shape for x in hstate])
                        hstate = tuple(x[jnp.newaxis, :] for x in hstate)
                        # print("hstate shape (tuple)", [x.shape for x in hstate])
                    else:
                        # print("hstate shape", hstate.shape)
                        hstate = hstate[jnp.newaxis, :]
                        # print("hstate shape", hstate.shape)

                batch = (
                    hstate,
                    traj_batch,
                    advantages.squeeze(),
                    targets.squeeze(),
                )
                # print(
                #     "batch shapes:",
                #     batch[0].shape,
                #     batch[1].obs.shape,
                #     batch[1].done.shape,
                #     batch[2].shape,
                #     batch[3].shape,
                # )
                # print("hstate shape", hstate.shape)

                permutation = jax.random.permutation(_rng, num_actors)

                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], model_config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                (train_state, rng), total_loss = jax.lax.scan(
                    _update_minbatch, (train_state, rng), minibatches
                )

                update_state = (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            rng, _rng = jax.random.split(rng)
            update_state = (
                train_state,
                initial_hstate,
                traj_batch,
                advantages,
                targets,
                _rng,
            )
            # UPDATE_EPOCHS 횟수만큼 미니배치 업데이트를 반복 수행
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, model_config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            
            # 환경 스텝 수집 중 기록된 정보를 메트릭 딕셔너리로 가져옴
            # (shaped_reward, original_reward, anneal_factor, combined_reward 등 포함)
            metric = traj_batch.info

            # [STA-PH1] Logging Logic
            if config.get("PH1_ENABLED", False):
                 ph1_pen = metric.get("ph1_penalty", 0.0) # Mean penalty (scalar if meaned, but here it's from info)
                 # metric comes from traj_batch.info which is already meaned in the metric block below?
                 # No, 'metric' variable here is exactly 'traj_batch.info'.
                 # And standard logging takes mean.
                 
                 # We want specific logging: Agent 0 Reward, Agent 1 Reward, Penalties
                 # The 'metric' dict contains 'ph1_penalty' as (steps, actors) array?
                 # info was: info["ph1_penalty"] = dist_penalty (Actors,)
                 # traj_batch stacks them: (Steps, Actors)
                 
                 # Separate Agent 0 and Agent 1 data
                 # Agents are interleaved or stacked? 
                 # batchify stacks: [A0_E0, ..., A0_En, A1_E0, ..., A1_En]
                 # So first half is Agent 0, second half is Agent 1
                 
                 n_actors = model_config["NUM_ACTORS"]
                 half = n_actors // 2
                 
                 # Helper to extract mean for agent 0 and 1
                 def _get_agent_means(key):
                     if key not in metric: return 0.0, 0.0
                     val = metric[key] # (Steps, Actors)
                     # Mean over steps first
                     val_mean = val.mean(axis=0) # (Actors,)
                     return val_mean[:half].mean(), val_mean[half:].mean()

                 if "ph1_penalty_env" in metric:
                     metric["ph1_penalty_env_mean"] = metric["ph1_penalty_env"].mean()
                 if "ph1_dist_env" in metric:
                     metric["ph1_dist_env_mean"] = metric["ph1_dist_env"].mean()
                 if "ph1_penalty_env_slots" in metric:
                     for slot_idx in range(ph1_penalty_slots):
                         metric[f"ph1_penalty_env_slot{slot_idx + 1}_mean"] = (
                             metric["ph1_penalty_env_slots"][..., slot_idx].mean()
                         )
                 if "ph1_dist_env_slots" in metric:
                     for slot_idx in range(ph1_penalty_slots):
                         metric[f"ph1_dist_env_slot{slot_idx + 1}_mean"] = (
                             metric["ph1_dist_env_slots"][..., slot_idx].mean()
                         )
                 
                 # Process combined_reward (penalized)
                 rew0, rew1 = _get_agent_means("combined_reward")
                 metric["reward_agent0_penalized"] = rew0
                 metric["reward_agent1_penalized"] = rew1
                 
                 # Process combined_reward_no_penalty (pure)
                 pure0, pure1 = _get_agent_means("combined_reward_no_penalty")
                 metric["reward_agent0_pure"] = pure0
                 metric["reward_agent1_pure"] = pure1

            # --------------------------------------------------------------
            # entropy 통계 파생 메트릭
            #   - policy_entropy_mean: rollout 전체 평균 엔트로피
            # --------------------------------------------------------------
            if "policy_entropy" in metric:
                metric["policy_entropy_mean"] = metric["policy_entropy"].mean()

            # 손실 함수 관련 메트릭 추출
            total_loss, aux_data = loss_info
            (
                value_loss,
                loss_actor,
                entropy,
                ratio,
                pred_loss,
                pred_accuracy,
            ) = aux_data

            # PPO 학습 손실 메트릭 추가
            metric["total_loss"] = total_loss      # 전체 손실 (actor + value + entropy + pred)
            metric["value_loss"] = value_loss      # 가치 함수(critic) MSE 손실
            metric["loss_actor"] = loss_actor      # 정책(actor) clipped surrogate 손실
            metric["entropy"] = entropy            # 정책 엔트로피 (탐험 정도)
            metric["ratio"] = ratio                # PPO ratio (new_prob / old_prob)
            metric["pred_loss"] = pred_loss        # 파트너 행동 예측 손실
            metric["pred_accuracy"] = pred_accuracy # 파트너 행동 예측 정확도
            metric["ph1_beta_current"] = ph1_beta_current
            metric["ph1_beta_progress"] = ph1_beta_progress

            # 모든 메트릭 값을 배치/스텝 차원에 대해 평균 계산 (스칼라로 축약)
            # 이미 스칼라인 경우(PH1 메트릭 등)는 그대로 둔다.
            def _safe_mean(x):
                if hasattr(x, "mean"):
                    return x.mean()
                return x

            metric = jax.tree_util.tree_map(_safe_mean, metric)

            # 업데이트 스텝 카운터 증가 및 메트릭에 기록
            update_step += 1
            metric["update_step"] = update_step
            # 환경과 상호작용한 총 스텝 수 계산 (update * rollout_length * num_envs)
            metric["env_step"] = (
                update_step * model_config["NUM_STEPS"] * model_config["NUM_ENVS"]                                                                 
            )

            # --------------------------------------------------------------
            # Hard-stop on NaN/Inf in training state/losses/metrics
            # --------------------------------------------------------------
            if abort_on_nan:
                def _all_finite_tree(tree):
                    leaves = jax.tree_util.tree_leaves(tree)
                    if len(leaves) == 0:
                        return jnp.array(True)
                    flags = [jnp.all(jnp.isfinite(x)) for x in leaves]
                    return jnp.all(jnp.stack(flags))

                params_finite = _all_finite_tree(train_state.params)
                losses_finite = jnp.all(
                    jnp.stack(
                        [
                            jnp.all(jnp.isfinite(total_loss)),
                            jnp.all(jnp.isfinite(value_loss)),
                            jnp.all(jnp.isfinite(loss_actor)),
                            jnp.all(jnp.isfinite(entropy)),
                            jnp.all(jnp.isfinite(ratio)),
                            jnp.all(jnp.isfinite(pred_loss)),
                            jnp.all(jnp.isfinite(pred_accuracy)),
                        ]
                    )
                )
                metric_finite = _all_finite_tree(metric)
                all_finite = params_finite & losses_finite & metric_finite

                def _raise_nan_abort(us, es):
                    us_i = int(np.array(us))
                    es_i = int(np.array(es))
                    raise RuntimeError(
                        f"[ABORT_ON_NAN] Non-finite detected during training at "
                        f"update_step={us_i}, env_step={es_i}. Aborting run."
                    )

                jax.lax.cond(
                    all_finite,
                    lambda _: None,
                    lambda _: jax.debug.callback(
                        _raise_nan_abort,
                        update_step,
                        metric["env_step"],
                    ),
                    operand=None,
                )

            # WandB 로깅 콜백: JAX 계산 그래프 밖에서 실행되도록 debug.callback 사용
            # 각 시드별로 구분된 네임스페이스(rng{seed}/)를 prefix로 추가하여 로깅
            def callback(metric, original_seed):
                # vmap을 사용하면 metric과 original_seed가 배치(배열) 형태로 들어옵니다.
                # 따라서 배열인 경우 순회하며 각각 로깅해야 합니다.
                
                # numpy 변환 (호스트 측 실행이므로 안전)
                original_seed = np.array(original_seed)
                
                if original_seed.ndim > 0:
                    # 배치가 있는 경우 (vmap 사용 시)
                    for i in range(original_seed.shape[0]):
                        seed = original_seed[i]
                        # 해당 시드의 메트릭만 추출
                        single_metric = {k: v[i] for k, v in metric.items()}
                        
                        # print(f"[DEBUG] Logging for seed {seed}")
                        wandb_log = {f"rng{int(seed)}/{k}": v for k, v in single_metric.items()}
                        wandb.log(wandb_log)
                else:
                    # 스칼라인 경우 (단일 시드)
                    # print(f"[DEBUG] Logging for seed {original_seed}")
                    wandb_log = {f"rng{int(original_seed)}/{k}": v for k, v in metric.items()}
                    wandb.log(wandb_log)

            jax.debug.callback(callback, metric, original_seed)

            # --------------------------------------------------------------
            # PH1 online evaluation callback (normal / recent / random tilde{s})
            # --------------------------------------------------------------
            if ph1_eval_enabled:
                steps_per_update = model_config["NUM_STEPS"] * model_config["NUM_ENVS"]
                curr_env_step = update_step * steps_per_update
                prev_env_step = curr_env_step - steps_per_update

                if ph1_eval_every_env_steps > 0:
                    do_eval = (curr_env_step // ph1_eval_every_env_steps) > (
                        prev_env_step // ph1_eval_every_env_steps
                    )
                else:
                    do_eval = (update_step % ph1_eval_every_updates) == 0

                # recent batch candidate: most recent pool slice (env-major)
                recent_tilde_batch = ph1_pool_states[:, -1]

                if ph1_eval_video_every_env_steps > 0:
                    do_video = (curr_env_step // ph1_eval_video_every_env_steps) > (
                        prev_env_step // ph1_eval_video_every_env_steps
                    )
                else:
                    do_video = do_eval

                def _ph1_eval_cb(
                    params,
                    update_step_host,
                    curr_env_step_host,
                    recent_tilde_batch_host,
                    original_seed_host,
                    do_eval_host,
                    do_video_host,
                ):
                    from overcooked_v2_experiments.ppo.ph1_online_eval import (
                        save_ph1_video_snapshot,
                    )

                    update_step_val = int(np.array(update_step_host))
                    curr_env_step_val = int(np.array(curr_env_step_host))
                    seed_np = np.array(original_seed_host)
                    if seed_np.ndim > 0:
                        # 배치 시드가 들어오면 안정적으로 첫 요소를 스칼라로 사용
                        seed_val = int(seed_np.flat[0])
                    else:
                        seed_val = int(seed_np)
                    dedup_key = (seed_val, curr_env_step_val)
                    with _PH1_EVAL_LOCK:
                        if dedup_key in _PH1_EVAL_SEEN_KEYS:
                            return
                        _PH1_EVAL_SEEN_KEYS.add(dedup_key)

                    do_eval_val = bool(np.array(do_eval_host))
                    do_video_val = bool(np.array(do_video_host))
                    recent_batch_np = (
                        None if recent_tilde_batch_host is None else np.array(recent_tilde_batch_host)
                    )

                    snapshot_required = do_eval_val or (ph1_eval_defer_video and do_video_val)

                    # Offline-only policy: during training, never run eval rollout.
                    # We only capture snapshots to replay eval/video after training.
                    if snapshot_required:
                        try:
                            snap_dir = save_ph1_video_snapshot(
                                params=params,
                                config=config,
                                update_step=curr_env_step_val,
                                recent_tilde_batch=recent_batch_np,
                                seed=seed_val,
                            )
                            if snap_dir is not None:
                                try:
                                    wandb.log(
                                        {
                                            f"rng{seed_val}/eval_ph1/deferred_video_snapshot_path": snap_dir,
                                            f"rng{seed_val}/eval_ph1/snapshot_do_eval": int(do_eval_val),
                                            f"rng{seed_val}/eval_ph1/snapshot_do_video": int(do_video_val),
                                            "eval/update_step": update_step_val,
                                        }
                                    )
                                except Exception:
                                    pass
                            else:
                                try:
                                    wandb.log(
                                        {
                                            f"rng{seed_val}/eval_ph1/snapshot_warning": 1.0,
                                            "eval/update_step": update_step_val,
                                        }
                                    )
                                except Exception:
                                    pass
                        except Exception as e:
                            try:
                                wandb.log(
                                    {
                                        f"rng{seed_val}/eval_ph1/snapshot_error": str(e),
                                        "eval/update_step": update_step_val,
                                    }
                                )
                            except Exception:
                                pass

                should_call_eval_cb = do_eval | (jnp.array(ph1_eval_defer_video) & do_video)

                jax.lax.cond(
                    should_call_eval_cb,
                    lambda _: jax.debug.callback(
                        _ph1_eval_cb,
                        train_state.params,
                        update_step,
                        curr_env_step,
                        recent_tilde_batch,
                        original_seed,
                        do_eval,
                        do_video,
                    ),
                    lambda _: None,
                    operand=None,
                )

            if num_checkpoints > 0:
                checkpoint_idx_selector = checkpoint_steps == update_step
                checkpoint_idx = jnp.argmax(checkpoint_idx_selector)
                checkpoint_states = jax.lax.cond(
                    jnp.any(checkpoint_idx_selector),
                    _update_checkpoint,
                    lambda c, _p, _i: c,
                    checkpoint_states,
                    train_state.params,
                    checkpoint_idx,
                )
                recent_tilde_ckpt, has_recent_ckpt = _extract_recent_tilde_from_pool(ph1_pool_states)
                ph1_recent_ckpts = jax.lax.cond(
                    jnp.any(checkpoint_idx_selector),
                    lambda x: x.at[checkpoint_idx].set(recent_tilde_ckpt),
                    lambda x: x,
                    ph1_recent_ckpts,
                )
                ph1_recent_has = jax.lax.cond(
                    jnp.any(checkpoint_idx_selector),
                    lambda x: x.at[checkpoint_idx].set(has_recent_ckpt),
                    lambda x: x,
                    ph1_recent_has,
                )

            # [STA-PH1] Update per-env sampling probabilities using the env-specific pool.
            next_ph1_pool_states = ph1_pool_states
            next_ph1_probs = ph1_probs

            if ph1_enabled:
                use_partner_pred = ph1_enabled and use_ph1_partner_pred and use_partner_modeling

                # Use the most recent rollout step as the reference batch per environment.
                # Shape conventions:
                #  - traj_batch.obs: (T, Actors, H, W, C_obs)
                #  - traj_batch.done: (T, Actors)
                #  - traj_batch.hstate: pytree with leading (T, Actors, ...)
                # We reshape actors -> (Agents, Envs, ...), then take env-wise batches of size B=Agents.
                obs_last = traj_batch.obs[-1].reshape(env.num_agents, model_config["NUM_ENVS"], *state_shape)
                done_last = traj_batch.done[-1].reshape(env.num_agents, model_config["NUM_ENVS"])

                h_last = jax.tree_util.tree_map(
                    lambda x: x[-1].reshape(env.num_agents, model_config["NUM_ENVS"], *x.shape[2:]),
                    traj_batch.hstate,
                )

                if use_partner_pred:
                    pp_last = traj_batch.partner_prediction[-1].reshape(
                        env.num_agents, model_config["NUM_ENVS"], ACTION_DIM
                    )
                else:
                    pp_last = jnp.zeros((env.num_agents, model_config["NUM_ENVS"], ACTION_DIM), dtype=jnp.float32)
                ph1_prob_blocked_slots = (
                    ph1_penalty_slots if ph1_multi_penalty_enabled else 1
                )

                def _compute_env_probs(obs_env, done_env, h_env, pool_env, pp_env):
                    # obs_env: (Agents, H, W, C_obs)
                    # done_env: (Agents,)
                    # h_env: pytree with (Agents, ...)
                    # pool_env: (Pool, H, W, C_full)
                    probs_env, _v = compute_ph1_probs(
                        network.apply,
                        train_state.params,
                        obs_env,
                        done_env,
                        h_env,
                        pool_env,
                        pp_env,
                        use_partner_pred=use_partner_pred,
                        blocked_input_slots=ph1_prob_blocked_slots,
                        beta=ph1_beta_current,
                        normal_prob=config.get("PH1_NORMAL_PROB", 0.5),
                    )
                    return probs_env

                computed_probs = jax.vmap(_compute_env_probs)(
                    jnp.swapaxes(obs_last, 0, 1),
                    jnp.swapaxes(done_last, 0, 1),
                    jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), h_last),
                    next_ph1_pool_states,
                    jnp.swapaxes(pp_last, 0, 1),
                )

                next_ph1_probs = jax.lax.select(ph1_pool_ready, computed_probs, ph1_probs)
                ph1_pool_ready = True

            runner_state = (
                train_state,
                checkpoint_states,
                env_state,
                last_obs,
                last_done,
                update_step,
                next_initial_hstate,
                next_population_hstate,
                last_population_annealing_mask,
                next_fcp_pop_agent_idxs,
                obs_history,
                act_history,
                last_partner_action,
                next_last_action,
                next_ego_idxs,
                blocked_states_env,
                episode_returns_penalized,
                returned_episode_returns_penalized,
                episode_hit_counts,
                returned_episode_hit_counts,
                next_ph1_pool_states,
                next_ph1_probs,
                ph1_pool_ready,
                ph1_recent_ckpts,
                ph1_recent_has,
                rng,
            )
            return runner_state, metric

        initial_update_step = 0
        if update_step_offset is not None:
            initial_update_step = update_step_offset

        initial_checkpoints = jax.tree_util.tree_map(
            lambda p: jnp.zeros((num_checkpoints,) + p.shape, p.dtype),
            train_state.params,
        )

        if num_checkpoints > 0:
            initial_checkpoints = jax.lax.cond(
                (checkpoint_steps[0] == 0) & (initial_update_step == 0),
                _update_checkpoint,
                lambda c, _p, _i: c,
                initial_checkpoints,
                train_state.params,
                0,
            )

        init_fcp_pop_idxs = None
        if population is not None and not is_policy_population:
            init_fcp_pop_idxs = jax.random.randint(
                _rng, (model_config["NUM_ACTORS"],), 0, fcp_population_size
            )

        rng, _rng = jax.random.split(rng)

        # E3T History Buffers
        initial_obs_history = jnp.zeros((model_config["NUM_ACTORS"], 5, *state_shape))
        initial_act_history = jnp.zeros((model_config["NUM_ACTORS"], 5), dtype=jnp.int32)
        initial_last_partner_action = jnp.zeros((model_config["NUM_ACTORS"],), dtype=jnp.int32)
        initial_last_action = jnp.zeros((model_config["NUM_ACTORS"],), dtype=jnp.int32)

        # [E3T] Initial Ego Indices (Random init)
        initial_ego_idxs = jax.random.randint(_rng, (model_config["NUM_ENVS"],), 0, env.num_agents)

        # [Stablock] 에피소드 시작 시 partner에게 부여할 차단 좌표 초기화
        partner_pos = jnp.zeros((model_config["NUM_ENVS"], 2), dtype=jnp.int32)
        if ph1_enabled or stablock_enabled:
            pos_y, pos_x = _extract_pos_axes(env_state)
            partner_idxs = (initial_ego_idxs + 1) % env.num_agents
            env_range = jnp.arange(model_config["NUM_ENVS"])
            partner_y = pos_y[partner_idxs, env_range]
            partner_x = pos_x[partner_idxs, env_range]
            partner_pos = jnp.stack([partner_y, partner_x], axis=-1)
        if ph1_enabled:
            # PH1 blocked target is always the global full state.
            if ph1_multi_penalty_enabled:
                init_shape = (
                    model_config["NUM_ENVS"],
                    ph1_penalty_slots,
                ) + ph1_block_shape
            else:
                init_shape = (model_config["NUM_ENVS"],) + ph1_block_shape
            initial_blocked_states_env = jnp.full(init_shape, -1.0, dtype=jnp.float32)
        elif stablock_enabled:
            initial_blocked_states_env = initialize_blocked_states(
                _rng,
                env_state,
                stablock_enabled,
                model_config["NUM_ENVS"],
                partner_pos,
                no_block_prob=stablock_no_block_prob,
            )
        else:
            initial_blocked_states_env = jnp.full(
                (model_config["NUM_ENVS"], 2), -1, dtype=jnp.int32
            )
        # Debug: print initial blocked state set (first few envs)
        # jax.debug.print("Initial blocked_states_env shape: {}", initial_blocked_states_env.shape)
        # jax.debug.print("Initial blocked_states_env sample (first 8): {}", initial_blocked_states_env[:8])
        
        # Debug: print possible blocked coords for first env
        if stablock_enabled:
            def print_possible_coords(grid_np, partner_pos_np):
                coords = enumerate_reachable_positions(grid_np, partner_pos_np)
                # print(f"Possible blocked coords for first env (N={coords.shape[0]}): {coords}")
            # jax.debug.callback(print_possible_coords, env_state.env_state.grid[0], partner_pos[0])

        # [PH1] Initialize Target Pool
        ph1_pool_size = config.get("PH1_POOL_SIZE", 100)
        
        # Per-env ring buffer pool: (Envs, Pool, H, W, C_full)
        if ph1_enabled:
            pool_shape = (model_config["NUM_ENVS"], ph1_pool_size) + ph1_block_shape
            initial_ph1_pool_states = jnp.full(pool_shape, -1.0, dtype=jnp.float32)
        else:
            # Placeholder to keep runner_state structure consistent
            initial_ph1_pool_states = jnp.zeros((model_config["NUM_ENVS"], ph1_pool_size, 1), dtype=jnp.float32)

        # Per-env probs: (Envs, Pool+1) (Last is None)
        initial_ph1_probs = jnp.zeros((model_config["NUM_ENVS"], ph1_pool_size + 1), dtype=jnp.float32)
        initial_ph1_probs = initial_ph1_probs.at[:, -1].set(1.0)

        initial_ph1_pool_ready = False
        if ph1_enabled:
            initial_ph1_recent_ckpts = jnp.zeros((num_checkpoints,) + ph1_block_shape, dtype=jnp.float32)
        else:
            initial_ph1_recent_ckpts = jnp.zeros((num_checkpoints, 1), dtype=jnp.float32)
        initial_ph1_recent_has = jnp.zeros((num_checkpoints,), dtype=jnp.bool_)

        runner_state = (
            train_state,
            initial_checkpoints,
            env_state,
            obsv,
            jnp.zeros((model_config["NUM_ACTORS"]), dtype=bool),
            initial_update_step,
            init_hstate,
            init_population_hstate,
            init_population_annealing_mask,
            init_fcp_pop_idxs,
            initial_obs_history,
            initial_act_history,
            initial_last_partner_action,
            initial_last_action,
            initial_ego_idxs,
            initial_blocked_states_env,
            jnp.zeros((model_config["NUM_ENVS"],), dtype=jnp.float32),
            jnp.zeros((model_config["NUM_ENVS"],), dtype=jnp.float32),
            jnp.zeros((model_config["NUM_ENVS"],), dtype=jnp.int32),
            jnp.zeros((model_config["NUM_ENVS"],), dtype=jnp.int32),
            initial_ph1_pool_states,
            initial_ph1_probs,
            initial_ph1_pool_ready,
            initial_ph1_recent_ckpts,
            initial_ph1_recent_has,
            _rng,
        )

        # Save an initial snapshot at step=0 so offline eval can include 0, N, 2N, ...
        if ph1_eval_enabled:
            def _ph1_snapshot0_cb(params, original_seed_host):
                from overcooked_v2_experiments.ppo.ph1_online_eval import save_ph1_video_snapshot

                seed_np = np.array(original_seed_host)
                if seed_np.ndim > 0:
                    seed_val = int(seed_np.flat[0])
                else:
                    seed_val = int(seed_np)
                dedup_key = (seed_val, 0)
                with _PH1_EVAL_LOCK:
                    if dedup_key in _PH1_EVAL_SEEN_KEYS:
                        return
                    _PH1_EVAL_SEEN_KEYS.add(dedup_key)

                try:
                    snap_dir = save_ph1_video_snapshot(
                        params=params,
                        config=config,
                        update_step=0,
                        recent_tilde_batch=None,
                        seed=seed_val,
                    )
                    if snap_dir is not None:
                        try:
                            wandb.log(
                                {
                                    f"rng{seed_val}/eval_ph1/deferred_video_snapshot_path": snap_dir,
                                    "eval/update_step": 0,
                                }
                            )
                        except Exception:
                            pass
                except Exception as e:
                    try:
                        wandb.log(
                            {
                                f"rng{seed_val}/eval_ph1/snapshot_error": str(e),
                                "eval/update_step": 0,
                            }
                        )
                    except Exception:
                        pass

            jax.lax.cond(
                jnp.array(initial_update_step == 0),
                lambda _: jax.debug.callback(
                    _ph1_snapshot0_cb,
                    train_state.params,
                    original_seed,
                ),
                lambda _: None,
                operand=None,
            )

        num_update_steps = model_config["NUM_UPDATES"]
        if update_step_num_overwrite is not None:
            num_update_steps = update_step_num_overwrite
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, num_update_steps
        )

        # jax.debug.print("Runner state {x}", x=runner_state)
        # jax.debug.print("neg5 {x}", x=runner_state[-5])
        return {"runner_state": runner_state, "metrics": metric}

    return train
