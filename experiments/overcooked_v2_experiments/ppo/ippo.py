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
from models.rnn import ScannedRNN
import matplotlib.pyplot as plt
from jaxmarl.environments.overcooked_v2.overcooked import ObservationType
from overcooked_v2_experiments.eval.policy import AbstractPolicy
from overcooked_v2_experiments.ppo.models.abstract import ActorCriticBase
from .models.model import get_actor_critic, initialize_carry
from overcooked_v2_experiments.eval.policy import AbstractPolicy
from flax import core


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


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_train(
    config,
    update_step_offset=None,
    update_step_num_overwrite=None,
    population_config=None,
):
    env_config = config["env"]
    model_config = config["model"]

    # E3T 설정 로드
    alg_name = config.get("ALG_NAME", "SP")
    e3t_epsilon = config.get("E3T_EPSILON", 0.05)
    use_partner_modeling = config.get("USE_PARTNER_MODELING", True)

    confidence_config = config.get("confidence_trigger", {})
    use_confidence_trigger = bool(confidence_config.get("enabled", False))
    print(f"trigger : {use_confidence_trigger}")

    # Optional device selection for environment to mitigate GPU OOM.
    # Usage via Hydra override: +ENV_DEVICE=cpu  (default: gpu / auto)
    env_device = config.get("ENV_DEVICE", None)  # None -> default placement
    if env_device is not None and env_device not in ("cpu", "gpu"):
        raise ValueError(f"ENV_DEVICE must be one of 'cpu','gpu' (or unset). Got: {env_device}")

    # Create env under a device context so large static buffers (layout grids etc.) live on CPU when requested.
    if env_device == "cpu":
        with jax.default_device(jax.devices("cpu")[0]):
            env = jaxmarl.make(env_config["ENV_NAME"], **env_config["ENV_KWARGS"])
    else:
        env = jaxmarl.make(env_config["ENV_NAME"], **env_config["ENV_KWARGS"])

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

    num_checkpoints = config["NUM_CHECKPOINTS"]
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

    env = OvercookedV2LogWrapper(env, replace_info=False)

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

    # ------------------------------------------------------------------
    # 확신(trigger) 옵션 파싱
    #   - mode: under(불확실시 발동) / over(과도한 확신 시 발동)
    #   - entropy_threshold: 엔트로피 기준값
    #   - cooldown_steps: 트리거 이후 강제 행동 지속 스텝 수
    #   - target: 어떤 에이전트에 적용할지 (ego/partner/both)
    # ------------------------------------------------------------------
    confidence_mode = str(confidence_config.get("mode", "under")).lower()
    confidence_threshold = float(confidence_config.get("entropy_threshold", 0.8))
    confidence_cooldown_steps = int(confidence_config.get("cooldown_steps", 5))
    confidence_target = str(confidence_config.get("target", "partner")).lower()
    n_threshold = int(confidence_config.get("n_threshold", 1.0))
    cooldown_value = jnp.array(confidence_cooldown_steps, dtype=jnp.int32)

    # NUM_ACTORS = (env.num_agents * NUM_ENVS)이므로, 각 슬롯이 어떤 에이전트인지 구분하기 위해
    # 반복되는 인덱스 벡터를 만들어 ego/partner 마스크를 구성한다.
    # batchify 결과는 Agent Major 순서 ([Ag0_Env0, ..., Ag0_EnvN, Ag1_Env0, ...])이므로
    # repeat를 사용하여 [0, 0, ..., 1, 1, ...] 형태로 만들어야 한다.
    actor_indices = jnp.repeat(
        jnp.arange(env.num_agents, dtype=jnp.int32), model_config["NUM_ENVS"]
    )
    ego_actor_mask = actor_indices == 0
    partner_actor_mask = actor_indices == (env.num_agents - 1)
    confidence_target_mask = jnp.ones_like(actor_indices, dtype=bool)
    if confidence_target == "ego":
        confidence_target_mask = ego_actor_mask
    elif confidence_target == "partner" and env.num_agents > 1:
        confidence_target_mask = partner_actor_mask

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

        # INIT NETWORK
        network = get_actor_critic(config)

        rng, _rng = jax.random.split(rng)

        init_x = (
            jnp.zeros(
                (1, model_config["NUM_ENVS"], *env.observation_space().shape),
            ),
            jnp.zeros((1, model_config["NUM_ENVS"])),
        )
        init_hstate = initialize_carry(config, model_config["NUM_ENVS"])

        if init_hstate is not None:
            print("init_hstate", init_hstate.shape)
        # jax.debug.print("check1 {x}", x=init_hstate.flatten()[0])

        print("init_x", init_x[0].shape, init_x[1].shape)

        # E3T: Prepare dummy inputs for Single Initialization
        dummy_partner_prediction = None
        dummy_obs_hist = None
        dummy_act_hist = None
        
        if alg_name == "E3T" and use_partner_modeling:
             # Shape: (Time=1, Batch=NUM_ENVS, ActionDim=6)
             dummy_partner_prediction = jnp.zeros((1, model_config["NUM_ENVS"], ACTION_DIM))
             dummy_obs_hist = jnp.zeros((1, 5, *env.observation_space().shape))
             dummy_act_hist = jnp.zeros((1, 5), dtype=jnp.int32)

        # Single Init: Initialize all parameters at once to avoid collision
        # 단일 초기화: 파라미터 충돌 방지를 위해 모든 모듈을 한 번에 초기화
        network_params = network.init(
            _rng, 
            init_hstate, 
            init_x, 
            partner_prediction=dummy_partner_prediction,
            obs_history=dummy_obs_hist,
            act_history=dummy_act_hist
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

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, model_config["NUM_ENVS"])
        obsv, env_state = jax.vmap(reset_fn)(reset_rng)
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
                confidence_counters,
                confidence_episode_counts,
                obs_history,
                act_history,
                last_partner_action,
                rng,
            ) = runner_state

            # jax.debug.print("check3 {x}", x=initial_hstate.flatten()[0])

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
                    confidence_counters,
                    confidence_episode_counts,
                    obs_history,
                    act_history,
                    last_partner_action,
                    completed_trigger_sum,
                    completed_episode_count,
                    rng,
                ) = env_step_state

                # Calculate actor indices and is_ego
                actor_indices = jnp.repeat(
                    jnp.arange(env.num_agents, dtype=jnp.int32), model_config["NUM_ENVS"]
                )
                is_ego = actor_indices == 0

                # jax.debug.print("check4 {x}", x=hstate.flatten()[0])

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)

                obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape(
                    -1, *env.observation_space().shape
                )
                if cast_obs_bf16:
                    obs_batch = obs_batch.astype(jnp.bfloat16)

                # --------------------------------------------------------------
                # E3T History Update & Partner Prediction
                # --------------------------------------------------------------
                # 1. History Update
                if alg_name == "E3T":
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
                if alg_name == "E3T" and use_partner_modeling:
                    # (1) 실제 파트너 예측 (Ego용)
                    real_prediction = network.apply(train_state.params, obs_history, act_history, method='predict_partner')
                    
                    # (2) 무작위 행동 임베딩 (Partner용)
                    # 파트너는 Ego를 예측하지 않고, 무작위 행동(또는 노이즈)을 조건으로 받아 자신의 정책을 수행한다고 가정
                    rng, rng_rand_input = jax.random.split(rng)
                    rand_act_input = jax.random.randint(rng_rand_input, (model_config["NUM_ACTORS"],), 0, ACTION_DIM)
                    rand_pred_input = jax.nn.one_hot(rand_act_input, num_classes=ACTION_DIM)
                    
                    # L2 Normalize (PartnerPredictionModule 출력 스케일과 맞춤)
                    norm = jnp.linalg.norm(rand_pred_input, axis=-1, keepdims=True)
                    rand_pred_input = rand_pred_input / (norm + 1e-6)
                    
                    # (3) Ego vs Partner 구분하여 입력 선택
                    # Ego(Agent 0)는 예측값 사용, Partner(Agent 1)는 무작위 입력 사용
                    is_ego = (actor_indices == 0)
                    
                    # real_prediction: (Batch, 6)
                    # rand_pred_input: (Batch, 6)
                    combined_prediction = jnp.where(is_ego[:, None], real_prediction, rand_pred_input)
                    
                    # (Batch, ActionDim) -> (1, Batch, ActionDim) for Actor input
                    partner_prediction = combined_prediction[jnp.newaxis, ...]

                ac_in = (
                    obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                )

                hstate, pi, value = network.apply(train_state.params, hstate, ac_in, partner_prediction=partner_prediction)

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
                confidence_trigger_mask = jnp.zeros(
                    (model_config["NUM_ACTORS"],), dtype=jnp.bool_
                )
                random_replace_mask = jnp.zeros_like(confidence_trigger_mask)

                if use_confidence_trigger:
                    # 2) 모드에 따라 임계값 비교 (under: 높을수록 불확신, over: 낮을수록 과신)
                    if confidence_mode == "under":
                        trigger_mask = policy_entropy > confidence_threshold
                    else:
                        trigger_mask = policy_entropy < confidence_threshold

                    # 3) 대상 에이전트 + 현재 카운터가 0인 슬롯 + 에피소드별 트리거 횟수 제한에만 트리거 허용
                    confidence_episode_counts_per_actor = jnp.tile(confidence_episode_counts, env.num_agents)
                    trigger_mask = (
                        trigger_mask
                        & confidence_target_mask
                        & (confidence_counters == 0)
                        & (confidence_episode_counts_per_actor < n_threshold)
                    )

                    # 4) 트리거된 슬롯은 cooldown_steps 값으로 카운터를 세팅
                    confidence_counters = jnp.where(
                        trigger_mask,
                        jnp.full_like(confidence_counters, cooldown_value),
                        confidence_counters,
                    )
                    confidence_trigger_mask = trigger_mask

                    # 레이아웃마다 NUM_ENVS * num_agents 순서로 펼쳐져 있으므로, env 단위로 reshape하여
                    # 이번 스텝에서 새롭게 발동한 trigger 횟수를 episode 누적 버퍼에 추가한다.
                    # trigger_mask: (num_agents * num_envs,) -> reshape to (num_agents, num_envs) -> sum over agents
                    trigger_events_by_env = trigger_mask.reshape(
                        (env.num_agents, model_config["NUM_ENVS"])
                    ).sum(axis=0)
                    confidence_episode_counts = confidence_episode_counts + trigger_events_by_env.astype(
                        jnp.float32
                    )

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
                            obs_population, last_done, population_hstate, _rng
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

                if use_confidence_trigger:
                    # --------------------------------------------------------------
                    # 랜덤 파트너 액션 주입
                    #   - random_replace_mask: 냉각(cooldown)이 남아 강제 랜덤 행동이 필요한 슬롯
                    #   - mask(True)면 정책 액션 유지, False면 완전 균등샘플(random_actions)로 대체
                    # --------------------------------------------------------------
                    random_replace_mask = confidence_target_mask & (confidence_counters > 0)
                    keep_policy_action_mask = ~random_replace_mask

                    rng, _rng_random = jax.random.split(rng)
                    random_actions = jax.random.randint(
                        _rng_random,
                        action_for_env.shape,
                        0,
                        num_action_choices,
                        dtype=action_for_env.dtype,
                    )

                    # 기존 panic 모드처럼 mask(True) → 정책 샘플 유지, False → 무작위 액션으로 치환
                    action_for_env = jnp.where(
                        keep_policy_action_mask,
                        action_for_env,
                        random_actions,
                    )

                # E3T 알고리즘: 파트너 정책 혼합 (Mixture Partner Policy)
                # [E3T] 파트너 행동 변경 로직 (Mixture Policy)
                # 파트너는 (1-epsilon) 확률로 자신의 정책을 따르고, epsilon 확률로 무작위 행동을 수행함
                if alg_name == "E3T":
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

                # 메트릭 로깅용으로 각 보상 타입을 info에 저장
                shaped_reward = jnp.array(
                    [info["shaped_reward"][a] for a in env.agents]
                )
                combined_reward = jnp.array([reward[a] for a in env.agents])

                info["shaped_reward"] = shaped_reward      # 환경의 중간 단계 보상 (쉐이핑)
                info["original_reward"] = original_reward  # 환경의 원본 sparse 보상
                info["anneal_factor"] = jnp.full_like(shaped_reward, anneal_factor)  # 현재 감쇠 계수
                info["combined_reward"] = combined_reward  # PPO 학습에 실제 사용되는 최종 보상

                # --------------------------------------------------------------
                # confidence / entropy 디버깅 정보를 info에 추가
                #   - policy_entropy: 각 슬롯별 정책 엔트로피 (항상 기록)
                #   - confidence_trigger: 이번 스텝에서 새롭게 발동한 슬롯 (0/1)
                #   - confidence_cooldown: 남은 쿨다운 스텝
                #   - random_step_active: 랜덤 액션으로 대체된 슬롯 (0/1)
                # --------------------------------------------------------------
                info["policy_entropy"] = policy_entropy
                if use_confidence_trigger:
                    info["confidence_trigger"] = confidence_trigger_mask.astype(jnp.int32)
                    info["confidence_cooldown"] = confidence_counters
                    info["random_step_active"] = random_replace_mask.astype(jnp.int32)

                info = jax.tree_util.tree_map(
                    lambda x: x.reshape((model_config["NUM_ACTORS"])), info
                )

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
                )

                if use_confidence_trigger:
                    # 스텝이 끝날 때마다 1씩 감소 (음수 방지)
                    confidence_counters = jnp.maximum(
                        confidence_counters - 1, 0
                    )

                    # 에피소드가 종료된 슬롯은 카운터 완전 초기화
                    episode_reset_mask = jnp.tile(
                        done["__all__"], env.num_agents
                    )
                    confidence_counters = jnp.where(
                        episode_reset_mask,
                        0,
                        confidence_counters,
                    )

                    # env 단위 episode 종료 시점에 이번 episode에서 발생한 trigger 횟수를 로깅 버퍼에 누적
                    episode_done_env = done["__all__"].astype(jnp.float32)
                    completed_trigger_sum = completed_trigger_sum + jnp.sum(
                        confidence_episode_counts * episode_done_env
                    )
                    completed_episode_count = completed_episode_count + jnp.sum(
                        episode_done_env
                    )
                    confidence_episode_counts = jnp.where(
                        episode_done_env,
                        0.0,
                        confidence_episode_counts,
                    )

                # jax.debug.print("check6 {x}", x=hstate.flatten()[0])

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
                    confidence_counters,
                    confidence_episode_counts,
                    obs_history,
                    act_history,
                    last_partner_action,
                    completed_trigger_sum,
                    completed_episode_count,
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
                confidence_counters,
                confidence_episode_counts,
                obs_history,
                act_history,
                last_partner_action,
                jnp.zeros((), dtype=jnp.float32),  # completed trigger sum (per update)
                jnp.zeros((), dtype=jnp.float32),  # completed episode count (per update)
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
                confidence_counters,
                confidence_episode_counts,
                obs_history,
                act_history,
                last_partner_action,
                completed_trigger_sum,
                completed_episode_count,
                rng,
            ) = env_step_state

            # jax.debug.print("check7 {x}", x=next_initial_hstate)

            # print("Hilfeeeee", traj_batch.done.shape, traj_batch.action.shape)

            # CALCULATE ADVANTAGE
            last_obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape(
                -1, *env.observation_space().shape
            )
            if cast_obs_bf16:
                last_obs_batch = last_obs_batch.astype(jnp.bfloat16)
            ac_in = (
                last_obs_batch[np.newaxis, :],
                last_done[np.newaxis, :],
            )

            # [E3T] Calculate partner_prediction for last_val (Value Function)
            partner_prediction = None
            if alg_name == "E3T" and use_partner_modeling:
                # 1. Predict
                pred_logits = network.apply(train_state.params, obs_history, act_history, method='predict_partner')
                
                # 2. Random Input for Partner
                rng, rng_rand_input = jax.random.split(rng)
                rand_act_input = jax.random.randint(rng_rand_input, (model_config["NUM_ACTORS"],), 0, ACTION_DIM)
                rand_pred_input = jax.nn.one_hot(rand_act_input, num_classes=ACTION_DIM)
                norm = jnp.linalg.norm(rand_pred_input, axis=-1, keepdims=True)
                rand_pred_input = rand_pred_input / (norm + 1e-6)
                
                # 3. Combine
                is_ego = (actor_indices == 0)
                combined_prediction = jnp.where(is_ego[:, None], pred_logits, rand_pred_input)
                
                # 4. Add Time Dimension (1, Batch, ActionDim)
                partner_prediction = combined_prediction[jnp.newaxis, ...]

            _, _, last_val = network.apply(
                train_state.params, next_initial_hstate, ac_in, partner_prediction=partner_prediction
            )

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
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        hstate = init_hstate
                        if hstate is not None:
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

                        if alg_name == "E3T" and use_partner_modeling:
                            # 1. Forward Pass for Prediction
                            # traj_batch.obs_history: (Time, Batch, 5, H, W, C)
                            # traj_batch.act_history: (Time, Batch, 5)
                            
                            # Flatten Time and Batch dimensions for network input
                            T, B = traj_batch.obs_history.shape[:2]
                            obs_hist_flat = traj_batch.obs_history.reshape(T * B, *traj_batch.obs_history.shape[2:])
                            act_hist_flat = traj_batch.act_history.reshape(T * B, *traj_batch.act_history.shape[2:])
                            
                            # Predict Partner Action (파트너 행동 예측)
                            # obs_history와 act_history를 사용하여 파트너의 다음 행동을 예측합니다.
                            pred_logits = network.apply(params, obs_hist_flat, act_hist_flat, method='predict_partner')
                            
                            # 2. Label Matching
                            target_labels = traj_batch.partner_action.reshape(T * B).astype(jnp.int32)
                            
                            # 3. Compute Cross-Entropy Loss (예측 손실 계산)
                            pred_loss_vec = optax.softmax_cross_entropy_with_integer_labels(logits=pred_logits, labels=target_labels)
                            
                            # [E3T] Masking: Only Ego agents contribute to prediction loss
                            # Ego 에이전트만 예측 손실에 기여하도록 마스킹 (Partner는 랜덤 행동을 하므로 예측 불가/불필요)
                            is_ego_flat = traj_batch.is_ego.reshape(T * B)
                            pred_loss = (pred_loss_vec * is_ego_flat).sum() / (is_ego_flat.sum() + 1e-8)
                            
                            # Apply Coefficient
                            pred_loss = pred_loss * config.get("PRED_LOSS_COEF", 1.0)
                            
                            # Calculate Accuracy
                            pred_labels = jnp.argmax(pred_logits, axis=-1)
                            pred_accuracy = jnp.mean(pred_labels == target_labels)
                            
                            # 4. Stop Gradient for Ego Policy (Actor 입력용)
                            # Actor-Critic 네트워크에 입력으로 주기 위해 gradient를 차단합니다.
                            # 이는 PPO 손실이 예측 네트워크를 업데이트하지 않도록 하기 위함입니다.
                            partner_prediction_flat = jax.lax.stop_gradient(pred_logits)
                            partner_prediction = partner_prediction_flat.reshape(T, B, -1)

                        # RERUN NETWORK
                        # Actor-Critic 네트워크 재실행 (Value, Policy 계산)
                        # partner_prediction을 함께 전달하여 shape mismatch (134 vs 128) 해결
                        _, pi, value = network.apply(
                            params,
                            hstate,
                            (traj_batch.obs, traj_batch.done),
                            partner_prediction=partner_prediction
                        )

                        print("value shape", value.shape)
                        print("targets shape", targets.shape)
                        print("pi shape", pi.logits.shape)

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

                        return total_loss, (value_loss, loss_actor, entropy, ratio, pred_loss, pred_accuracy)

                    def _perform_update():
                        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                        total_loss, grads = grad_fn(
                            train_state.params,
                            init_hstate,
                            traj_batch,
                            advantages,
                            targets,
                        )

                        # jax.debug.print(
                        #     "grads {x}, hstate {y}, mask {z}",
                        #     x=jax.tree_util.tree_flatten(grads)[0][0][0],
                        #     y=init_hstate.flatten()[0],
                        #     z=traj_batch.train_mask.sum(),
                        # )

                        new_train_state = train_state.apply_gradients(grads=grads)
                        return new_train_state, total_loss

                    def _no_op():
                        # jax.debug.print("No update")
                        return train_state, (0.0, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

                    # jax.debug.print(
                    #     "train_mask {x}, {y}",
                    #     x=traj_batch.train_mask.sum(),
                    #     y=traj_batch.train_mask.any(),
                    # )

                    train_state, total_loss = jax.lax.cond(
                        traj_batch.train_mask.any(),
                        _perform_update,
                        _no_op,
                    )
                    return train_state, total_loss

                train_state, init_hstate, traj_batch, advantages, targets, rng = (
                    update_state
                )
                rng, _rng = jax.random.split(rng)

                num_actors = model_config["NUM_ACTORS"]

                hstate = init_hstate
                if hstate is not None:
                    print("hstate shape", hstate.shape)
                    hstate = hstate[jnp.newaxis, :]
                    print("hstate shape", hstate.shape)

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

                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
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

            # --------------------------------------------------------------
            # confidence/entropy 통계 파생 메트릭
            #   - policy_entropy_mean: rollout 전체 평균 엔트로피
            #   - confidence_trigger_per_episode: (이번 업데이트 내 완료된 episode의 트리거 평균 횟수)
            #   - confidence_trigger_events_sum / confidence_completed_episodes: 분자/분모 디버깅용
            # --------------------------------------------------------------
            if "policy_entropy" in metric:
                metric["policy_entropy_mean"] = metric["policy_entropy"].mean()

            if use_confidence_trigger:
                trigger_per_episode = jnp.where(
                    completed_episode_count > 0,
                    completed_trigger_sum / completed_episode_count,
                    0.0,
                )
                metric["confidence_trigger_per_episode"] = trigger_per_episode
                metric["confidence_trigger_events_sum"] = completed_trigger_sum
                metric["confidence_completed_episodes"] = completed_episode_count

            # 손실 함수 관련 메트릭 추출
            total_loss, aux_data = loss_info
            value_loss, loss_actor, entropy, ratio, pred_loss, pred_accuracy = aux_data

            # PPO 학습 손실 메트릭 추가
            metric["total_loss"] = total_loss      # 전체 손실 (actor + value + entropy + pred)
            metric["value_loss"] = value_loss      # 가치 함수(critic) MSE 손실
            metric["loss_actor"] = loss_actor      # 정책(actor) clipped surrogate 손실
            metric["entropy"] = entropy            # 정책 엔트로피 (탐험 정도)
            metric["ratio"] = ratio                # PPO ratio (new_prob / old_prob)
            metric["pred_loss"] = pred_loss        # 파트너 행동 예측 손실
            metric["pred_accuracy"] = pred_accuracy # 파트너 행동 예측 정확도

            # 모든 메트릭 값을 배치/스텝 차원에 대해 평균 계산 (스칼라로 축약)
            metric = jax.tree_util.tree_map(lambda x: x.mean(), metric)

            # confidence 관련 핵심 메트릭은 항상 존재하도록 별도 스칼라 추가
            if use_confidence_trigger:
                metric["confidence_trigger_rate"] = metric.get(
                    "confidence_trigger", jnp.array(0.0, dtype=jnp.float32)
                )
                metric["confidence_cooldown_mean"] = metric.get(
                    "confidence_cooldown", jnp.array(0.0, dtype=jnp.float32)
                )
                metric["random_step_active_rate"] = metric.get(
                    "random_step_active", jnp.array(0.0, dtype=jnp.float32)
                )
            else:
                zero = jnp.array(0.0, dtype=jnp.float32)
                metric["confidence_trigger_rate"] = zero
                metric["confidence_cooldown_mean"] = zero
                metric["random_step_active_rate"] = zero

            # 업데이트 스텝 카운터 증가 및 메트릭에 기록
            update_step += 1
            metric["update_step"] = update_step
            # 환경과 상호작용한 총 스텝 수 계산 (update * rollout_length * num_envs)
            metric["env_step"] = (
                update_step * model_config["NUM_STEPS"] * model_config["NUM_ENVS"]                                                                 
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
            jax.debug.callback(callback, metric, original_seed)

            if num_checkpoints > 0:
                checkpoint_idx_selector = checkpoint_steps == update_step
                checkpoint_states = jax.lax.cond(
                    jnp.any(checkpoint_idx_selector),
                    _update_checkpoint,
                    lambda c, _p, _i: c,
                    checkpoint_states,
                    train_state.params,
                    jnp.argmax(checkpoint_idx_selector),
                )

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
                confidence_counters,
                confidence_episode_counts,
                obs_history,
                act_history,
                last_partner_action,
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

        # 각 액터 슬롯별로 트리거 잔여 스텝을 추적할 버퍼 초기화
        confidence_counter_shape = (model_config["NUM_ACTORS"],)
        initial_confidence_counters = jnp.zeros(
            confidence_counter_shape, dtype=jnp.int32
        )
        initial_confidence_episode_counts = jnp.zeros(
            (model_config["NUM_ENVS"],), dtype=jnp.float32
        )

        # E3T History Buffers
        initial_obs_history = jnp.zeros((model_config["NUM_ACTORS"], 5, *env.observation_space().shape))
        initial_act_history = jnp.zeros((model_config["NUM_ACTORS"], 5), dtype=jnp.int32)
        initial_last_partner_action = jnp.zeros((model_config["NUM_ACTORS"],), dtype=jnp.int32)

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
            initial_confidence_counters,
            initial_confidence_episode_counts,
            initial_obs_history,
            initial_act_history,
            initial_last_partner_action,
            _rng,
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
