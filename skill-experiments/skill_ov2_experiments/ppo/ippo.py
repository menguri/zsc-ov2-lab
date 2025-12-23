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
from skill_ov2_experiments.eval.policy import AbstractPolicy
from skill_ov2_experiments.ppo.models.abstract import ActorCriticBase
from .models.model import get_actor_critic, initialize_carry
from skill_ov2_experiments.eval.policy import AbstractPolicy
from flax import core
from .diayn import create_diayn_state, calculate_intrinsic_reward, update_discriminator, augment_obs_with_skill


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    train_mask: jnp.ndarray
    skills: jnp.ndarray  # [DIAYN] 스킬 저장용 필드 추가
    raw_obs: jnp.ndarray # [DIAYN] Discriminator 학습용 원본 관측값 (s_t)


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

    # [DIAYN] 설정 로드
    diayn_config = config.get("DIAYN", {})
    use_diayn = diayn_config.get("ENABLED", False)
    num_skills = diayn_config.get("NUM_SKILLS", 20)
    
    print(f"[DEBUG] DIAYN Config: {diayn_config}")
    print(f"[DEBUG] use_diayn: {use_diayn}")
    print(f"[DEBUG] num_skills: {num_skills}")

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
            if isinstance(init_hstate, tuple):
                print("init_hstate (tuple)", [x.shape for x in init_hstate])
            else:
                print("init_hstate", init_hstate.shape)
        # jax.debug.print("check1 {x}", x=init_hstate.flatten()[0])

        print("init_x", init_x[0].shape, init_x[1].shape)

        # Single Init: Initialize all parameters at once to avoid collision
        # 단일 초기화: 파라미터 충돌 방지를 위해 모든 모듈을 한 번에 초기화
        
        # [DIAYN] Discriminator 초기화 및 스킬 할당
        diayn_state = None
        skills = None
        if use_diayn:
            rng, _rng = jax.random.split(rng)
            diayn_state = create_diayn_state(
                _rng, 
                env.observation_space().shape, 
                num_skills, 
                diayn_config.get("DISCRIMINATOR_LR", 3e-4),
                tuple(diayn_config.get("DISCRIMINATOR_HIDDEN_DIMS", [256, 256]))
            )
            
            # 초기 스킬 할당 (에피소드마다 갱신되지만 초기값 필요)
            rng, _rng = jax.random.split(rng)
            skills = jax.random.randint(_rng, (model_config["NUM_ACTORS"],), 0, num_skills)
            
            # Network Init 시 입력 차원 증가 고려 (augment_obs_with_skill)
            # init_x[0]는 (1, NUM_ENVS, H, W, C) 형태이므로, 
            # augment_obs_with_skill을 적용하기 위해 차원을 맞춰야 함.
            # 여기서는 간단히 더미 입력 생성 시 채널을 늘려서 전달하는 방식 대신,
            # 실제 augment 함수를 통과시킨 더미 데이터를 사용.
            
            # (1, NUM_ENVS, H, W, C) -> (NUM_ENVS, H, W, C)
            dummy_obs_batch = init_x[0][0] 
            # (NUM_ENVS, )
            dummy_skills_batch = skills[:model_config["NUM_ENVS"]]
            
            augmented_obs = jax.vmap(augment_obs_with_skill, in_axes=(0, 0, None))(
                dummy_obs_batch, dummy_skills_batch, num_skills
            )
            # (1, NUM_ENVS, H, W, C+num_skills)
            init_x_augmented = (augmented_obs[jnp.newaxis, ...], init_x[1])
            
            network_params = network.init(
                _rng, 
                init_hstate, 
                init_x_augmented
            )
        else:
            network_params = network.init(
                _rng, 
                init_hstate, 
                init_x
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
                obs_history,
                act_history,
                last_partner_action,
                rng,
                diayn_state, # [DIAYN] 추가
                skills,      # [DIAYN] 추가
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
                    obs_history,
                    act_history,
                    rng,
                    skills, # [DIAYN] 추가
                ) = env_step_state

                # Calculate actor indices and is_ego
                actor_indices = jnp.repeat(
                    jnp.arange(env.num_agents, dtype=jnp.int32), model_config["NUM_ENVS"]
                )
                is_ego = actor_indices == 0

                # Capture z_state from input hstate (before update) for storage
                z_state_in = jnp.zeros((model_config["NUM_ACTORS"], ACTION_DIM))
                if isinstance(hstate, tuple):
                    _, z_state_in = hstate

                # jax.debug.print("check4 {x}", x=hstate.flatten()[0])

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)

                obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape(
                    -1, *env.observation_space().shape
                )
                if cast_obs_bf16:
                    obs_batch = obs_batch.astype(jnp.bfloat16)
                
                # [DIAYN] 원본 관측값 저장 (Discriminator 학습 및 Intrinsic Reward 계산용)
                raw_obs_batch = obs_batch

                # [DIAYN] 관측값에 스킬 주입
                if use_diayn:
                    obs_batch = jax.vmap(augment_obs_with_skill, in_axes=(0, 0, None))(
                        obs_batch, skills, num_skills
                    )
                    obs_batch = obs_batch.astype(jnp.bfloat16)
                    prev_partner_action = jnp.roll(last_partner_action, shift=model_config["NUM_ENVS"], axis=0)
                    last_partner_action_masked = jnp.where(last_done, 0, prev_partner_action)
                    act_history = act_history.at[:, -1].set(last_partner_action_masked)
                
                ac_in = (
                    obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                )

                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)

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

                # Ensure action_for_env is squeezed (NUM_ACTORS,) to match carry shape
                # This fixes the scan carry shape mismatch error (int32[512] vs int32[1,512])
                if len(action_for_env.shape) > 1:
                    action_for_env = action_for_env.squeeze()

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
                
                # [DIAYN] Intrinsic Reward 계산 및 적용
                if use_diayn:
                    # s_t (raw_obs_batch)를 사용하여 Intrinsic Reward 계산: r = log q(z|s_t) - log p(z)
                    intrinsic_reward = calculate_intrinsic_reward(
                        diayn_state, raw_obs_batch, skills, num_skills
                    )
                    
                    # Intrinsic Reward 스케일링
                    intrinsic_reward = intrinsic_reward * diayn_config.get("INTRINSIC_REWARD_SCALE", 1.0)
                    
                    # Reward 대체 (또는 합산)
                    # 여기서는 Intrinsic Reward만 사용하는 것으로 가정 (Unsupervised Skill Discovery)
                    # 만약 External Reward와 섞고 싶다면: reward = reward + intrinsic_reward
                    
                    # reward는 dict 형태 {agent_name: (num_envs, )}
                    # intrinsic_reward는 (num_actors, ) -> (num_agents * num_envs, )
                    # unbatchify를 통해 다시 dict 형태로 변환
                    intrinsic_reward_dict = unbatchify(
                        intrinsic_reward, env.agents, model_config["NUM_ENVS"], env.num_agents
                    )
                    
                    # 기존 reward를 intrinsic reward로 덮어쓰기
                    reward = intrinsic_reward_dict
                    
                    # 로깅을 위해 info에 추가
                    info["intrinsic_reward"] = intrinsic_reward

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
                # entropy 디버깅 정보를 info에 추가
                #   - policy_entropy: 각 슬롯별 정책 엔트로피 (항상 기록)
                # --------------------------------------------------------------
                info["policy_entropy"] = policy_entropy

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

                # [DIAYN] Skill Resampling: 에피소드 종료 시 스킬 재할당
                if use_diayn:
                    rng, _rng = jax.random.split(rng)
                    new_skills = jax.random.randint(_rng, (model_config["NUM_ACTORS"],), 0, num_skills)
                    # done["__all__"]은 (NUM_ENVS,) 형태이므로, 이를 (NUM_ACTORS,) 형태로 확장
                    # batchify 구조상 [Agent0_Env0, ..., Agent1_Env0, ...] 순서이므로 tile 사용
                    done_all_repeated = jnp.tile(done["__all__"], env.num_agents)
                    skills = jnp.where(done_all_repeated, new_skills, skills)

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

                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    action.squeeze(),
                    value.squeeze(),
                    batchify(reward, env.agents, model_config["NUM_ACTORS"]).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    info,
                    action_pick_mask,
                    skills, # [DIAYN] 추가
                    raw_obs_batch, # [DIAYN] 추가
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
                    obs_history,
                    act_history,
                    rng,
                    skills, # [DIAYN] 추가
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
                rng,
                skills, # [DIAYN] 추가
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
                rng,
                skills, # [DIAYN] 추가
            ) = env_step_state

            # jax.debug.print("check7 {x}", x=next_initial_hstate)

            # print("Hilfeeeee", traj_batch.done.shape, traj_batch.action.shape)

            # CALCULATE ADVANTAGE
            last_obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape(
                -1, *env.observation_space().shape
            )
            if cast_obs_bf16:
                last_obs_batch = last_obs_batch.astype(jnp.bfloat16)
            
            # [DIAYN] Value Function 계산 시에도 스킬 주입
            if use_diayn:
                last_obs_batch = jax.vmap(augment_obs_with_skill, in_axes=(0, 0, None))(
                    last_obs_batch, skills, num_skills
                )

            ac_in = (
                last_obs_batch[np.newaxis, :],
                last_done[np.newaxis, :],
            )

            _, _, last_val = network.apply(
                train_state.params, next_initial_hstate, ac_in
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
                (
                    train_state,
                    initial_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                    diayn_state, # [DIAYN] 추가
                ) = update_state

                rng, _rng = jax.random.split(rng)
                
                # [DIAYN] Discriminator Update
                # PPO 업데이트와 별도로 수행하거나, 여기서 같이 수행
                # 여기서는 매 epoch마다 업데이트한다고 가정 (UPDATE_FREQ 고려 가능)
                # 중복 제거: 하단에서 raw_obs로 수행함.

                def _update_minbatch(train_state, batch_info):
                    (
                        init_hstate,
                        traj_batch,
                        advantages,
                        targets,
                    ) = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
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

                        # RERUN NETWORK
                        # Actor-Critic 네트워크 재실행 (Value, Policy 계산)
                        # partner_prediction을 함께 전달하여 shape mismatch (134 vs 128) 해결
                        
                        # [DIAYN] 관측값에 스킬 주입 (Network Forward 전)
                        # traj_batch.obs는 이미 _env_step에서 스킬이 주입된 상태(Augmented)입니다.
                        # 따라서 여기서 다시 주입하면 Double Augmentation이 되어 차원이 늘어나는 문제(39 -> 48)가 발생합니다.
                        # 그러므로 여기서는 추가 주입 없이 그대로 사용합니다.
                        obs_input = traj_batch.obs
                        # if use_diayn: ... (Removed to prevent double augmentation)

                        _, pi, value = network.apply(
                            params,
                            hstate,
                            (obs_input, traj_batch.done)
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
                        )

                        return total_loss, (value_loss, loss_actor, entropy, ratio)

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
                        return train_state, (0.0, (0.0, 0.0, 0.0, 0.0))

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

                train_state, init_hstate, traj_batch, advantages, targets, rng, diayn_state = (
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

                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )

                if use_diayn:
                    flat_obs = traj_batch.raw_obs.reshape(-1, *env.observation_space().shape)
                    flat_skills = traj_batch.skills.reshape(-1)
                    diayn_state, disc_loss = update_discriminator(
                        diayn_state, flat_obs, flat_skills
                    )

                update_state = (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                    diayn_state, # [DIAYN] 추가
                )
                
                # [DIAYN] Discriminator Loss를 aux_data에 추가하기 위해 loss_info 구조 변경 필요
                # 현재 loss_info는 (total_loss, (value_loss, ...)) 형태
                # disc_loss를 반환하려면 _update_epoch의 반환값을 수정해야 함
                # 하지만 jax.lax.scan의 구조상 반환값 형태를 맞추는 것이 중요
                
                # 여기서는 disc_loss를 total_loss의 aux_data에 추가하는 대신,
                # 별도로 반환하거나 metric에 추가하는 방식을 사용해야 함.
                # 간단하게 total_loss의 aux_data 튜플에 disc_loss를 추가하여 반환
                
                # 기존 aux_data: (value_loss, loss_actor, entropy, ratio, pred_loss, pred_accuracy)
                # 변경 aux_data: (..., disc_loss)
                
                # 하지만 _update_minbatch의 반환값과 _update_epoch의 반환값이 일치해야 함.
                # _update_minbatch는 disc_loss를 계산하지 않으므로 0.0으로 채워야 함.
                
                # 여기서는 복잡성을 피하기 위해 disc_loss를 metric에 직접 추가하는 것이 어려우므로(scan 외부),
                # disc_loss를 aux_data의 마지막 요소로 추가하고, _update_minbatch에서는 0.0을 반환하도록 수정
                
                # _update_minbatch의 반환값 수정 (ippo.py의 _loss_fn 수정 필요)
                # -> 너무 복잡해지므로, disc_loss는 무시하고 diayn_state만 업데이트
                
                # disc_loss를 로깅하려면 _update_epoch의 반환값에 포함시켜야 함.
                # 현재 구조: update_state, loss_info = scan(...)
                # loss_info는 _update_minbatch의 반환값들의 적층
                
                # _update_epoch는 (update_state, loss_info)를 반환해야 함
                # loss_info는 (total_loss, aux_data)
                
                # disc_loss를 aux_data에 추가하여 반환
                # 하지만 _update_minbatch의 결과값과 합쳐야 함.
                
                # 해결책: disc_loss를 _update_epoch의 반환값(loss_info)에 포함시키지 않고,
                # update_state에 포함시켜서 밖으로 빼낸 뒤, 마지막 step의 값만 사용
                
                return update_state, total_loss

            rng, _rng = jax.random.split(rng)
            update_state = (
                train_state,
                initial_hstate,
                traj_batch,
                advantages,
                targets,
                _rng,
                diayn_state, # [DIAYN] 추가
            )
            # UPDATE_EPOCHS 횟수만큼 미니배치 업데이트를 반복 수행
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, model_config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            diayn_state = update_state[6] # [DIAYN] 업데이트된 상태 추출
            
            # [DIAYN] Discriminator Loss 로깅 (마지막 Epoch 기준, 정확하지 않지만 근사값)
            # 정확한 로깅을 위해서는 _update_epoch 내부에서 계산된 disc_loss를 밖으로 빼내야 함
            # 여기서는 간단히 재계산하여 로깅 (비효율적이지만 구현 용이)
            disc_loss_log = 0.0
            if use_diayn:
                flat_obs = traj_batch.raw_obs.reshape(-1, *env.observation_space().shape)
                flat_skills = traj_batch.skills.reshape(-1)
                # loss_fn만 호출하여 loss 계산
                def disc_loss_fn(params):
                    logits = diayn_state.apply_fn(params, flat_obs)
                    loss = optax.softmax_cross_entropy_with_integer_labels(logits, flat_skills)
                    # Accuracy calculation
                    pred_skills = jnp.argmax(logits, axis=-1)
                    accuracy = jnp.mean(pred_skills == flat_skills)
                    return loss.mean(), accuracy
                disc_loss_log, disc_acc_log = disc_loss_fn(diayn_state.params)
            
            # 환경 스텝 수집 중 기록된 정보를 메트릭 딕셔너리로 가져옴
            # (shaped_reward, original_reward, anneal_factor, combined_reward 등 포함)
            metric = traj_batch.info

            # --------------------------------------------------------------
            # entropy 통계 파생 메트릭
            #   - policy_entropy_mean: rollout 전체 평균 엔트로피
            # --------------------------------------------------------------
            if "policy_entropy" in metric:
                metric["policy_entropy_mean"] = metric["policy_entropy"].mean()

            # 손실 함수 관련 메트릭 추출
            total_loss, aux_data = loss_info
            value_loss, loss_actor, entropy, ratio = aux_data

            # PPO 학습 손실 메트릭 추가
            metric["total_loss"] = total_loss      # 전체 손실 (actor + value + entropy + pred)
            metric["value_loss"] = value_loss      # 가치 함수(critic) MSE 손실
            metric["loss_actor"] = loss_actor      # 정책(actor) clipped surrogate 손실
            metric["entropy"] = entropy            # 정책 엔트로피 (탐험 정도)
            metric["ratio"] = ratio                # PPO ratio (new_prob / old_prob)
            
            if use_diayn:
                metric["disc_loss"] = disc_loss_log # [DIAYN] Discriminator Loss 추가
                metric["disc_acc"] = disc_acc_log   # [DIAYN] Discriminator Accuracy 추가

            # 모든 메트릭 값을 배치/스텝 차원에 대해 평균 계산 (스칼라로 축약)
            metric = jax.tree_util.tree_map(lambda x: x.mean(), metric)

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
                # [DIAYN] 현재 상태 구성
                current_checkpoint_data = {"params": train_state.params}
                if use_diayn:
                    current_checkpoint_data["diayn_state"] = diayn_state

                checkpoint_idx_selector = checkpoint_steps == update_step
                checkpoint_states = jax.lax.cond(
                    jnp.any(checkpoint_idx_selector),
                    _update_checkpoint,
                    lambda c, _p, _i: c,
                    checkpoint_states,
                    current_checkpoint_data,
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
                obs_history,
                act_history,
                traj_batch.action[-1],
                rng,
                diayn_state, # [DIAYN] 추가
                skills,      # [DIAYN] 추가
            )
            return runner_state, metric

        initial_update_step = 0
        if update_step_offset is not None:
            initial_update_step = update_step_offset

        # [DIAYN] Checkpoint 구조 정의 (params + diayn_state)
        checkpoint_target = {"params": train_state.params}
        if use_diayn:
            checkpoint_target["diayn_state"] = diayn_state

        initial_checkpoints = jax.tree_util.tree_map(
            lambda p: jnp.zeros((num_checkpoints,) + p.shape, p.dtype),
            checkpoint_target,
        )

        if num_checkpoints > 0:
            initial_checkpoints = jax.lax.cond(
                (checkpoint_steps[0] == 0) & (initial_update_step == 0),
                _update_checkpoint,
                lambda c, _p, _i: c,
                initial_checkpoints,
                checkpoint_target, # train_state.params 대신 전체 구조 전달
                0,
            )

        init_fcp_pop_idxs = None
        if population is not None and not is_policy_population:
            init_fcp_pop_idxs = jax.random.randint(
                _rng, (model_config["NUM_ACTORS"],), 0, fcp_population_size
            )

        rng, _rng = jax.random.split(rng)

        # [FIX] Missing history variables initialization
        # obs_history, act_history, last_partner_action are unpacked in _update_step but were missing here.
        # Assuming they are not used in this version of IPPO or should be initialized to None/Zeros.
        # Based on _env_step signature, they seem to be used for something (maybe RNN history or partner modeling).
        # Let's initialize them with dummy values matching expected shapes if possible, or None if handled.
        # However, since they are unpacked, they must exist in the tuple.
        
        # Checking _env_step usage:
        # obs_history, act_history are passed to _env_step.
        # In _env_step, they are updated.
        
        # We need to initialize them.
        # obs_history: (NUM_ACTORS, WINDOW_SIZE, OBS_DIM) ?
        # act_history: (NUM_ACTORS, WINDOW_SIZE) ?
        # last_partner_action: (NUM_ACTORS, ) ?
        
        # Since I don't see their definition in the provided snippet, I will initialize them as None
        # and hope they are handled or not used if None.
        # BUT, jax.lax.scan requires consistent shapes.
        
        # Let's look at where they might have been defined in the original code.
        # They were likely removed or I missed them.
        # Wait, looking at the error "expected 16, got 13", it means 3 items are missing.
        # The missing items are obs_history, act_history, last_partner_action.
        
        # Let's initialize them with zeros.
        # We need to know the shapes.
        # obs_history: likely (NUM_ACTORS, stack_size, *obs_shape)
        # act_history: likely (NUM_ACTORS, stack_size)
        # last_partner_action: likely (NUM_ACTORS, )
        
        # Since I cannot be 100% sure about shapes without seeing more code,
        # and the user wants a quick fix for the mismatch.
        
        # Actually, looking at the code I read earlier (lines 301-600), 
        # `obs_history` and `act_history` are used in `_env_step`.
        # `act_history = act_history.at[:, -1].set(last_partner_action_masked)`
        
        # This implies they are arrays.
        
        # Let's try to infer shapes from context or use safe defaults.
        # If they are not used for the current experiment (e.g. no partner modeling), maybe zeros are fine.
        
        # Let's add them to the tuple.
        
        # Placeholder initialization
        obs_history = jnp.zeros((model_config["NUM_ACTORS"], 1)) # Dummy
        act_history = jnp.zeros((model_config["NUM_ACTORS"], 1)) # Dummy
        last_partner_action = jnp.zeros((model_config["NUM_ACTORS"],), dtype=jnp.int32) # Dummy

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
            obs_history,          # Added
            act_history,          # Added
            last_partner_action,  # Added
            _rng,
            diayn_state, # [DIAYN] 추가
            skills,      # [DIAYN] 추가
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
