from typing import List
import jax
import jax.numpy as jnp
import chex
from .policy import AbstractPolicy, PolicyPairing


@chex.dataclass
class PolicyRollout:
    state_seq: chex.Array
    actions_seq: chex.Array
    total_reward: chex.Scalar
    prediction_accuracy: chex.Array = None  # (num_agents,)


def init_rollout(policies: List[AbstractPolicy], env):
    num_agents = env.num_agents

    assert len(policies) == num_agents

    # print("Policy types", [type(p) for p in policies])

    init_hstate = {f"agent_{i}": policies[i].init_hstate(1) for i in range(num_agents)}

    @jax.jit
    def _get_actions(obs, done, hstate, key, **kwargs):
        sample_keys = jax.random.split(key, num_agents)

        actions = {}
        next_hstates = {}
        all_extras = {}
        
        obs_history = kwargs.get("obs_history", None)
        act_history = kwargs.get("act_history", None)

        for i, policy in enumerate(policies):
            agent_id = f"agent_{i}"

            obs_agent, done_agent, hstate_agent = (
                obs[agent_id],
                done[agent_id],
                hstate[agent_id],
            )

            # 가능한 경우 에이전트별 히스토리 추출
            policy_kwargs = {}
            if obs_history is not None:
                policy_kwargs["obs_history"] = obs_history[agent_id]
            if act_history is not None:
                policy_kwargs["act_history"] = act_history[agent_id]

            action, next_hstate, extras = policy.compute_action(
                obs_agent, done_agent, hstate_agent, sample_keys[i], **policy_kwargs
            )
            actions[agent_id] = action
            next_hstates[agent_id] = next_hstate
            all_extras[agent_id] = extras

        return actions, next_hstates, all_extras

    return init_hstate, _get_actions


def get_rollout(policies: PolicyPairing, env, key, algorithm="PPO") -> PolicyRollout:
    init_hstate, _get_actions = init_rollout(policies, env)

    # obs_history 및 act_history 초기화 (E3T인 경우에만)
    init_obs_history = None
    init_act_history = None
    if algorithm in ["E3T", "STL"]:
        # k=5라고 가정
        k = 5
        obs_shape = env.observation_space().shape
        # obs_history는 각 에이전트에 대한 배열의 딕셔너리입니다
        init_obs_history = {
            f"agent_{i}": jnp.zeros((k, *obs_shape)) for i in range(env.num_agents)
        }
        # act_history는 각 에이전트에 대한 정수 배열의 딕셔너리입니다
        init_act_history = {
            f"agent_{i}": jnp.zeros((k,), dtype=jnp.int32) for i in range(env.num_agents)
        }

    @jax.jit
    def _perform_step(carry, key):
        obs, state, done, total_reward, hstate, obs_history, act_history = carry

        # obs_history 업데이트 (E3T인 경우에만)
        next_obs_history = None
        if algorithm in ["E3T", "STL"]:
            def update_history(hist, new_obs, is_done):
                # 완료 시 리셋
                hist = jax.lax.select(is_done, jnp.zeros_like(hist), hist)
                
                # 시프트 및 추가
                hist = jnp.roll(hist, shift=-1, axis=0)
                hist = hist.at[-1].set(new_obs)
                return hist

            next_obs_history = {}
            for i in range(env.num_agents):
                agent_id = f"agent_{i}"
                next_obs_history[agent_id] = update_history(
                    obs_history[agent_id], obs[agent_id], done[agent_id]
                )

        key_sample, key_step = jax.random.split(key, 2)
        
        # E3T인 경우에만 obs_history 및 act_history 전달
        kwargs = {}
        if algorithm in ["E3T", "STL"]:
            kwargs["obs_history"] = next_obs_history if next_obs_history is not None else obs_history
            kwargs["act_history"] = act_history # act_history는 아직 업데이트 전 (이전 스텝까지의 파트너 행동)

        actions, next_hstate, extras = _get_actions(obs, done, hstate, key_sample, **kwargs)

        # Calculate prediction accuracy
        prediction_correct = jnp.zeros(env.num_agents, dtype=jnp.float32)
        prediction_mask = jnp.zeros(env.num_agents, dtype=jnp.float32)

        if algorithm in ["E3T", "STL"]:
            for i in range(env.num_agents):
                agent_id = f"agent_{i}"
                partner_idx = (i + 1) % env.num_agents
                partner_id = f"agent_{partner_idx}"
                
                if agent_id in extras and "partner_prediction" in extras[agent_id]:
                    pred_logits = extras[agent_id]["partner_prediction"]
                    pred_action = jnp.argmax(pred_logits)
                    true_action = actions[partner_id]
                    
                    is_correct = (pred_action == true_action).astype(jnp.float32)
                    prediction_correct = prediction_correct.at[i].set(is_correct)
                    prediction_mask = prediction_mask.at[i].set(1.0)

        # STEP ENV
        next_obs, next_state, reward, next_done, info = env.step(
            key_step, state, actions
        )
        
        # act_history 업데이트 (E3T인 경우에만)
        # 주의: act_history는 '파트너'의 행동을 저장해야 함
        # agent_0의 act_history에는 agent_1의 행동을, agent_1에는 agent_0의 행동을 저장
        next_act_history = None
        if algorithm in ["E3T", "STL"]:
            def update_act_history(hist, partner_act, is_done):
                # 완료 시 리셋
                hist = jax.lax.select(is_done, jnp.zeros_like(hist), hist)
                
                # 시프트 및 추가
                hist = jnp.roll(hist, shift=-1, axis=0)
                hist = hist.at[-1].set(partner_act)
                return hist
            
            next_act_history = {}
            # 2인용 게임 가정
            if env.num_agents == 2:
                # Agent 0의 파트너는 Agent 1
                next_act_history["agent_0"] = update_act_history(
                    act_history["agent_0"], actions["agent_1"], done["agent_0"]
                )
                # Agent 1의 파트너는 Agent 0
                next_act_history["agent_1"] = update_act_history(
                    act_history["agent_1"], actions["agent_0"], done["agent_1"]
                )
            else:
                # 2인 이상인 경우 정의가 모호하므로 일단 자기 자신을 제외한 첫 번째 에이전트 등으로 정의하거나
                # E3T가 2인용으로 설계되었다면 에러를 띄우는 게 맞음.
                # 여기서는 일단 0 <-> 1 만 처리하고 나머지는 0으로 채움 (임시)
                for i in range(env.num_agents):
                    agent_id = f"agent_{i}"
                    partner_idx = (i + 1) % env.num_agents # 다음 에이전트를 파트너로 가정
                    partner_id = f"agent_{partner_idx}"
                    next_act_history[agent_id] = update_act_history(
                        act_history[agent_id], actions[partner_id], done[agent_id]
                    )

        new_total_reward = total_reward + reward["agent_0"]

        carry = (next_obs, next_state, next_done, new_total_reward, next_hstate, next_obs_history, next_act_history)
        return carry, (next_state, actions, prediction_correct, prediction_mask)

    key, key_r = jax.random.split(key, 2)
    obs, state = env.reset(key_r)

    init_done = {f"agent_{i}": False for i in range(env.num_agents)}
    init_done["__all__"] = False

    keys = jax.random.split(key, env.max_steps)
    carry = (
        obs,
        state,
        init_done,
        0.0,
        init_hstate,
        init_obs_history,
        init_act_history,
    )
    carry, (state_seq, actions_seq, prediction_correct_seq, prediction_mask_seq) = jax.lax.scan(_perform_step, carry, keys)

    total_reward = carry[3] # Index 3 is total_reward

    # Calculate mean accuracy per agent
    total_correct = jnp.sum(prediction_correct_seq, axis=0)
    total_count = jnp.sum(prediction_mask_seq, axis=0)
    prediction_accuracy = jnp.where(total_count > 0, total_correct / total_count, 0.0)

    return PolicyRollout(
        state_seq=state_seq,
        actions_seq=actions_seq,
        total_reward=total_reward,
        prediction_accuracy=prediction_accuracy
    )
