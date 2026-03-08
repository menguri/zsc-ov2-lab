from typing import List
import jax
import jax.numpy as jnp
import chex
from .policy import AbstractPolicy, PolicyPairing
from overcooked_v2_experiments.ppo.models.e3t import DiClusterEncoder


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
        # [E3D Support] Extract partner_prediction from kwargs
        partner_preds_dict = kwargs.get("partner_prediction", None)

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
            
            # [E3D Support] Pass partner_prediction if available
            if partner_preds_dict is not None and agent_id in partner_preds_dict:
                 policy_kwargs["partner_prediction"] = partner_preds_dict[agent_id]

            action, next_hstate, extras = policy.compute_action(
                obs_agent, done_agent, hstate_agent, sample_keys[i], **policy_kwargs
            )
            actions[agent_id] = action
            next_hstates[agent_id] = next_hstate
            all_extras[agent_id] = extras

        return actions, next_hstates, all_extras

    return init_hstate, _get_actions


def get_rollout(policies: PolicyPairing, env, key, algorithm="PPO", ego_direction=False, ov_type="ov2") -> PolicyRollout:
    init_hstate, _get_actions = init_rollout(policies, env)

    # obs_history 및 act_history 초기화 (E3T/E3D인 경우에만)
    init_obs_history = None
    init_act_history = None
    if algorithm in ["E3T", "STL", "E3D"]:
        # k=5라고 가정
        k = 5
        # [E3D Support] Use HISTORY_LENGTH from config if available
        if algorithm == "E3D":
             # policies is PolicyPairing, efficient list-like access
             try:
                 # Check first policy's config
                 if hasattr(policies[0], "config"):
                     cfg = policies[0].config
                     if "model" in cfg:
                         k = cfg["model"].get("HISTORY_LENGTH", k)
                     else:
                         k = cfg.get("HISTORY_LENGTH", k)
             except Exception:
                 pass
        
        obs_shape = env.observation_space().shape
        # obs_history는 각 에이전트에 대한 배열의 딕셔너리입니다
        init_obs_history = {
            f"agent_{i}": jnp.zeros((k, *obs_shape)) for i in range(env.num_agents)
        }
        # act_history는 각 에이전트에 대한 정수 배열의 딕셔너리입니다
        init_act_history = {
            f"agent_{i}": jnp.zeros((k,), dtype=jnp.int32) for i in range(env.num_agents)
        }
    
    # E3D Encoder
    di_cluster_encoder = DiClusterEncoder()
    # ov_type is passed as argument



    @jax.jit
    def _perform_step(carry, key):
        obs, state, done, total_reward, hstate, obs_history, act_history = carry

        # obs_history 업데이트 (E3T/E3D인 경우에만)
        next_obs_history = None
        if algorithm in ["E3T", "STL", "E3D"]:
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
        # E3D인 경우 partner_prediction (z) 계산 및 전달
        kwargs = {}
        if algorithm in ["E3T", "STL"]:
            kwargs["obs_history"] = next_obs_history if next_obs_history is not None else obs_history
            kwargs["act_history"] = act_history # act_history는 아직 업데이트 전 (이전 스텝까지의 파트너 행동)
        elif algorithm == "E3D":
             # E3D Logic: Compute z using DiClusterEncoder
             # Prepare batch-like inputs for DiClusterEncoder (add Batch dim)
             # obs_history dict -> array (B, Time, ...) where B=1 per agent
             # But DiClusterEncoder takes batch of history.
             
             # Agent별로 z 계산
             z_inputs_obs = []
             z_inputs_act = []
             
             # Use updated history
             current_obs_hist = next_obs_history if next_obs_history is not None else obs_history
             current_act_hist = act_history
             
             # Stack agents
             # obs_hist_batch: (NumAgents, 5, H, W, C)
             obs_hist_batch = jnp.stack([current_obs_hist[f"agent_{i}"] for i in range(env.num_agents)])
             act_hist_batch = jnp.stack([current_act_hist[f"agent_{i}"] for i in range(env.num_agents)])
             
             # Add Batch Dimension: (NumAgents, 5, ...) -> (1, NumAgents, 5, ...) ???
             # No, DiClusterEncoder expects (Batch, Time, ...). Here Batch=NumAgents
             # Wait, DiClusterEncoder applies to a batch of trajectories.
             # Here we have NumAgents parallel "environments" effectively for z calculation.
             
             z_ego_onehot, z_part_onehot = di_cluster_encoder.apply({}, obs_hist_batch, act_hist_batch, ov_type)
             # z_ego_onehot: (NumAgents, 16)
             
             # Construct Z
             # Same logic as ippo.py
             real_z_part = z_part_onehot
             real_z_ego = z_ego_onehot
             
             if ego_direction:
                  real_z_input = jnp.concatenate([real_z_part, real_z_ego], axis=-1)
             else:
                  real_z_input = real_z_part
                  
             # Partner Logic (Random)
             # Here, during rollout, we assume each agent is playing according to its policy (Ego).
             # We feed "real_z_input" to each agent's policy.
             # Note: visualize_ppo uses PPOParams which are trained weights.
             # If we want to simulate "Self-Play" where both are Ego, we give Real Z to both.
             # Training logic had "Ego vs Partner(Random)".
             # In evaluation (SP/XP), agents act as themselves (Ego).
             # So we provide the Real Z derived from their perspective.
             
             # Construct partner_prediction dictionary
             extras_z = {}
             # Add Batch dim (1, Dict) -> Policy expects (Batch, Dim)
             # Here batch size is 1 per agent call in _get_actions loop
             # Or we can pass dict and let _get_actions handle it.
             # _get_actions iterates policies.
             # Let's pass a dictionary of z inputs: {agent_id: z_vector}
             
             extras_z = {f"agent_{i}": real_z_input[i][jnp.newaxis, :] for i in range(env.num_agents)}
             kwargs["partner_prediction"] = extras_z

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
        
        # act_history 업데이트 (E3T/E3D인 경우에만)
        # 주의: act_history는 '파트너'의 행동을 저장해야 함
        # agent_0의 act_history에는 agent_1의 행동을, agent_1에는 agent_0의 행동을 저장
        next_act_history = None
        if algorithm in ["E3T", "STL", "E3D"]:
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
