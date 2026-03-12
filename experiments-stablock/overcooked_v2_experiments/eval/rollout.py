from typing import List
import jax
import jax.numpy as jnp
import chex
from .policy import AbstractPolicy, PolicyPairing
from .utils import extract_global_full_obs, extract_pos_yx


@chex.dataclass
class PolicyRollout:
    state_seq: chex.Array
    actions_seq: chex.Array
    value_seq: chex.Array
    pos_seq: chex.Array
    total_reward: chex.Scalar
    penalized_total_reward: chex.Scalar = None
    ph1_distance_mean: chex.Scalar = None
    ph1_penalty_mean: chex.Scalar = None
    prediction_accuracy: chex.Array = None  # (num_agents,)
    value_by_et_seq: chex.Array = None
    step_reward_seq: chex.Array = None
    cumulative_reward_seq: chex.Array = None
    ph1_distance_seq: chex.Array = None
    ph1_penalty_seq: chex.Array = None


def init_rollout(policies: List[AbstractPolicy], env):
    num_agents = env.num_agents

    assert len(policies) == num_agents

    # print("Policy types", [type(p) for p in policies])

    init_hstate = {f"agent_{i}": policies[i].init_hstate(1) for i in range(num_agents)}

    def _get_actions(obs, done, hstate, key, **kwargs):
        sample_keys = jax.random.split(key, num_agents)

        actions = {}
        next_hstates = {}
        all_extras = {}
        
        obs_history = kwargs.get("obs_history", None)
        act_history = kwargs.get("act_history", None)
        blocked_states = kwargs.get("blocked_states", None)
        # agent_idx conditioning was removed from the PPO policy/network.

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
            if blocked_states is not None and agent_id in blocked_states:
                policy_kwargs["blocked_states"] = blocked_states[agent_id]

            action, next_hstate, extras = policy.compute_action(
                obs_agent, done_agent, hstate_agent, sample_keys[i], **policy_kwargs
            )
            actions[agent_id] = action
            next_hstates[agent_id] = next_hstate
            all_extras[agent_id] = extras

        return actions, next_hstates, all_extras

    return init_hstate, _get_actions


def get_rollout(
    policies: PolicyPairing,
    env,
    key,
    algorithm="PPO",
    stablock_enabled=None,
    forced_blocked_states=None,
    value_by_et=False,
    et_candidates=None,
    target_agent=None,
    ph1_forced_tilde_state=None,
    ph1_omega: float = 1.0,
    ph1_sigma: float = 1.0,
    max_rollout_steps=None,
    env_device=None,  # None | "cpu" | "gpu"
) -> PolicyRollout:
    init_hstate, _get_actions = init_rollout(policies, env)
    ph1_enabled = "PH1" in algorithm
    env_name = "overcooked_v2" if hasattr(env, "get_obs_default") else "overcooked"

    # PH1 uses the same predictor-conditioned execution path as E3T during eval.
    e3t_like = (
        algorithm in ["E3T", "STL"]
        or "E3T" in algorithm
        or "STL" in algorithm
        or ph1_enabled
    )
    policy0_cfg = getattr(policies[0], "config", {}) if len(policies) > 0 else {}
    ph1_multi_penalty_enabled = bool(policy0_cfg.get("PH1_MULTI_PENALTY_ENABLED", False))
    ph1_max_penalty_count = int(policy0_cfg.get("PH1_MAX_PENALTY_COUNT", 1))
    if isinstance(policy0_cfg, dict) and "alg" in policy0_cfg:
        ph1_multi_penalty_enabled = bool(
            policy0_cfg["alg"].get(
                "PH1_MULTI_PENALTY_ENABLED", ph1_multi_penalty_enabled
            )
        )
        ph1_max_penalty_count = int(
            policy0_cfg["alg"].get("PH1_MAX_PENALTY_COUNT", ph1_max_penalty_count)
        )
    ph1_max_penalty_count = max(1, ph1_max_penalty_count)
    ph1_penalty_slots = ph1_max_penalty_count if ph1_multi_penalty_enabled else 1

    def _encode_policy_metric_emb(policy, blocked_actor):
        return policy.network.apply(
            policy.params,
            blocked_actor,
            method=policy.network.encode_blocked,
        )

    def _get_blocked_metric_slots_from_extras(agent_extras, fallback_slot):
        if isinstance(agent_extras, dict):
            slot_metric = agent_extras.get("blocked_emb_slots", None)
            if slot_metric is not None:
                slot_metric = jnp.asarray(slot_metric, dtype=jnp.float32)
                if slot_metric.ndim == 1:
                    slot_metric = slot_metric[None, :]
                return slot_metric
        fallback = fallback_slot
        if isinstance(agent_extras, dict):
            fallback = agent_extras.get("blocked_emb", fallback_slot)
        fallback = jnp.asarray(fallback, dtype=jnp.float32)
        if fallback.ndim == 1:
            fallback = fallback[None, :]
        if fallback.shape[0] == 1 and ph1_penalty_slots > 1:
            fallback = jnp.tile(fallback, (ph1_penalty_slots, 1))
        return fallback[:ph1_penalty_slots]

    # Optional backend pinning for environment interaction.
    reset_fn = env.reset
    step_fn = env.step
    if env_device is not None:
        dev = str(env_device).strip().lower()
        if dev == "cpu":
            reset_fn = jax.jit(env.reset, backend="cpu")
            step_fn = jax.jit(env.step, backend="cpu")
        elif dev in ("gpu", "cuda"):
            reset_fn = jax.jit(env.reset, backend="gpu")
            step_fn = jax.jit(env.step, backend="gpu")

    # 초기 reset 먼저 수행해서 PH1 full-view shape에 맞춘 history를 구성
    key, key_r = jax.random.split(key, 2)
    obs, state = reset_fn(key_r)

    # obs_history 및 act_history 초기화 (E3T인 경우에만)
    init_obs_history = None
    init_act_history = None
    if e3t_like:
        # k=5라고 가정
        k = 5
        obs_shape = obs["agent_0"].shape
        # obs_history는 각 에이전트에 대한 배열의 딕셔너리입니다
        init_obs_history = {
            f"agent_{i}": jnp.zeros((k, *obs_shape)) for i in range(env.num_agents)
        }
        # act_history는 각 에이전트에 대한 정수 배열의 딕셔너리입니다
        init_act_history = {
            f"agent_{i}": jnp.zeros((k,), dtype=jnp.int32) for i in range(env.num_agents)
        }

    stablock_enabled = (
        [False] * env.num_agents if stablock_enabled is None else stablock_enabled
    )
    stablock_enabled = [bool(x) for x in stablock_enabled]

    blocked_states = None
    ph1_forced_tilde_state = (
        jnp.array(ph1_forced_tilde_state, dtype=jnp.float32)
        if ph1_forced_tilde_state is not None
        else None
    )
    if not ph1_enabled:
        if forced_blocked_states is not None:
            blocked_states = {
                f"agent_{i}": jnp.array(forced_blocked_states[i], dtype=jnp.int32)
                for i in range(env.num_agents)
            }
        elif any(stablock_enabled):
            blocked_states = {}
            for i, enabled in enumerate(stablock_enabled):
                if enabled:
                    blocked_states[f"agent_{i}"] = jnp.array([-1, -1], dtype=jnp.int32)

    if value_by_et:
        if et_candidates is None or target_agent is None:
            raise ValueError("et_candidates and target_agent are required when value_by_et=True")
        et_candidates = [jnp.array(et, dtype=jnp.int32) for et in et_candidates]

    def _build_ph1_blocked_states_step(global_full):
        if ph1_penalty_slots <= 1:
            if ph1_forced_tilde_state is None:
                tilde = jnp.full(global_full.shape, -1.0, dtype=jnp.float32)
            else:
                forced = ph1_forced_tilde_state.astype(jnp.float32)
                if forced.ndim == global_full.ndim + 1:
                    forced = forced[0]
                tilde = forced
        else:
            tilde_slots = jnp.full(
                (ph1_penalty_slots,) + global_full.shape,
                -1.0,
                dtype=jnp.float32,
            )
            if ph1_forced_tilde_state is not None:
                forced = ph1_forced_tilde_state.astype(jnp.float32)
                if forced.ndim == global_full.ndim + 1:
                    use_slots = min(ph1_penalty_slots, int(forced.shape[0]))
                    tilde_slots = tilde_slots.at[:use_slots].set(forced[:use_slots])
                else:
                    tilde_slots = tilde_slots.at[0].set(forced)
            tilde = tilde_slots
        return {f"agent_{i}": tilde for i in range(env.num_agents)}

    def _compute_ph1_eval_step_stats(next_state, extras, blocked_states_step):
        step_penalty_env = jnp.float32(0.0)
        step_dist_env = jnp.float32(0.0)
        policy0_uses_blocked = True
        if hasattr(policies[0], "config") and isinstance(getattr(policies[0], "config"), dict):
            policy0_uses_blocked = bool(
                policies[0].config.get("LEARNER_USE_BLOCKED_INPUT", True)
            )

        if (
            ph1_enabled
            and policy0_uses_blocked
            and blocked_states_step is not None
            and hasattr(policies[0], "network")
            and hasattr(policies[0], "params")
        ):
            policy0 = policies[0]
            global_full_next = extract_global_full_obs(env, next_state, env_name)
            global_full_next_actor = jnp.stack(
                [global_full_next for _ in range(env.num_agents)], axis=0
            )
            z_next = _encode_policy_metric_emb(policy0, global_full_next_actor)

            blocked_actor = jnp.stack(
                [blocked_states_step[f"agent_{i}"] for i in range(env.num_agents)],
                axis=0,
            )
            if blocked_actor.ndim == 4:
                blocked_actor = blocked_actor[:, None, ...]
            num_slots = int(blocked_actor.shape[1])
            flat_blocks = blocked_actor.reshape(blocked_actor.shape[0], num_slots, -1)
            valid_slots = ~jnp.all(flat_blocks == -1.0, axis=-1)

            blocked_emb_slots = jnp.stack(
                [
                    _get_blocked_metric_slots_from_extras(
                        extras[f"agent_{i}"],
                        jnp.zeros_like(z_next[i]),
                    )[:num_slots]
                    for i in range(env.num_agents)
                ],
                axis=0,
            )
            if blocked_emb_slots.shape[1] == 1 and num_slots > 1:
                blocked_emb_slots = jnp.tile(blocked_emb_slots, (1, num_slots, 1))
            blocked_emb_slots = blocked_emb_slots[:, :num_slots, :]

            z_next_slots = z_next[:, None, :]
            lat_dist_slots = jnp.sqrt(
                jnp.sum((z_next_slots - blocked_emb_slots) ** 2, axis=-1)
            )
            lat_dist_slots = jnp.where(valid_slots, lat_dist_slots, 0.0)
            penalty_slots = ph1_omega * jnp.exp(-ph1_sigma * lat_dist_slots)
            penalty_slots = jnp.where(valid_slots, penalty_slots, 0.0)

            step_penalty_env = jnp.sum(penalty_slots, axis=-1).mean()
            step_dist_env = jnp.sum(lat_dist_slots, axis=-1).mean()

        return step_penalty_env, step_dist_env

    def _perform_step(carry, key):
        (
            obs,
            state,
            done,
            total_reward,
            penalized_total_reward,
            hstate,
            obs_history,
            act_history,
            dist_sum,
            pen_sum,
            stat_count,
        ) = carry

        # obs_history 업데이트 (E3T인 경우에만)
        next_obs_history = None
        if e3t_like:
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
                obs_src = obs[agent_id]
                next_obs_history[agent_id] = update_history(
                    obs_history[agent_id], obs_src, done[agent_id]
                )

        key_sample, key_step = jax.random.split(key, 2)

        # E3T인 경우에만 obs_history 및 act_history 전달
        kwargs = {}
        if e3t_like:
            kwargs["obs_history"] = next_obs_history if next_obs_history is not None else obs_history
            kwargs["act_history"] = act_history  # act_history는 아직 업데이트 전 (이전 스텝까지의 파트너 행동)
        if blocked_states is not None and e3t_like:
            kwargs["blocked_states"] = blocked_states

        obs_for_policy = obs
        blocked_states_step = None
        if ph1_enabled:
            # PH1 policy execution now uses each agent's observation (no full-obs override).
            # Only `blocked_states` should carry the global full target (tilde{s}).
            global_full = extract_global_full_obs(env, state, env_name)  # (H, W, C_full)
            blocked_states_step = _build_ph1_blocked_states_step(global_full)
            kwargs["blocked_states"] = blocked_states_step

        actions, next_hstate, extras = _get_actions(obs_for_policy, done, hstate, key_sample, **kwargs)

        # Collect critic values per agent (if provided by policy)
        value_vec = jnp.stack(
            [extras[f"agent_{i}"]["value"] for i in range(env.num_agents)]
        )

        # Collect agent positions (y, x) at current timestep
        pos_y, pos_x = extract_pos_yx(state, env_name)
        pos_vec = jnp.stack([pos_y, pos_x], axis=-1)

        # Calculate prediction accuracy
        prediction_correct = jnp.zeros(env.num_agents, dtype=jnp.float32)
        prediction_mask = jnp.zeros(env.num_agents, dtype=jnp.float32)

        if e3t_like:
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
        next_obs, next_state, reward, next_done, info = step_fn(
            key_step, state, actions
        )
        
        # act_history 업데이트 (E3T인 경우에만)
        # 주의: act_history는 '파트너'의 행동을 저장해야 함
        # agent_0의 act_history에는 agent_1의 행동을, agent_1에는 agent_0의 행동을 저장
        next_act_history = None
        if e3t_like:
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

        # PH1 evaluation-time penalty reconstruction
        step_penalty_env, step_dist_env = _compute_ph1_eval_step_stats(
            next_state,
            extras,
            blocked_states_step,
        )

        new_total_reward = total_reward + reward["agent_0"]
        penalized_total_reward = penalized_total_reward + (reward["agent_0"] - step_penalty_env)
        dist_sum = dist_sum + step_dist_env
        pen_sum = pen_sum + step_penalty_env
        stat_count = stat_count + jnp.float32(1.0)

        carry = (
            next_obs,
            next_state,
            next_done,
            new_total_reward,
            penalized_total_reward,
            next_hstate,
            next_obs_history,
            next_act_history,
            dist_sum,
            pen_sum,
            stat_count,
        )
        return carry, (
            next_state,
            actions,
            prediction_correct,
            prediction_mask,
            value_vec,
            pos_vec,
            reward["agent_0"],
            step_dist_env,
            step_penalty_env,
        )

    def _perform_step_with_et(carry, key):
        (
            obs,
            state,
            done,
            total_reward,
            penalized_total_reward,
            hstate,
            obs_history,
            act_history,
            dist_sum,
            pen_sum,
            stat_count,
        ) = carry

        # obs_history 업데이트 (E3T인 경우에만)
        next_obs_history = None
        if e3t_like:
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
                obs_src = obs[agent_id]
                next_obs_history[agent_id] = update_history(
                    obs_history[agent_id], obs_src, done[agent_id]
                )

        key_sample, key_step = jax.random.split(key, 2)

        # E3T인 경우에만 obs_history 및 act_history 전달
        kwargs = {}
        if e3t_like:
            kwargs["obs_history"] = next_obs_history if next_obs_history is not None else obs_history
            kwargs["act_history"] = act_history  # act_history는 아직 업데이트 전 (이전 스텝까지의 파트너 행동)
        if blocked_states is not None and e3t_like:
            kwargs["blocked_states"] = blocked_states

        obs_for_policy = obs
        blocked_states_step = None
        if ph1_enabled:
            global_full = extract_global_full_obs(env, state, env_name)  # (H, W, C_full)
            blocked_states_step = _build_ph1_blocked_states_step(global_full)
            kwargs["blocked_states"] = blocked_states_step

        actions, next_hstate, extras = _get_actions(obs_for_policy, done, hstate, key_sample, **kwargs)

        # Collect critic values per agent (if provided by policy)
        value_vec = jnp.stack(
            [extras[f"agent_{i}"]["value"] for i in range(env.num_agents)]
        )

        # Collect agent positions (y, x) at current timestep
        pos_y, pos_x = extract_pos_yx(state, env_name)
        pos_vec = jnp.stack([pos_y, pos_x], axis=-1)

        # Calculate prediction accuracy
        prediction_correct = jnp.zeros(env.num_agents, dtype=jnp.float32)
        prediction_mask = jnp.zeros(env.num_agents, dtype=jnp.float32)

        if e3t_like:
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
        next_obs, next_state, reward, next_done, info = step_fn(
            key_step, state, actions
        )

        # act_history 업데이트 (E3T인 경우에만)
        # 주의: act_history는 '파트너'의 행동을 저장해야 함
        # agent_0의 act_history에는 agent_1의 행동을, agent_1에는 agent_0의 행동을 저장
        next_act_history = None
        if e3t_like:
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
                    partner_idx = (i + 1) % env.num_agents  # 다음 에이전트를 파트너로 가정
                    partner_id = f"agent_{partner_idx}"
                    next_act_history[agent_id] = update_act_history(
                        act_history[agent_id], actions[partner_id], done[agent_id]
                    )

        # PH1 evaluation-time penalty reconstruction
        step_penalty_env, step_dist_env = _compute_ph1_eval_step_stats(
            next_state,
            extras,
            blocked_states_step,
        )

        new_total_reward = total_reward + reward["agent_0"]
        penalized_total_reward = penalized_total_reward + (reward["agent_0"] - step_penalty_env)
        dist_sum = dist_sum + step_dist_env
        pen_sum = pen_sum + step_penalty_env
        stat_count = stat_count + jnp.float32(1.0)

        # Compute value for each e_t candidate for target agent
        agent_id = f"agent_{target_agent}"
        policy = policies[target_agent]
        obs_agent = obs_for_policy[agent_id]
        done_agent = done[agent_id]
        hstate_agent = hstate[agent_id]

        policy_kwargs = {}
        if e3t_like:
            obs_hist_src = next_obs_history if next_obs_history is not None else obs_history
            policy_kwargs["obs_history"] = obs_hist_src[agent_id]
            policy_kwargs["act_history"] = act_history[agent_id]

        value_by_et_list = []
        for et in et_candidates:
            if not ph1_enabled:
                policy_kwargs["blocked_states"] = et
            _, _, extras_et = policy.compute_action(
                obs_agent, done_agent, hstate_agent, key_sample, **policy_kwargs
            )
            value_by_et_list.append(extras_et["value"])

        value_by_et_vec = jnp.stack(value_by_et_list)

        carry = (
            next_obs,
            next_state,
            next_done,
            new_total_reward,
            penalized_total_reward,
            next_hstate,
            next_obs_history,
            next_act_history,
            dist_sum,
            pen_sum,
            stat_count,
        )
        return carry, (
            next_state,
            actions,
            prediction_correct,
            prediction_mask,
            value_vec,
            pos_vec,
            value_by_et_vec,
            reward["agent_0"],
            step_dist_env,
            step_penalty_env,
        )

    init_done = {f"agent_{i}": False for i in range(env.num_agents)}
    init_done["__all__"] = False

    rollout_steps = int(env.max_steps)
    if max_rollout_steps is not None:
        try:
            rollout_steps = max(1, min(rollout_steps, int(max_rollout_steps)))
        except Exception:
            rollout_steps = int(env.max_steps)
    keys = jax.random.split(key, rollout_steps)
    carry = (
        obs,
        state,
        init_done,
        jnp.float32(0.0),
        jnp.float32(0.0),
        init_hstate,
        init_obs_history,
        init_act_history,
        jnp.float32(0.0),
        jnp.float32(0.0),
        jnp.float32(0.0),
    )
    value_by_et_seq = None
    step_reward_seq = None
    cumulative_reward_seq = None
    ph1_distance_seq = None
    ph1_penalty_seq = None
    if value_by_et:
        outputs = []
        for k in keys:
            carry, out = _perform_step_with_et(carry, k)
            outputs.append(out)
        def _stack_tree(seq):
            return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *seq)

        (
            state_seq_list,
            actions_seq_list,
            prediction_correct_seq_list,
            prediction_mask_seq_list,
            value_seq_list,
            pos_seq_list,
            value_by_et_seq_list,
            step_reward_seq_list,
            ph1_distance_seq_list,
            ph1_penalty_seq_list,
        ) = zip(*outputs)

        state_seq = _stack_tree(state_seq_list)
        actions_seq = _stack_tree(actions_seq_list)
        prediction_correct_seq = jnp.stack(prediction_correct_seq_list)
        prediction_mask_seq = jnp.stack(prediction_mask_seq_list)
        value_seq = jnp.stack(value_seq_list)
        pos_seq = jnp.stack(pos_seq_list)
        value_by_et_seq = jnp.stack(value_by_et_seq_list)
        step_reward_seq = jnp.stack(step_reward_seq_list)
        cumulative_reward_seq = jnp.cumsum(step_reward_seq)
        ph1_distance_seq = jnp.stack(ph1_distance_seq_list)
        ph1_penalty_seq = jnp.stack(ph1_penalty_seq_list)
    else:
        carry, (
            state_seq,
            actions_seq,
            prediction_correct_seq,
            prediction_mask_seq,
            value_seq,
            pos_seq,
            step_reward_seq,
            ph1_distance_seq,
            ph1_penalty_seq,
        ) = jax.lax.scan(_perform_step, carry, keys)
        cumulative_reward_seq = jnp.cumsum(step_reward_seq)

    total_reward = carry[3] # Index 3 is total_reward
    penalized_total_reward = carry[4]
    ph1_distance_mean = jnp.where(carry[10] > 0, carry[8] / carry[10], 0.0)
    ph1_penalty_mean = jnp.where(carry[10] > 0, carry[9] / carry[10], 0.0)

    # Calculate mean accuracy per agent
    total_correct = jnp.sum(prediction_correct_seq, axis=0)
    total_count = jnp.sum(prediction_mask_seq, axis=0)
    prediction_accuracy = jnp.where(total_count > 0, total_correct / total_count, 0.0)

    return PolicyRollout(
        state_seq=state_seq,
        actions_seq=actions_seq,
        value_seq=value_seq,
        pos_seq=pos_seq,
        total_reward=total_reward,
        penalized_total_reward=penalized_total_reward,
        ph1_distance_mean=ph1_distance_mean,
        ph1_penalty_mean=ph1_penalty_mean,
        prediction_accuracy=prediction_accuracy,
        value_by_et_seq=value_by_et_seq,
        step_reward_seq=step_reward_seq,
        cumulative_reward_seq=cumulative_reward_seq,
        ph1_distance_seq=ph1_distance_seq,
        ph1_penalty_seq=ph1_penalty_seq,
    )
