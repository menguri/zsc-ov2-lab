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
        
        for i, policy in enumerate(policies):
            agent_id = f"agent_{i}"

            obs_agent, done_agent, hstate_agent = (
                obs[agent_id],
                done[agent_id],
                hstate[agent_id],
            )

            action, next_hstate, extras = policy.compute_action(
                obs_agent, done_agent, hstate_agent, sample_keys[i]
            )
            actions[agent_id] = action
            next_hstates[agent_id] = next_hstate
            all_extras[agent_id] = extras

        return actions, next_hstates, all_extras

    return init_hstate, _get_actions


def get_rollout(policies: PolicyPairing, env, key, algorithm="PPO") -> PolicyRollout:
    init_hstate, _get_actions = init_rollout(policies, env)

    @jax.jit
    def _perform_step(carry, key):
        obs, state, done, total_reward, hstate = carry

        key_sample, key_step = jax.random.split(key, 2)
        
        actions, next_hstate, extras = _get_actions(obs, done, hstate, key_sample)

        # Calculate prediction accuracy (Dummy)
        prediction_correct = jnp.zeros(env.num_agents, dtype=jnp.float32)
        prediction_mask = jnp.zeros(env.num_agents, dtype=jnp.float32)

        # STEP ENV
        next_obs, next_state, reward, next_done, info = env.step(
            key_step, state, actions
        )
        
        next_total_reward = total_reward + sum(reward.values())

        return (
            next_obs,
            next_state,
            next_done,
            next_total_reward,
            next_hstate,
        ), (state, actions, prediction_correct)

    key_reset, key_scan = jax.random.split(key)
    obs, state = env.reset(key_reset)
    done = {f"agent_{i}": False for i in range(env.num_agents)}
    done["__all__"] = False

    init_carry = (obs, state, done, 0.0, init_hstate)

    # 400 steps rollout
    final_carry, (state_seq, actions_seq, pred_acc_seq) = jax.lax.scan(
        _perform_step, init_carry, jax.random.split(key_scan, 400)
    )

    # Average prediction accuracy
    # pred_acc_seq: (T, num_agents)
    # We just return the mean over time for now, or 0 if not applicable
    mean_pred_acc = jnp.mean(pred_acc_seq, axis=0)

    return PolicyRollout(
        state_seq=state_seq,
        actions_seq=actions_seq,
        total_reward=final_carry[3],
        prediction_accuracy=mean_pred_acc
    )
