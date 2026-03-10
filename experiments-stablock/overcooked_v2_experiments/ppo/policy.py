from functools import partial
from overcooked_v2_experiments.eval.policy import AbstractPolicy
from overcooked_v2_experiments.ppo.models.abstract import ActorCriticBase
from overcooked_v2_experiments.ppo.models.model import (
    get_actor_critic,
    initialize_carry,
)
import jax.numpy as jnp
from overcooked_v2_experiments.eval.policy import PolicyPairing
import jax
from flax import core
from typing import Any
import copy
import chex


@chex.dataclass
class PPOParams:
    params: core.FrozenDict[str, Any]


class PPOPolicy(AbstractPolicy):
    network: ActorCriticBase
    params: core.FrozenDict[str, Any]
    config: core.FrozenDict[str, Any]
    stochastic: bool = True
    with_batching: bool = False

    def __init__(self, params, config, stochastic=True, with_batching=False):
        config = copy.deepcopy(config)
        self.config = config
        self.stochastic = stochastic
        self.with_batching = with_batching

        self.network = get_actor_critic(config)

        self.params = params

    def compute_action(
        self,
        obs,
        done,
        hstate,
        key,
        params=None,
        obs_history=None,
        act_history=None,
        blocked_states=None,
    ):
        if params is None:
            params = self.params
        assert params is not None

        done = jnp.array(done)

        def _add_dim(tree):
            return jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], tree)

        ac_in = (obs, done)
        ac_in = _add_dim(ac_in)
        if not self.with_batching:
            ac_in = _add_dim(ac_in)

        # print("ac_in shapes", ac_in[0].shape, ac_in[1].shape, hstate.shape, type(params))

        # E3T: obs_history가 제공되면 파트너 행동 예측 수행
        partner_prediction = None
        
        # Check if this policy is E3T/PH1
        alg_name = self.config.get("ALG_NAME", "")
        if "alg" in self.config:
            alg_name = self.config["alg"].get("ALG_NAME", alg_name)

        ph1_enabled = "PH1" in alg_name or bool(self.config.get("PH1_ENABLED", False))
        if "alg" in self.config:
            ph1_enabled = ph1_enabled or bool(self.config["alg"].get("PH1_ENABLED", False))
        state_prediction = bool(self.config.get("STATE_PREDICTION", False))
        action_prediction = bool(self.config.get("ACTION_PREDICTION", True)) and (not state_prediction)
        learner_use_blocked_input = bool(self.config.get("LEARNER_USE_BLOCKED_INPUT", True))
        if "alg" in self.config:
            learner_use_blocked_input = bool(
                self.config["alg"].get("LEARNER_USE_BLOCKED_INPUT", learner_use_blocked_input)
            )

        e3t_like = ("E3T" in alg_name) or ("STL" in alg_name)
        if obs_history is not None:
            e3t_like = True
        
        # Check for STL (anchor)
        is_anchor = False
        if "anchor" in self.config:
            is_anchor = self.config["anchor"]
        elif "alg" in self.config and "anchor" in self.config["alg"]:
            is_anchor = self.config["alg"]["anchor"]
        elif "model" in self.config and "anchor" in self.config["model"]:
            is_anchor = self.config["model"]["anchor"]
            
        if e3t_like and is_anchor:
            alg_name = "STL"
            
        # E3T/STL Logic
        obs_hist_in = None
        act_hist_in = None
        partner_prediction = None
        pred_logits_for_extras = None
        blocked_states_in = None

        if obs_history is not None and e3t_like:
            # obs_history: (k, H, W, C) -> (1, k, H, W, C) (Batch 추가)
            # with_batching=True이면 이미 (Batch, k, H, W, C)이므로 추가 안 함
            if not self.with_batching:
                obs_hist_in = _add_dim(obs_history)
            else:
                obs_hist_in = obs_history
            
            # act_history 처리
            if act_history is not None:
                # act_history: (k,) -> (1, k) (Batch 추가)
                if not self.with_batching:
                    act_hist_in = _add_dim(act_history)
                else:
                    act_hist_in = act_history
            else:
                # act_history가 없으면 더미(0) 사용
                # obs_hist_in: (Batch, k, H, W, C)
                B = obs_hist_in.shape[0]
                k = obs_hist_in.shape[1]
                act_hist_in = jnp.zeros((B, k), dtype=jnp.int32)

            # Extract z_state from hstate if available
            z_state = None
            if isinstance(hstate, tuple) and len(hstate) == 2:
                _, z_state = hstate

            # predict_partner 호출 (Batch, k, ...) -> (Batch, ActionDim)
            # method='predict_partner'를 사용하여 ActorCriticRNN 내부의 predict_partner 메서드 호출
            if state_prediction:
                pred_state = self.network.apply(
                    params,
                    obs_hist_in,
                    act_hist_in,
                    z_state=z_state,
                    method="predict_partner_state",
                )
                pred_logits_for_extras = pred_state["action_logits"]
                partner_prediction = pred_state["context_z"][jnp.newaxis, ...]
            elif action_prediction:
                pred = self.network.apply(
                    params,
                    obs_hist_in,
                    act_hist_in,
                    z_state=z_state,
                    method="predict_partner",
                )
                pred_logits_for_extras = pred
                # (Batch, ActionDim) -> (1, Batch, ActionDim)
                partner_prediction = pred[jnp.newaxis, ...]

        if blocked_states is not None and learner_use_blocked_input and (e3t_like or ph1_enabled):
            blocked_states = jnp.array(blocked_states)
            if not self.with_batching:
                blocked_states_in = _add_dim(blocked_states)
            else:
                blocked_states_in = blocked_states
            if blocked_states_in.ndim == 2:
                blocked_states_in = blocked_states_in[jnp.newaxis, ...]

        # network.apply 호출
        # partner_prediction이 None이면 모델 내부에서 무시됨 (기존 로직)
        # Pass obs_history/act_history to ensure z_state is updated in __call__
        # [Fix] Unpack 4 values (added extras return from ActorCriticRNN)
        next_hstate, pi, value, net_extras = self.network.apply(
            params, 
            hstate, 
            ac_in, 
            partner_prediction=partner_prediction,
            obs_history=obs_hist_in,
            act_history=act_hist_in,
            blocked_states=blocked_states_in,
        )

        if self.stochastic:
            action = pi.sample(seed=key)
        else:
            action = jnp.argmax(pi.probs, axis=-1)

        if self.with_batching:
            action = action[0]
        else:
            action = action[0, 0]
        # action = action[0, 0]
        # action = action.flatten()
        # print("Action", action)

        # Extract scalar value for this agent
        if self.with_batching:
            value_scalar = value[0]
        else:
            value_scalar = value[0, 0]

        extras = {"value": value_scalar}
        if pred_logits_for_extras is not None:
            if self.with_batching:
                extras["partner_prediction"] = pred_logits_for_extras
            else:
                extras["partner_prediction"] = pred_logits_for_extras[0]

        if net_extras is not None and isinstance(net_extras, dict):
            blocked_emb = net_extras.get("blocked_emb", None)
            if blocked_emb is not None:
                if self.with_batching:
                    extras["blocked_emb"] = blocked_emb[0]
                else:
                    extras["blocked_emb"] = blocked_emb[0, 0]
            blocked_emb_slots = net_extras.get("blocked_emb_slots", None)
            if blocked_emb_slots is not None:
                if self.with_batching:
                    extras["blocked_emb_slots"] = blocked_emb_slots[0]
                else:
                    extras["blocked_emb_slots"] = blocked_emb_slots[0, 0]

        return action, next_hstate, extras

    def init_hstate(self, batch_size, key=None):
        # assert batch_size == 1 or self.with_batching
        return initialize_carry(self.config, batch_size)


def policy_checkoints_to_policy_pairing(checkpoints: PPOParams, config):
    policies = []
    for checkpoint in checkpoints:
        policies.append(PPOPolicy(checkpoint.params, config))

    return PolicyPairing(*policies)
