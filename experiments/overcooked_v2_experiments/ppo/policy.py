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
        self.config = config
        self.stochastic = stochastic
        self.with_batching = with_batching

        self.network = get_actor_critic(config)

        self.params = params

    def compute_action(self, obs, done, hstate, key, params=None, obs_history=None, act_history=None):
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
        if obs_history is not None:
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

            # predict_partner 호출 (Batch, k, ...) -> (Batch, ActionDim)
            # method='predict_partner'를 사용하여 ActorCriticRNN 내부의 predict_partner 메서드 호출
            pred = self.network.apply(params, obs_hist_in, act_hist_in, method='predict_partner')
            
            # (Batch, ActionDim) -> (1, Batch, ActionDim) (Time 차원 추가, ActorCriticRNN 입력용)
            partner_prediction = pred[jnp.newaxis, ...]

        # network.apply 호출
        # partner_prediction이 None이면 모델 내부에서 무시됨 (기존 로직)
        next_hstate, pi, _ = self.network.apply(params, hstate, ac_in, partner_prediction=partner_prediction)

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

        extras = {}
        if partner_prediction is not None:
            if self.with_batching:
                extras["partner_prediction"] = partner_prediction[0]
            else:
                extras["partner_prediction"] = partner_prediction[0, 0]

        return action, next_hstate, extras

    def init_hstate(self, batch_size, key=None):
        # assert batch_size == 1 or self.with_batching
        print("Initializing hstate with batch size", batch_size)
        return initialize_carry(self.config, batch_size)


def policy_checkoints_to_policy_pairing(checkpoints: PPOParams, config):
    policies = []
    for checkpoint in checkpoints:
        policies.append(PPOPolicy(checkpoint.params, config))

    return PolicyPairing(*policies)
