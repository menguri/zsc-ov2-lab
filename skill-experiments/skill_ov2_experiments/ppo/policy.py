from functools import partial
from skill_ov2_experiments.eval.policy import AbstractPolicy
from skill_ov2_experiments.ppo.models.abstract import ActorCriticBase
from skill_ov2_experiments.ppo.models.model import (
    get_actor_critic,
    initialize_carry,
)
import jax.numpy as jnp
from skill_ov2_experiments.eval.policy import PolicyPairing
import jax
from flax import core
from typing import Any
import chex
from .diayn import augment_obs_with_skill


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

        # network.apply 호출
        next_hstate, pi, _ = self.network.apply(
            params, 
            hstate, 
            ac_in
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

        extras = {}

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


class DIAYNPPOPolicy(PPOPolicy):
    skill_idx: int
    num_skills: int

    def __init__(self, params, config, skill_idx, num_skills, stochastic=True, with_batching=False):
        super().__init__(params, config, stochastic, with_batching)
        self.skill_idx = skill_idx
        self.num_skills = num_skills

    def compute_action(self, obs, done, hstate, key, params=None, obs_history=None, act_history=None):
        # obs: (H, W, C) or (Batch, H, W, C) if with_batching
        
        # augment_obs_with_skill은 (H, W, C) 입력을 기대함
        # with_batching인 경우 vmap을 사용해야 함
        
        if self.with_batching:
             # obs: (Batch, H, W, C)
             augmented_obs = jax.vmap(augment_obs_with_skill, in_axes=(0, None, None))(
                 obs, self.skill_idx, self.num_skills
             )
        else:
             # obs: (H, W, C)
             augmented_obs = augment_obs_with_skill(obs, self.skill_idx, self.num_skills)
             
        return super().compute_action(augmented_obs, done, hstate, key, params, obs_history, act_history)
