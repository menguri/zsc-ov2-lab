import abc
from typing import List, Tuple
import chex
from functools import partial
import jax
from jax.tree_util import register_pytree_node_class
from flax import struct


class AbstractPolicy(abc.ABC):
    @abc.abstractmethod
    @partial(jax.jit, static_argnums=(0,))
    def compute_action(self, obs, done, hstate, key, **kwargs) -> Tuple[int, chex.Array, dict]:
        """
        관측값, 완료 플래그, 은닉 상태, 랜덤 키가 주어졌을 때 행동을 계산합니다.

        Args:
            obs (chex.Array): 관측값.
            done (chex.Array): 완료 플래그.
            hstate (chex.Array): 은닉 상태.
            key (chex.Array): 랜덤 키.
            **kwargs: obs_history, act_history와 같은 추가적인 인자들.

        Returns:
            Tuple[int, chex.Array, dict]: 행동, 새로운 은닉 상태, 추가 정보를 포함하는 튜플.
        """

        pass

    def init_hstate(self, batch_size, key=None) -> chex.Array:
        return None


@register_pytree_node_class
class PolicyPairing:
    policies: List[AbstractPolicy]

    def __init__(self, *policies):
        self.policies = policies

    @staticmethod
    def from_single_policy(policy: AbstractPolicy, num_agents: int):
        return PolicyPairing(*[policy for _ in range(num_agents)])

    def __getitem__(self, i):
        return self.policies[i]

    def __len__(self):
        return len(self.policies)

    def __iter__(self):
        return iter(self.policies)

    def __repr__(self):
        return f"PolicyPairing({self.policies})"

    def __str__(self):
        return f"PolicyPairing({self.policies})"

    def tree_flatten(self):
        children = self.policies
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
