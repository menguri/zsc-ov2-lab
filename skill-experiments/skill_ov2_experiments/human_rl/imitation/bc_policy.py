from typing import Any
import jax
import jax.numpy as jnp
from skill_ov2_experiments.eval.policy import AbstractPolicy
from skill_ov2_experiments.human_rl.imitation.bc_model import BCModel
from jaxmarl.environments.overcooked_v2.common import Actions
from .utils import load_bc_checkpoint
import distrax
from flax import core
import chex
from .utils import remove_indices_and_renormalize


STATE_HISTORY_LEN = 3


@chex.dataclass
class BCHState:
    actions: chex.Array  # Shape: (STATE_HISTORY_LEN,)
    pos: chex.Array  # Shape: (STATE_HISTORY_LEN, 2)
    dir: chex.Array  # Shape: (STATE_HISTORY_LEN, 4)

    @staticmethod
    def from_numpy(arr: jnp.ndarray):
        history_len = STATE_HISTORY_LEN
        idx = 0

        actions_len = history_len
        pos_len = history_len * 2
        dir_len = history_len * 4

        actions = arr[idx : idx + actions_len].reshape((history_len,))
        idx += actions_len

        pos = arr[idx : idx + pos_len].reshape((history_len, 2))
        idx += pos_len

        dir = arr[idx : idx + dir_len].reshape((history_len, 4))
        idx += dir_len

        return BCHState(
            actions=actions,
            pos=pos,
            dir=dir,
        )

    def to_numpy(self):
        actions = self.actions.reshape(-1)
        pos = self.pos.reshape(-1)
        dir = self.dir.reshape(-1)
        data = jnp.concatenate([actions, pos, dir])
        return data

    @staticmethod
    def init_empty():
        return BCHState(
            actions=jnp.zeros((STATE_HISTORY_LEN,), dtype=jnp.int32),
            pos=jnp.zeros((STATE_HISTORY_LEN, 2), dtype=jnp.int32),
            dir=jnp.zeros((STATE_HISTORY_LEN, 4), dtype=jnp.int32),
        )

    @staticmethod
    def _extract_pos_dir_from_obs(obs):
        pos = obs[-2:]
        dir = obs[:4]
        return pos, dir

    def append(self, action, obs):
        pos, dir = self._extract_pos_dir_from_obs(obs)

        # Update actions
        actions = jnp.concatenate(
            [self.actions[1:], jnp.array([action], dtype=self.actions.dtype)]
        )
        # Update self state
        pos = jnp.concatenate([self.pos[1:], jnp.array([pos], dtype=self.pos.dtype)])
        dir = jnp.concatenate([self.dir[1:], jnp.array([dir], dtype=self.dir.dtype)])

        # Return updated state
        return BCHState(
            actions=actions,
            pos=pos,
            dir=dir,
        )

    def is_stuck(self, obs):
        pos, dir = self._extract_pos_dir_from_obs(obs)

        # Get the recent positions and directions
        pos_history = self.pos  # Shape: (stuck_time, 2)
        dir_history = self.dir  # Shape: (stuck_time, 4)

        # Check if all positions and directions are equal
        pos_equal = jnp.all(pos_history == pos)
        dir_equal = jnp.all(dir_history == dir)

        return pos_equal & dir_equal


class BCPolicy(AbstractPolicy):
    network: BCModel
    params: core.FrozenDict[str, Any]
    stochastic: bool = True
    unblock_if_stuck: bool = True

    def __init__(self, params, stochastic=True, unblock_if_stuck=True):
        network = BCModel(action_dim=len(Actions), hidden_dims=(64, 64))

        self.network = network
        self.params = params
        self.stochastic = stochastic
        self.unblock_if_stuck = unblock_if_stuck

    @classmethod
    def from_pretrained(cls, layout, split, run_id, stochastic=True):
        assert split in ["train", "test", "all"]

        _, params = load_bc_checkpoint(layout, split, run_id)

        return cls(params, stochastic=stochastic)

    def compute_action(self, obs, done, hstate, key):
        logits = self.network.apply({"params": self.params}, obs)
        action_probs = jax.nn.softmax(logits)

        if self.unblock_if_stuck:
            is_batched = hstate.shape[0] > 1

            def _handle_stuck(bc_hstate, obs, action_probs):
                new_action_probs = remove_indices_and_renormalize(
                    action_probs, bc_hstate.actions
                )

                is_stuck = bc_hstate.is_stuck(obs)
                return jnp.where(is_stuck, new_action_probs, action_probs)

            hstate = jnp.where(
                done[..., jnp.newaxis], self.init_hstate(hstate.shape[0]), hstate
            )

            if is_batched:
                bc_hstate = jax.vmap(BCHState.from_numpy)(hstate)
                action_probs = jax.vmap(_handle_stuck)(bc_hstate, obs, action_probs)
            else:
                bc_hstate = BCHState.from_numpy(hstate[0])
                action_probs = _handle_stuck(bc_hstate, obs, action_probs)

        if self.stochastic:

            def _pick_action(key, action_probs):
                return jax.random.choice(key, len(Actions), axis=-1, p=action_probs)

            if is_batched:
                keys = jax.random.split(key, action_probs.shape[0])
                action = jax.vmap(_pick_action)(keys, action_probs)
            else:
                action = _pick_action(key, action_probs)
        else:
            action = jnp.argmax(action_probs, axis=-1)

        if self.unblock_if_stuck:

            def _append_and_to_numpy(bc_hstate, action, obs):
                return bc_hstate.append(action, obs).to_numpy()

            if is_batched:
                hstate = jax.vmap(_append_and_to_numpy)(bc_hstate, action, obs)
            else:
                hstate = _append_and_to_numpy(bc_hstate, action, obs)
                hstate = hstate[jnp.newaxis, ...]

        return action, hstate, {}

    def init_hstate(self, batch_size, key=None) -> chex.Array:
        hstate = None

        if self.unblock_if_stuck:
            hstate = BCHState.init_empty().to_numpy()
            hstate = jnp.repeat(hstate[jnp.newaxis, ...], batch_size, axis=0)

        return hstate
