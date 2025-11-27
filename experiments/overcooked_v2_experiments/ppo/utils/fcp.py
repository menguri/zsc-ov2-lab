from functools import partial
from overcooked_v2_experiments.eval.policy import AbstractPolicy
from overcooked_v2_experiments.ppo.policy import PPOParams, PPOPolicy
import chex
import jax.numpy as jnp
import jax


@chex.dataclass
class HStateWrapper:
    hstates: chex.Array
    params_idxs: chex.Scalar


class FCPWrapperPolicy(AbstractPolicy):
    population: PPOParams
    population_size: chex.Scalar

    policy: PPOPolicy

    def __init__(self, config, *params: PPOParams):
        stacked_params = jax.tree_util.tree_map(lambda *v: jnp.stack(v), *params)
        population_size = jax.tree_util.tree_leaves(stacked_params)[0].shape[0]

        policy = PPOPolicy(params=None, config=config)

        self.population = stacked_params
        self.population_size = population_size
        self.policy = policy

    def compute_action(self, obs, done, hstate, key):
        params_idxs, hstates = hstate.params_idxs, hstate.hstates

        batch_size = hstates.shape[0]

        key, subkey = jax.random.split(key)
        new_params_idxs = self._sample_param_idxs(batch_size, subkey)
        params_idxs = jnp.where(done, new_params_idxs, params_idxs)

        def _compute_action(policy_idx, obs, done, hstate, key):
            params = self._get_params(policy_idx)
            print("test:", obs.shape, done.shape, hstate.shape, key.shape, type(params))
            return self.policy.compute_action(obs, done, hstate, key, params=params)

        action_keys = jax.random.split(key, batch_size)
        actions, next_hstates = jax.vmap(_compute_action)(
            params_idxs, obs, done, hstates, action_keys
        )

        return actions, HStateWrapper(hstates=next_hstates, params_idxs=params_idxs)

    def init_hstate(self, batch_size, key):
        assert key is not None

        params_idxs = self._sample_param_idxs(batch_size, key)

        hstate_init_func = partial(self.policy.init_hstate, batch_size=1)
        policy_hstates = jax.vmap(hstate_init_func, axis_size=batch_size)()

        hstate = HStateWrapper(hstates=policy_hstates, params_idxs=params_idxs)
        return hstate

    def _sample_param_idxs(self, batch_size, key):
        params_idxs = jax.random.randint(key, (batch_size,), 0, self.population_size)
        return params_idxs

    def _get_params(self, policy_idx):
        params = jax.tree_util.tree_map(lambda x: x[policy_idx], self.population)
        return params.params


# from functools import partial
# from overcooked_v2_experiments.eval.policy import AbstractPolicy
# from overcooked_v2_experiments.ppo.policy import PPOParams, PPOPolicy
# from overcooked_v2_experiments.ppo.models.model import get_actor_critic, initialize_carry
# import chex
# import jax.numpy as jnp
# import jax
# from collections.abc import Mapping
# try:
#     from flax.core.frozen_dict import FrozenDict, unfreeze
# except Exception:  # Fallback for some Flax versions
#     from flax.core import FrozenDict
#     from flax.core import unfreeze


# @chex.dataclass
# class HStateWrapper:
#     hstates: chex.Array
#     params_idxs: chex.Scalar


# class FCPWrapperPolicy(AbstractPolicy):
#     population: PPOParams
#     population_size: chex.Scalar

#     policy: PPOPolicy
#     fcp_device: str

#     def __init__(self, policy_config, main_config, *params: PPOParams):
#         stacked_params = jax.tree_util.tree_map(lambda *v: jnp.stack(v), *params)
#         population_size = jax.tree_util.tree_leaves(stacked_params)[0].shape[0]

#         policy = PPOPolicy(params=None, config=policy_config)
#         network = get_actor_critic(policy_config)

#         self.fcp_device = main_config.get("FCP_DEVICE", "cpu")
#         print(f"[FCP] Storing population on device: {self.fcp_device}")
#         device = jax.devices(self.fcp_device)[0]

#         self.population = jax.tree_util.tree_map(lambda x: jax.device_put(x, device), stacked_params)
#         self.population_size = population_size
#         self.policy = policy
#         self.network = network

#     def compute_action(self, obs, done, hstate, key):
#         params_idxs, hstates = hstate.params_idxs, hstate.hstates

#         batch_size = hstates.shape[0]

#         key, subkey = jax.random.split(key)
#         new_params_idxs = self._sample_param_idxs(batch_size, subkey)
#         params_idxs = jnp.where(done, new_params_idxs, params_idxs)

#         def _compute_action(policy_idx, obs, done, hstate, key):
#             # Retrieve per-policy params placed on GPU.
#             params_tree = self._get_params(policy_idx)  # inner param tree (layer dict)

#             # Ensure plain dict for Flax variables collections (avoid nested FrozenDict wrappers)
#             if isinstance(params_tree, FrozenDict):
#                 params_tree = unfreeze(params_tree)
#             elif isinstance(params_tree, Mapping):
#                 # already a mapping; keep as-is
#                 params_tree = dict(params_tree)
#             # Build variables with exactly one params layer.
#             # If params_tree already looks like a variables dict {"params": ...}, pass it through.
#             if isinstance(params_tree, Mapping) and set(params_tree.keys()) == {"params"}:
#                 variables = dict(params_tree)
#             else:
#                 variables = {"params": params_tree}

#             # Match ippo apply contract shapes:
#             # - obs: (time=1, batch=1, H, W, C)
#             # - done: (time=1, batch=1)
#             # - hstate: (batch=1, hidden)
#             obs_in = obs[jnp.newaxis, jnp.newaxis, ...]
#             done_in = jnp.array([done])[jnp.newaxis, :]
#             hstate_in = hstate[jnp.newaxis, :]

#             ac_in = (obs_in, done_in)
#             # Reshape hstate to (1, hidden_size) for RNN input
#             hstate_reshaped = hstate_in.reshape(1, -1)
#             new_hstate, pi, _ = self.network.apply(variables, hstate_reshaped, ac_in)
#             new_hstate = new_hstate.squeeze(axis=0)
#             action = pi.sample(seed=key)
#             return action.squeeze(), new_hstate

#         action_keys = jax.random.split(key, batch_size)
#         actions, next_hstates = jax.vmap(_compute_action)(
#             params_idxs, obs, done, hstates, action_keys
#         )

#         return actions, HStateWrapper(hstates=next_hstates, params_idxs=params_idxs)

#     def init_hstate(self, batch_size, key):
#         assert key is not None

#         params_idxs = self._sample_param_idxs(batch_size, key)

#         hstate_init_func = partial(self.policy.init_hstate, batch_size=1)
#         policy_hstates = jax.vmap(hstate_init_func, axis_size=batch_size)()

#         hstate = HStateWrapper(hstates=policy_hstates, params_idxs=params_idxs)
#         return hstate

#     def _sample_param_idxs(self, batch_size, key):
#         params_idxs = jax.random.randint(key, (batch_size,), 0, self.population_size)
#         return params_idxs

#     def _get_params(self, policy_idx):
#         # 정책 풀에서 파라미터 선택
#         params = jax.tree_util.tree_map(lambda x: x[policy_idx], self.population)

#         # CPU에 저장된 경우 GPU로 전송
#         if self.fcp_device == "cpu":
#             params = jax.tree_util.tree_map(lambda x: jax.device_put(x, jax.devices("gpu")[0]), params)

#         # 중첩된 "params" 구조 해제
#         inner = params
#         if isinstance(inner, (FrozenDict, Mapping)) and "params" in inner:
#             inner = inner["params"]
        
#         # Flax apply 함수를 위해 최종적으로 {"params": ...} 형태로 래핑
#         if "params" not in inner:
#             inner = {"params": inner}
            
#         return inner

#     # def _get_params(self, policy_idx):
#     #     # population에서 하나 뽑기
#     #     params_cpu = jax.tree_util.tree_map(lambda x: x[policy_idx], self.population)

#     #     inner = params_cpu

#     #     # 1) PPOParams/TrainState 같이 .params 속성이 있다면 먼저 벗기기
#     #     if hasattr(inner, "params"):
#     #         inner = inner.params

#     #     # 2) dict/FrozenDict 구조라면, "params" 중첩을 한 번 더 벗길지 확인
#     #     if isinstance(inner, (FrozenDict, Mapping)):
#     #         # 여러 겹 "params" 가 있을 수 있으니 while 로 정리
#     #         while isinstance(inner, (FrozenDict, Mapping)) and "params" in inner \
#     #               and isinstance(inner["params"], (FrozenDict, Mapping)):
#     #             inner = inner["params"]

#     #         # 여기서 inner 는 CNN/LSTM weight들이 들어있는 "leaf dict" 이거나
#     #         # 이미 {"params": ...}일 수 있음
#     #         # 우리가 Flax variables로 쓰고 싶은 건: {"params": <leaf dict>}
#     #         if "params" not in inner:
#     #             inner = {"params": inner}

#     #     # GPU로 옮기기
#     #     params_gpu = jax.tree_util.tree_map(
#     #         lambda x: jax.device_put(x, jax.devices("gpu")[0]), inner
#     #     )
#     #     return params_gpu  # 반드시 {"params": ...} 꼴
