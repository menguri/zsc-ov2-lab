import argparse
from typing import List
import jax
import jax.numpy as jnp
from jaxmarl.environments.overcooked_v2.common import Actions
from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2
from jaxmarl.environments.overcooked_v2.layouts import overcooked_v2_layouts as layouts
from jaxmarl.viz.overcooked_v2_visualizer import OvercookedV2Visualizer
from .store import load_all_checkpoints
from .rollout import init_rollout
import copy


class InteractiveOvercookedV2:

    def __init__(self, run_base_dir, policy_idxs: List[int], no_jit=False, debug=False):
        self.debug = debug
        self.no_jit = no_jit

        all_params, config = load_all_checkpoints(run_base_dir, final_only=True)

        env_kwargs = copy.deepcopy(config["env"]["ENV_KWARGS"])
        self.env = OvercookedV2(**env_kwargs)

        num_actors = self.env.num_agents

        assert (
            len(policy_idxs) == num_actors
        ), f"Expected {num_actors} policies, got {policy_idxs}"

        policies = []
        for i, policy_idx in enumerate(policy_idxs):
            if policy_idx == -1:
                # use policy 0 and we overwrite it later with the human action
                policy_idx = 0
                self.human_policy_idx = i

            if f"run_{policy_idx}" not in all_params:
                raise ValueError(f"Policy idx {policy_idx} not found in checkpoints")

            policy = all_params[f"run_{policy_idx}"]["ckpt_final"]
            policies.append(policy)

        self.policies = jax.tree_util.tree_map(lambda *v: jnp.stack(v), *policies)

        self.init_hstate, self._get_actions = init_rollout(self.policies, self.env)

        self.viz = OvercookedV2Visualizer()

    def run(self, key):
        self.key = key
        with jax.disable_jit(self.no_jit):
            self._run()

    def _run(self):
        self._reset()

        self.viz.window.reg_key_handler(self._handle_input)
        self.viz.show(block=True)

    def _handle_input(self, event):
        if self.debug:
            print("Pressed", event.key)

        ACTION_MAPPING = {
            "left": Actions.left,
            "right": Actions.right,
            "up": Actions.up,
            "down": Actions.down,
            " ": Actions.interact,
            "tab": Actions.stay,
        }

        match event.key:
            case "escape":
                self.viz.window.close()
                return
            case "backspace":
                self._reset()
                return
            case key if key in ACTION_MAPPING:
                action = ACTION_MAPPING[key]
            case key:
                print(f"Key {key} not recognized")
                return

        self._step(action)

    def _redraw(self):
        self.viz.render(self.state, agent_view_size=self.env.agent_view_size)

    def _reset(self):
        self.key, subkey = jax.random.split(self.key)

        self.obs, self.state = jax.jit(self.env.reset)(subkey)

        self.done = jnp.zeros(self.env.num_agents, dtype=bool)

        self.hstate = self.init_hstate

        self._redraw()

    def _step(self, action):
        self.key, subkey = jax.random.split(self.key)

        print(f"Human action (agent {self.human_policy_idx}): ", action)

        actions, self.hstate = self._get_actions(
            self.obs, self.done, self.hstate, subkey
        )

        actions[f"agent_{self.human_policy_idx}"] = jnp.array(action)

        print("Actions: ", actions)

        self.obs, self.state, reward, done, info = jax.jit(self.env.step_env)(
            subkey, self.state, actions
        )
        self.done = jnp.array([done[a] for a in self.env.agents])

        print(
            f"t={self.state.time}: reward={reward['agent_0']}, done = {done['__all__']}, shaped_reward = {info['shaped_reward']}"
        )

        if done["__all__"]:
            self._reset()
        else:
            self._redraw()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=str, help="run directory", required=True)
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to generate the environment with",
        default=0,
    )
    parser.add_argument(
        "--no_jit",
        default=False,
        help="Disable JIT compilation",
        action="store_true",
    )
    parser.add_argument(
        "--debug", default=False, help="Debug mode", action="store_true"
    )
    parser.add_argument(
        "--policy_idxs",
        type=int,
        nargs="+",
        help="Policy idxs to use. Use -1 for human",
        required=True,
    )
    args = parser.parse_args()

    if len(args.layout) == 0:
        raise ValueError("You must provide a layout.")

    interactive = InteractiveOvercookedV2(
        args.d, args.policy_idxs, no_jit=args.no_jit, debug=args.debug
    )

    key = jax.random.PRNGKey(args.seed)
    interactive.run(key)
