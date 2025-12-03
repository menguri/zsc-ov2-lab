from functools import partial
import itertools
from pathlib import Path
from omegaconf import OmegaConf
import wandb
import jax
import os
from datetime import datetime
import jax.numpy as jnp

from overcooked_v2_experiments.eval.policy import PolicyPairing
from overcooked_v2_experiments.eval.rollout import get_rollout
from overcooked_v2_experiments.human_rl.imitation.bc_policy import BCPolicy
from overcooked_v2_experiments.ppo.models.model import (
    get_actor_critic,
    initialize_carry,
)
from overcooked_v2_experiments.ppo.policy import (
    PPOParams,
    PPOPolicy,
    policy_checkoints_to_policy_pairing,
)
from overcooked_v2_experiments.ppo.utils.store import store_checkpoint
from overcooked_v2_experiments.ppo.utils.utils import (
    combine_first_two_tree_dim,
    get_num_devices,
    get_run_base_dir,
)
from overcooked_v2_experiments.ppo.utils.visualize_ppo import visualize_ppo_policy
from overcooked_v2_experiments.utils.utils import scanned_mini_batch_map

from .ippo import make_train
from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2, ObservationType
import copy
import jaxmarl

hp_indices = {
    "cramped_room": 2,
    "asymm_advantages": 5,
    "coord_ring": 8,
    "forced_coord": 5,
    "counter_circuit": 0,
}


def state_sample_run(config):
    print("Running state sample run")

    config = OmegaConf.to_container(config)

    num_seeds = config["NUM_SEEDS"]
    num_checkpoints = config["NUM_CHECKPOINTS"]
    num_iterations = config["NUM_ITERATIONS"]

    # assert num_checkpoints == 1
    assert num_iterations > 0

    model_name = config["model"]["TYPE"]
    layout_name = config["env"]["ENV_KWARGS"]["layout"]
    agent_view_size = config["env"]["ENV_KWARGS"].get("agent_view_size", None)
    avs_str = f"avs-{agent_view_size}" if agent_view_size is not None else "avs-full"
    run_name = f"sa_{layout_name}_{model_name}_{avs_str}"

    num_runs = num_seeds

    hp_policy = None
    if "BC" in config:
        print("Training with BC")
        split = "all"
        run_id = hp_indices[layout_name]
        print(f"Loading BC policy from {layout_name}-{split}-{run_id}")
        hp_policy = BCPolicy.from_pretrained(layout_name, split, run_id)

    with wandb.init(
        entity=config["wandb"]["ENTITY"],
        project=config["wandb"]["PROJECT"],
        tags=["IPPO", model_name, "OvercookedV2"],
        config=config,
        mode=config["wandb"]["WANDB_MODE"],
        name=run_name,
    ) as run:

        run_id = run.id
        run_base_dir = get_run_base_dir(run_id, config)
        config["RUN_BASE_DIR"] = run_base_dir
        print("Run base dir: ", run_base_dir)

        key = jax.random.PRNGKey(config["SEED"])

        def _run_iteration(
            key,
            state_buffer,
            prev_train_state=None,
            update_step_start=None,
            update_step_end=None,
        ):
            num_update_steps = update_step_end - update_step_start

            config_copy = copy.deepcopy(config)
            config_copy["env"]["ENV_KWARGS"]["initial_state_buffer"] = state_buffer

            keys = jax.random.split(key, num_seeds)
            train_jit = jax.jit(
                make_train(
                    config_copy,
                    update_step_offset=update_step_start,
                    update_step_num_overwrite=num_update_steps,
                )
            )

            # num_devices = len(jax.devices("gpu"))
            num_devices = get_num_devices()

            # if "gpu" in jax.devices()

            keys = keys.reshape((num_devices, -1, *keys.shape[1:]))
            if prev_train_state is not None:
                prev_train_state = jax.tree_util.tree_map(
                    lambda x: x.reshape((num_devices, -1, *x.shape[1:])),
                    prev_train_state,
                )

            # print("Keys shape: ", keys.shape)
            # print("Previous policies shape: ", previous_policies)

            ret = jax.pmap(jax.vmap(train_jit))(
                keys, initial_train_state=prev_train_state
            )

            return jax.tree_util.tree_map(lambda x: x.reshape((-1, *x.shape[2:])), ret)

        def _init_policy(key):
            env_config = config["env"]
            model_config = config["model"]
            env = jaxmarl.make(env_config["ENV_NAME"], **env_config["ENV_KWARGS"])

            network = get_actor_critic(config)

            init_x = (
                jnp.zeros(
                    (1, model_config["NUM_ENVS"], *env.observation_space().shape),
                ),
                jnp.zeros((1, model_config["NUM_ENVS"])),
            )
            init_hstate = initialize_carry(config, model_config["NUM_ENVS"])

            network_params = network.init(key, init_hstate, init_x)

            return network_params

        def _collect_states(previous_policies, key):
            start_time = datetime.now()

            def _process_combination_wrapper(
                num_rollouts=10,
                state_step_size=10,
                from_ppo_params=False,
                featurized_index=None,
            ):
                env_kwargs = copy.deepcopy(config["env"]["ENV_KWARGS"])

                if featurized_index is not None:
                    obs_type = [ObservationType.DEFAULT] * 2
                    obs_type[featurized_index] = ObservationType.FEATURIZED
                    env_kwargs["observation_type"] = obs_type

                env = OvercookedV2(**env_kwargs)

                def _process_combination(policies, key):
                    if from_ppo_params:
                        policies = policy_checkoints_to_policy_pairing(policies, config)

                    def _rollout_seed(key):
                        return get_rollout(policies, env, key)

                    keys = jax.random.split(key, num_rollouts)
                    rollouts = jax.vmap(_rollout_seed)(keys)

                    state_sequences = rollouts.state_seq
                    num_steps = env.max_steps

                    take_idxs = jnp.arange(0, num_steps, state_step_size)
                    # print("Take idxs: ", take_idxs)
                    sampled_state_sequences = jax.tree_util.tree_map(
                        lambda x: x[:, take_idxs], state_sequences
                    )
                    # print("Sampled state sequences: ", sampled_state_sequences)
                    flattened_state_sequences = combine_first_two_tree_dim(
                        sampled_state_sequences
                    )
                    # print("Flattened state sequences: ", flattened_state_sequences)

                    return flattened_state_sequences

                return _process_combination

            num_actors = 2
            if hp_policy is None:
                run_combinations = itertools.permutations(range(num_runs), num_actors)
                run_combinations = list(run_combinations)
                run_combinations += [[i] * num_actors for i in range(num_runs)]

                all_parings = []
                for run_combination in run_combinations:
                    run_combination = list(run_combination)

                    def _get_policy(run_num):
                        params = jax.tree_util.tree_map(
                            lambda x: x[run_num], previous_policies
                        )
                        return PPOParams(params=params)

                    policy_combination = PolicyPairing(
                        *[_get_policy(run_num) for run_num in run_combination]
                    )
                    all_parings.append(policy_combination)

                all_parings = jax.tree_util.tree_map(
                    lambda *v: jnp.stack(v), *all_parings
                )
                key, subkey = jax.random.split(key)

                comb_keys = jax.random.split(subkey, len(run_combinations))

                # def _ppo_policy_combination_wrapper(params, key):
                #     return __process_combination_wrapper(
                #         num_rollouts=10, state_step_size=10
                #     )(policies, key)

                # __process_combination_wrapper_jit = jax.jit(
                #     _ppo_policy_combination_wrapper
                # )
                state_collect_jit = jax.jit(
                    _process_combination_wrapper(
                        num_rollouts=10, state_step_size=10, from_ppo_params=True
                    )
                )

                state_buffer = scanned_mini_batch_map(state_collect_jit, 10)(
                    all_parings, comb_keys
                )
                state_buffer = combine_first_two_tree_dim(state_buffer)
            else:
                _state_collect_func_0 = _process_combination_wrapper(
                    num_rollouts=10, state_step_size=10, featurized_index=0
                )
                _state_collect_func_1 = _process_combination_wrapper(
                    num_rollouts=10, state_step_size=10, featurized_index=1
                )

                @jax.jit
                def _collect_bc_states(params, key):
                    ppo_policy = PPOPolicy(params, config)
                    ppo_hp_pairing = PolicyPairing(ppo_policy, hp_policy)
                    hp_ppo_pairing = PolicyPairing(hp_policy, ppo_policy)

                    hp_ppo_states = _state_collect_func_0(hp_ppo_pairing, key)
                    ppo_hp_states = _state_collect_func_1(ppo_hp_pairing, key)
                    return jax.tree_util.tree_map(
                        lambda x, y: jnp.concatenate([x, y], axis=0),
                        hp_ppo_states,
                        ppo_hp_states,
                    )

                key, subkey = jax.random.split(key)
                collect_keys = jax.random.split(subkey, num_seeds)
                state_buffer = jax.vmap(_collect_bc_states)(
                    previous_policies, collect_keys
                )
                state_buffer = combine_first_two_tree_dim(state_buffer)

            print(
                "State buffer shape: ",
                jax.tree_util.tree_map(lambda x: x.shape, state_buffer),
            )

            time_taken = datetime.now() - start_time
            print(f"Collecting states took {time_taken}")

            return state_buffer

        # MAIN TRAINING LOOP
        key, subkey = jax.random.split(key)
        policy_init_keys = jax.random.split(subkey, num_runs)
        previous_policies = jax.vmap(_init_policy)(policy_init_keys)

        model_config = config["model"]
        num_updates = (
            model_config["TOTAL_TIMESTEPS"]
            // model_config["NUM_STEPS"]
            // model_config["NUM_ENVS"]
        )
        update_step_offsets = jnp.linspace(
            0, num_updates, num_iterations + 1, dtype=jnp.int32
        )

        print("Update step offsets: ", update_step_offsets)

        prev_train_state = None

        for i in range(num_iterations):
            key, subkey = jax.random.split(key)
            state_buffer = _collect_states(previous_policies, subkey)
            # state_buffer = None

            update_step_start = update_step_offsets[i]
            update_step_end = update_step_offsets[i + 1]

            key, subkey = jax.random.split(key)
            out = _run_iteration(
                subkey,
                state_buffer,
                prev_train_state=prev_train_state,
                update_step_start=update_step_start,
                update_step_end=update_step_end,
            )
            prev_train_state = out["runner_state"][0]
            all_params = prev_train_state.params
            previous_policies = all_params

            final = i == num_iterations - 1
            # store checkpoints
            for run_num in range(num_runs):
                p = jax.tree_util.tree_map(lambda x: x[run_num], all_params)
                store_checkpoint(
                    config,
                    p,
                    run_num,
                    i,
                    final=final,
                )

        if config["VISUALIZE"]:
            visualize_ppo_policy(
                run_base_dir,
                key=jax.random.PRNGKey(config["SEED"]),
                final_only=True,
                num_seeds=2,
            )

            visualize_ppo_policy(
                run_base_dir,
                key=jax.random.PRNGKey(config["SEED"]),
                final_only=True,
                num_seeds=500,
                cross=True,
                no_viz=True,
            )
