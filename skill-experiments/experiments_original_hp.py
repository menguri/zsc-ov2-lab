import os
import sys
import jax
import csv
import jax.numpy as jnp

from pathlib import Path
from dataclasses import dataclass
from typing import List

from jaxmarl.environments.overcooked_v2.overcooked import ObservationType

from skill_ov2_experiments.eval.evaluate import (
    PolicyPairing,
    visualize_pairing,
    eval_pairing,
)
from skill_ov2_experiments.human_rl.imitation.bc_policy import BCPolicy
from skill_ov2_experiments.ppo.utils.store import load_checkpoint
from skill_ov2_experiments.ppo.policy import PPOParams, PPOPolicy


OUTPUT_DIR = Path("experiments/run_2")
OUTPUT_DIR.mkdir(exist_ok=True)

PPO_BASE_DIR = Path("runs/")

NUM_PPO_SEEDS = 10


eval_key = jax.random.PRNGKey(0)


layouts = [
    "cramped_room",
    "asymm_advantages",
    "coord_ring",
    "forced_coord",
    "counter_circuit",
]

# run 1
# ppo_sp_policy_runs = {
#     "cramped_room": "20240916-132140_vtvsjdlz_cramped_room_avs-full",
#     "asymm_advantages": "20240916-133532_9pc68xan_asymm_advantages_avs-full",
#     "coord_ring": "20240916-141153_a8zdm81j_coord_ring_avs-full",
#     "forced_coord": "20240916-142817_ldg7mimj_forced_coord_avs-full",
#     "counter_circuit": "20240916-145647_jurmvs3d_counter_circuit_avs-full",
# }

# ppo_br_policy_runs = {
#     "cramped_room": "",
#     "asymm_advantages": "",
#     "coord_ring": "",
#     "forced_coord": "",
#     "counter_circuit": "",
# }

# ppo_augmented_sp_policy_runs = {
#     "cramped_room": "20240916-132207_zxk250yh_cramped_room_avs-full",
#     "asymm_advantages": "20240916-134031_ppai24fo_asymm_advantages_avs-full",
#     "coord_ring": "20240916-141131_hp3fh62k_coord_ring_avs-full",
#     "forced_coord": "20240916-143230_8gpx0cth_forced_coord_avs-full",
#     "counter_circuit": "20240916-145322_r8k0jxpq_counter_circuit_avs-full",
# }

# hp_indices = {
#     "cramped_room": 0,
#     "asymm_advantages": 0,
#     "coord_ring": 0,
#     "forced_coord": 0,
#     "counter_circuit": 0,
# }


# run 1


ppo_sp_policy_runs = {
    "cramped_room": "20240916-205217_42wc90x4_cramped_room_avs-full",
    "asymm_advantages": "20240916-212459_4w9i5fsp_asymm_advantages_avs-full",
    "coord_ring": "20240916-223218_zck1vum2_coord_ring_avs-full",
    "forced_coord": "20240916-232403_qvpfm4en_forced_coord_avs-full",
    "counter_circuit": "20240917-000057_nlyrez4l_counter_circuit_avs-full",
}

# ppo_br_policy_runs = {
#     "cramped_room": "",
#     "asymm_advantages": "",
#     "coord_ring": "",
#     "forced_coord": "",
#     "counter_circuit": "",
# }

ppo_augmented_sp_policy_runs = {
    "cramped_room": "20240916-210609_xzazybwi_cramped_room_avs-full",
    "asymm_advantages": "20240916-215114_l67bfq37_asymm_advantages_avs-full",
    "coord_ring": "20240916-224843_icub10er_coord_ring_avs-full",
    "forced_coord": "20240916-234007_y0f5g77z_forced_coord_avs-full",
    "counter_circuit": "20240917-002430_61cfo8a9_counter_circuit_avs-full",
}


hp_indices = {
    "cramped_room": 2,
    "asymm_advantages": 5,
    "coord_ring": 8,
    "forced_coord": 5,
    "counter_circuit": 0,
}

hp = {l: BCPolicy.from_pretrained(l, "all", 1) for l in layouts}


@dataclass
class PPOParamContainer:
    config: dict
    params: List[PPOParams]


def load_ppo_policies(run_name) -> PPOParamContainer:
    run_dir = PPO_BASE_DIR / run_name

    first_config = None
    all_params = []
    for run_id in range(NUM_PPO_SEEDS):
        config, params = load_checkpoint(run_dir, run_id, "final")

        if first_config is None:
            first_config = config

        ppo_params = PPOParams(params=params)
        all_params.append(ppo_params)

    container = PPOParamContainer(config=first_config, params=all_params)
    return container


def run_pairing(pairing, layout, num_seeds=500) -> List[float]:
    obs_type = []
    for policy in pairing:
        if isinstance(policy, BCPolicy):
            obs_type.append(ObservationType.FEATURIZED)
        else:
            obs_type.append(ObservationType.DEFAULT)

    res = eval_pairing(
        pairing,
        layout,
        eval_key,
        num_seeds=num_seeds,
        env_kwargs={"observation_type": obs_type},
        no_viz=True,
    )

    return jnp.array([r.total_reward for r in res.values()])


@dataclass
class EvalRow:
    policy1: str
    policy2: str
    returns: jnp.ndarray

    def print_summery(self):
        returns = jnp.array(self.returns)
        avg = returns.mean()
        std = returns.std()

        print(f"Paring {self.policy1} + {self.policy2}: avg={avg:.2f}, std={std:.2f}")


def store_result(rows: List[EvalRow], layout: str, pairing_name: str):
    filename = f"{pairing_name}.csv"
    dir = OUTPUT_DIR / layout
    dir.mkdir(exist_ok=True)
    filepath = dir / filename

    with open(filepath, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        fieldnames = ["policy1", "policy2", "total_reward"]

        writer.writerow(fieldnames)

        for row in rows:
            row.print_summery()
            for run_return in row.returns:
                writer.writerow([row.policy1, row.policy2, run_return])
    print(f"Stored results in {filepath}")


def eval_hp(hp_policy, layout, no_save=False):
    if not no_save:
        print(f"Evaluating HP policy on {layout}")

    pairing = PolicyPairing(hp_policy, hp_policy)

    returns = run_pairing(pairing, layout)

    eval_row = EvalRow("hp", "hp", returns)
    if no_save:
        eval_row.print_summery()
    else:
        store_result([eval_row], layout, "hp-hp")


def eval_ppo(ppo_policies, layout):
    print(f"Evaluating PPO SP policy on {layout}")
    config = ppo_policies.config

    @jax.jit
    def _eval(ppo_params):
        policy = PPOPolicy(ppo_params.params, config)
        pairing = PolicyPairing(policy, policy)
        returns = run_pairing(pairing, layout)
        return returns

    all_params = ppo_policies.params
    stacked_params = jax.tree_util.tree_map(lambda *v: jnp.stack(v), *all_params)

    all_returns = jax.vmap(_eval)(stacked_params)

    eval_rows = []
    for i in range(NUM_PPO_SEEDS):
        returns = jax.tree_util.tree_map(lambda x: x[i], all_returns)

        eval_row = EvalRow(f"ppo_{i}", f"ppo_{i}", returns)

        eval_rows.append(eval_row)

    store_result(eval_rows, layout, "ppo_sp-ppo_sp")


def eval_mixed(ppo_policies, hp_policy, layout, ppo_first, ppo_method):
    first_method_name = ppo_method if ppo_first else "hp"
    second_method_name = "hp" if ppo_first else ppo_method

    print(f"Evaluating {first_method_name} + {second_method_name} on {layout}")

    config = ppo_policies.config

    # @jax.jit
    def _eval(ppo_params):
        policy = PPOPolicy(ppo_params.params, config)
        if ppo_first:
            pairing = PolicyPairing(policy, hp_policy)
        else:
            pairing = PolicyPairing(hp_policy, policy)
        returns = run_pairing(pairing, layout)
        return returns

    all_params = ppo_policies.params
    stacked_params = jax.tree_util.tree_map(lambda *v: jnp.stack(v), *all_params)

    all_returns = jax.vmap(_eval)(stacked_params)

    eval_rows = []
    for i in range(NUM_PPO_SEEDS):
        returns = jax.tree_util.tree_map(lambda x: x[i], all_returns)

        first_name = f"ppo_{i}" if ppo_first else "hp"
        second_name = "hp" if ppo_first else f"ppo_{i}"

        eval_row = EvalRow(first_name, second_name, returns)

        eval_rows.append(eval_row)

    store_result(eval_rows, layout, f"{first_method_name}-{second_method_name}")


def evaluate_layout(layout):
    sp_policy_run = ppo_sp_policy_runs[layout]
    # br_policy_run = ppo_br_policy_runs[layout]
    augmented_sp_policy_run = ppo_augmented_sp_policy_runs[layout]

    ppo_sp_policies = load_ppo_policies(sp_policy_run)
    # br_params = load_ppo_params(br_policy_run)
    ppo_augmented_sp_params = load_ppo_policies(augmented_sp_policy_run)

    hp_policy = hp[layout]

    print("\n----------------------------------------------\n")
    eval_hp(hp_policy, layout)
    print("\n----------------------------------------------\n")
    eval_ppo(ppo_sp_policies, layout)
    print("\n----------------------------------------------\n")
    eval_mixed(ppo_sp_policies, hp_policy, layout, True, "ppo_sp")
    eval_mixed(ppo_sp_policies, hp_policy, layout, False, "ppo_sp")
    print("\n----------------------------------------------\n")
    eval_mixed(ppo_augmented_sp_params, hp_policy, layout, True, "ppo_augmented_sp")
    eval_mixed(ppo_augmented_sp_params, hp_policy, layout, False, "ppo_augmented_sp")
    print("\n----------------------------------------------\n")


# evaluate_layout("cramped_room")

for layout in layouts:
    evaluate_layout(layout)


# def evaluate_all_bc(layout):
#     hp = [BCPolicy.from_pretrained(layout, "all", i) for i in range(10)]

#     print(f"\n---------------------{layout}-------------------------\n")
#     for i, policy in enumerate(hp):
#         print(f"Seed {i}")
#         eval_hp(policy, layout, no_save=True)
#     print("\n----------------------------------------------\n")


# for layout in layouts:
#     evaluate_all_bc(layout)
