import argparse
from collections import defaultdict
from typing import List
import jaxmarl
import sys
import os
import itertools
import jax.numpy as jnp
import jax
import numpy as np
import copy
from datetime import datetime
from pathlib import Path
import chex
import imageio
import csv

from jaxmarl.environments.overcooked_v2.layouts import overcooked_v2_layouts
from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(DIR))
sys.path.append(os.path.dirname(os.path.dirname(DIR)))

from .policy import AbstractPolicy, PolicyPairing
from .rollout import get_rollout
from .utils import (
    get_recipe_identifier,
    make_eval_env,
    render_state_frame,
    resolve_eval_engine,
)
from overcooked_v2_experiments.ppo.utils.stablock import enumerate_reachable_positions


@chex.dataclass
class PolicyVizualization:
    frame_seq: chex.Array
    total_reward: chex.Scalar
    prediction_accuracy: chex.Array = None
    value_by_partner_pos: list = None


def visualize_pairing(
    output_dir: Path,
    policies: PolicyPairing,
    layout_name,
    key,
    env_kwargs={},
    num_seeds=1,
    all_recipes=False,
    no_viz=False,
    no_csv=False,
    algorithm="PPO",
    stablock_enabled=None,
    latent_analysis=False,
    value_analysis=False,
    ph1_forced_tilde_state=None,
    old_overcooked=False,
    disable_old_overcooked_auto=False,
):
    runs = eval_pairing(
        policies,
        layout_name,
        key,
        env_kwargs,
        num_seeds,
        all_recipes,
        no_viz,
        algorithm=algorithm,
        stablock_enabled=stablock_enabled,
        latent_analysis=latent_analysis,
        value_analysis=value_analysis,
        ph1_forced_tilde_state=ph1_forced_tilde_state,
        old_overcooked=old_overcooked,
        disable_old_overcooked_auto=disable_old_overcooked_auto,
    )

    reward_sum = 0.0
    rows = []
    for annotation, viz in runs.items():
        frame_seq = viz.frame_seq
        total_reward = viz.total_reward
        pred_acc = viz.prediction_accuracy

        if not no_viz:
            viz_dir = output_dir / "visualizations"
            os.makedirs(viz_dir, exist_ok=True)
            viz_filename = viz_dir / f"{annotation}.gif"

            imageio.mimsave(viz_filename, frame_seq, "GIF", duration=0.5)

        reward_sum += total_reward
        row = [annotation, total_reward]
        if pred_acc is not None:
            # pred_acc is (num_agents,)
            # Add to row
            for i in range(pred_acc.shape[0]):
                row.append(float(pred_acc[i]))
        
        rows.append(row)
        print(f"\t{annotation}:\t{total_reward}")
    reward_mean = reward_sum / len(runs)
    print(f"\tMean reward:\t{reward_mean}")

    if not no_csv:
        summery_name = "reward_summary.csv"
        summery_file = output_dir / summery_name
        with open(summery_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            fieldnames = ["annotation", "total_reward"]
            # Add prediction accuracy columns if available
            if rows and len(rows[0]) > 2:
                num_agents = len(rows[0]) - 2
                for i in range(num_agents):
                    fieldnames.append(f"pred_acc_agent_{i}")

            writer.writerow(fieldnames)

            for row in rows:
                writer.writerow(row)
        print(f"Summary written to {summery_file}")

    if (latent_analysis or value_analysis) and not no_csv:
        value_rows = []
        for annotation, viz in runs.items():
            if viz.value_by_partner_pos:
                for row in viz.value_by_partner_pos:
                    value_rows.append([annotation] + row)

        value_summary_name = "value_by_partner_pos.csv"
        value_summary_file = output_dir / value_summary_name
        with open(value_summary_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "annotation",
                    "e_t",
                    "partner_pos",
                    "mean_value",
                    "count",
                ]
            )
            for row in value_rows:
                writer.writerow(row)
        print(f"Value summary written to {value_summary_file}")


def eval_pairing(
    policies: PolicyPairing,
    layout_name,
    key,
    env_kwargs={},
    num_seeds=1,
    all_recipes=False,
    no_viz=False,
    algorithm="PPO",
    stablock_enabled=None,
    latent_analysis=False,
    value_analysis=False,
    ph1_forced_tilde_state=None,
    old_overcooked=False,
    disable_old_overcooked_auto=False,
):
    assert not (
        all_recipes and num_seeds is not None
    ), "Only one of all_recipes and num_seeds can be set"
    assert "layout" not in env_kwargs, "Layout should be passed as layout_name"

    value_rows_by_annotation = {}

    env_kwargs = copy.deepcopy(env_kwargs)
    engine_name, _resolved_probe_kwargs = resolve_eval_engine(
        layout_name,
        env_kwargs,
        old_overcooked=old_overcooked,
        disable_auto=disable_old_overcooked_auto,
    )

    if value_analysis:
        if engine_name == "overcooked":
            raise ValueError("value_analysis is only supported with overcooked_v2 engine")
        env, _engine_name, _resolved_kwargs = make_eval_env(
            layout_name,
            env_kwargs,
            old_overcooked=old_overcooked,
            disable_auto=disable_old_overcooked_auto,
        )

        key, key_init = jax.random.split(key)
        _, state = env.reset(key_init)

        # Choose target agent for value analysis: 0 for agent0, 1 for agent1
        target_agent = 0  # Change to 1 for agent1

        pos_y = state.agents.pos.y
        pos_x = state.agents.pos.x
        target_pos = jnp.array([pos_y[target_agent], pos_x[target_agent]], dtype=jnp.int32)

        coords = enumerate_reachable_positions(state.grid, target_pos)
        print(f"Value analysis reachable coords for agent {target_agent} (N={coords.shape[0]}):", coords)
        if coords.shape[0] == 0:
            coords = np.array([[-1, -1]], dtype=np.int32)

        annotations = [f"value-analysis-agent{target_agent}"]

        rollout = get_rollout(
            policies,
            env,
            key,
            algorithm=algorithm,
            stablock_enabled=stablock_enabled,
            ph1_forced_tilde_state=ph1_forced_tilde_state,
            value_by_et=True,
            et_candidates=coords,
            target_agent=target_agent,
        )

        # Add batch dimension for compatibility with downstream rendering
        rollouts = jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], rollout)

        # e_t가 주어지는 대상(파트너)의 좌표로 집계
        partner_idx = target_agent
        pos_seq = np.array(rollout.pos_seq)
        value_by_et_seq = np.array(rollout.value_by_et_seq)
        partner_pos_seq = pos_seq[:, partner_idx, :]

        value_rows_by_annotation = {}
        rows = []
        for et_i, coord in enumerate(coords):
            agg = defaultdict(list)
            for t in range(value_by_et_seq.shape[0]):
                y = int(partner_pos_seq[t, 0])
                x = int(partner_pos_seq[t, 1])
                agg[(y, x)].append(float(value_by_et_seq[t, et_i]))

            for (y, x), values in sorted(agg.items()):
                e_t_str = f"({int(coord[0])},{int(coord[1])})"
                partner_pos_str = f"({y},{x})"
                rows.append(
                    [
                        e_t_str,
                        partner_pos_str,
                        float(np.mean(values)),
                        len(values),
                    ]
                )

        value_rows_by_annotation[annotations[0]] = rows

    elif latent_analysis:
        if engine_name == "overcooked":
            raise ValueError("latent_analysis is only supported with overcooked_v2 engine")
        env, _engine_name, _resolved_kwargs = make_eval_env(
            layout_name,
            env_kwargs,
            old_overcooked=old_overcooked,
            disable_auto=disable_old_overcooked_auto,
        )

        key, key_init = jax.random.split(key)
        _, state = env.reset(key_init)

        # Choose target agent for latent analysis: 0 for agent0, 1 for agent1
        target_agent = 0  # Change to 1 for agent1

        pos_y = state.agents.pos.y
        pos_x = state.agents.pos.x
        target_pos = jnp.array([pos_y[target_agent], pos_x[target_agent]], dtype=jnp.int32)

        coords = enumerate_reachable_positions(state.grid, target_pos)
        print(f"Latent analysis reachable coords for agent {target_agent} (N={coords.shape[0]}):", coords)
        if coords.shape[0] == 0:
            coords = np.array([[-1, -1]], dtype=np.int32)

        keys = jax.random.split(key, coords.shape[0])
        rollouts_list = []
        annotations = []

        for i, coord in enumerate(coords):
            forced_blocked_states = [jnp.array([-1, -1], dtype=jnp.int32)] * env.num_agents
            forced_blocked_states[target_agent] = jnp.array(coord, dtype=jnp.int32)

            rollout = get_rollout(
                policies,
                env,
                keys[i],
                algorithm=algorithm,
                stablock_enabled=stablock_enabled,
                forced_blocked_states=forced_blocked_states,
                ph1_forced_tilde_state=ph1_forced_tilde_state,
            )
            rollouts_list.append(rollout)
            annotations.append(f"latent-agent{target_agent}-y{int(coord[0])}_x{int(coord[1])}")

        rollouts = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *rollouts_list)

        # e_t가 주어지는 대상(파트너)의 좌표로 집계
        partner_idx = target_agent
        value_rows_by_annotation = {}
        for i, coord in enumerate(coords):
            pos_seq = np.array(rollouts.pos_seq[i])
            val_seq = np.array(rollouts.value_seq[i, :, partner_idx])
            partner_pos_seq = pos_seq[:, partner_idx, :]

            agg = defaultdict(list)
            for t in range(val_seq.shape[0]):
                y = int(partner_pos_seq[t, 0])
                x = int(partner_pos_seq[t, 1])
                agg[(y, x)].append(float(val_seq[t]))

            rows = []
            for (y, x), values in sorted(agg.items()):
                e_t_str = f"({int(coord[0])},{int(coord[1])})"
                partner_pos_str = f"({y},{x})"
                rows.append(
                    [
                        e_t_str,
                        partner_pos_str,
                        float(np.mean(values)),
                        len(values),
                    ]
                )

            value_rows_by_annotation[annotations[i]] = rows

    elif all_recipes:
        if engine_name == "overcooked":
            raise ValueError("all_recipes evaluation is not supported with overcooked(v1) engine")
        layout = overcooked_v2_layouts[layout_name]
        env_kwargs.pop("layout")

        possible_recipes = jnp.array(layout.possible_recipes)

        def _rollout_recipe(recipe):
            _layout = copy.deepcopy(layout)
            _layout.possible_recipes = [recipe]
            env = OvercookedV2(layout=_layout, **env_kwargs)

            rollout = get_rollout(
                policies,
                env,
                key,
                algorithm=algorithm,
                stablock_enabled=stablock_enabled,
                ph1_forced_tilde_state=ph1_forced_tilde_state,
            )

            return rollout

        rollouts = jax.vmap(_rollout_recipe)(possible_recipes)
        annotations = [
            "recipe-" + get_recipe_identifier(r) for r in layout.possible_recipes
        ]

    else:
        env, engine_name, _resolved_kwargs = make_eval_env(
            layout_name,
            env_kwargs,
            old_overcooked=old_overcooked,
            disable_auto=disable_old_overcooked_auto,
        )

        def _rollout_seed_body(carry, key):
            rollout = get_rollout(
                policies,
                env,
                key,
                algorithm=algorithm,
                stablock_enabled=stablock_enabled,
                ph1_forced_tilde_state=ph1_forced_tilde_state,
            )
            return carry, rollout

        keys = jax.random.split(key, num_seeds)
        # lax.scan을 사용하여 롤아웃을 순차적으로 실행합니다.
        _, rollouts = jax.lax.scan(_rollout_seed_body, None, keys)
        annotations = [f"seed-{i}" for i in range(num_seeds)]


    if no_viz:
        frame_seqs = [None] * len(annotations)
    else:
        agent_view_size = env_kwargs.get("agent_view_size", None)
        
        # 완전히 순차적으로 처리하여 메모리 문제 해결
        frame_seqs = []
        for i in range(len(annotations)):
            # 각 시드의 state_seq를 개별적으로 렌더링
            state_seq_i = jax.tree_util.tree_map(lambda x: x[i], rollouts.state_seq)
            
            # render_sequence 대신 각 타임스텝을 하나씩 렌더링
            frames = []
            num_steps = jax.tree_util.tree_leaves(state_seq_i)[0].shape[0]
            for t in range(num_steps):
                state_t = jax.tree_util.tree_map(lambda x: x[t], state_seq_i)
                render_engine = "overcooked_v2" if all_recipes else engine_name
                frame = render_state_frame(state_t, render_engine, agent_view_size)
                # GPU 메모리에서 CPU로 즉시 전송하여 메모리 절약
                frames.append(np.array(frame))
            
            # NumPy로 스택하여 GPU 메모리 부담 감소
            frame_seq = np.stack(frames)
            frame_seqs.append(frame_seq)

    return {
        annotation: PolicyVizualization(
            frame_seq=frame_seqs[i],
            total_reward=rollouts.total_reward[i],
            prediction_accuracy=rollouts.prediction_accuracy[i] if rollouts.prediction_accuracy is not None else None,
            value_by_partner_pos=value_rows_by_annotation.get(annotation)
        )
        for i, annotation in enumerate(annotations)
    }
