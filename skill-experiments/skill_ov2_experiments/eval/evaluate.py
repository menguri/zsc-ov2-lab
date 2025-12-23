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
from jaxmarl.viz.overcooked_v2_visualizer import OvercookedV2Visualizer
import csv

from jaxmarl.environments.overcooked_v2.layouts import overcooked_v2_layouts
from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2
from jaxmarl.viz.overcooked_v2_visualizer import OvercookedV2Visualizer

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(DIR))
sys.path.append(os.path.dirname(os.path.dirname(DIR)))

from .policy import AbstractPolicy, PolicyPairing
from .rollout import get_rollout
from .rollout import get_rollout
from .utils import get_recipe_identifier


@chex.dataclass
class PolicyVizualization:
    frame_seq: chex.Array
    total_reward: chex.Scalar
    prediction_accuracy: chex.Array = None
    state_seq: chex.Array = None


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
):
    runs = eval_pairing(
        policies, layout_name, key, env_kwargs, num_seeds, all_recipes, no_viz, algorithm=algorithm
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


def eval_pairing(
    policies: PolicyPairing,
    layout_name,
    key,
    env_kwargs={},
    num_seeds=1,
    all_recipes=False,
    no_viz=False,
    algorithm="PPO",
):
    assert not (
        all_recipes and num_seeds > 1
    ), "Only one of all_recipes and num_seeds can be set"
    assert "layout" not in env_kwargs, "Layout should be passed as layout_name"

    if all_recipes:
        layout = overcooked_v2_layouts[layout_name]
        # env_kwargs.pop("layout") # [FIX] 이미 assert에서 체크했으므로 불필요하며 에러 유발

        possible_recipes = jnp.array(layout.possible_recipes)

        def _rollout_recipe(recipe):
            _layout = copy.deepcopy(layout)
            _layout.possible_recipes = [recipe]
            env = OvercookedV2(layout=_layout, **env_kwargs)

            rollout = get_rollout(policies, env, key, algorithm=algorithm)

            return rollout

        rollouts = jax.vmap(_rollout_recipe)(possible_recipes)
        annotations = [
            "recipe-" + get_recipe_identifier(r) for r in layout.possible_recipes
        ]

    else:
        env = OvercookedV2(layout=layout_name, **env_kwargs)

        def _rollout_seed_body(carry, key):
            rollout = get_rollout(policies, env, key, algorithm=algorithm)
            return carry, rollout

        keys = jax.random.split(key, num_seeds)
        # lax.scan을 사용하여 롤아웃을 순차적으로 실행합니다.
        _, rollouts = jax.lax.scan(_rollout_seed_body, None, keys)
        annotations = [f"seed-{i}" for i in range(num_seeds)]


    if no_viz:
        frame_seqs = [None] * len(annotations)
    else:
        agent_view_size = env_kwargs.get("agent_view_size", None)
        viz = OvercookedV2Visualizer()
        
        # 완전히 순차적으로 처리하여 메모리 문제 해결 -> JIT 컴파일 속도를 위해 vmap 사용
        frame_seqs = []
        # rollouts.state_seq의 첫 번째 leaf를 확인하여 batch size 결정
        batch_size = jax.tree_util.tree_leaves(rollouts.state_seq)[0].shape[0]
        
        for i in range(batch_size):
            # 각 시드의 state_seq를 개별적으로 추출
            state_seq_i = jax.tree_util.tree_map(lambda x: x[i], rollouts.state_seq)
            
            # vmap을 사용하여 타임스텝 차원에 대해 렌더링 (컴파일 속도 향상)
            frame_seq = jax.vmap(viz._render_state, in_axes=(0, None))(state_seq_i, agent_view_size)
            frame_seqs.append(frame_seq)

    return {
        annotation: PolicyVizualization(
            frame_seq=frame_seqs[i],
            total_reward=rollouts.total_reward[i],
            prediction_accuracy=rollouts.prediction_accuracy[i] if rollouts.prediction_accuracy is not None else None,
            state_seq=jax.tree_util.tree_map(lambda x: x[i], rollouts.state_seq)
        )
        for i, annotation in enumerate(annotations)
    }
