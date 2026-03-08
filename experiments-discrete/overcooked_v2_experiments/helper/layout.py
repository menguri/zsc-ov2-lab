import argparse
from collections import defaultdict
import jaxmarl
import sys
import os
import itertools
import jax.numpy as jnp
import jax
import copy
from datetime import datetime
from pathlib import Path
import chex
import imageio
from jaxmarl.viz.overcooked_v2_visualizer import OvercookedV2Visualizer
import csv

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(DIR))
sys.path.append(os.path.dirname(os.path.dirname(DIR)))

from helper.rollout import get_rollout, PolicyRollout
from helper.store import store_model, load_model, load_all_checkpoints, PolicyCheckpoint
from jaxmarl.environments.overcooked_v2.layouts import overcooked_v2_layouts
from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2
from helper.rollout import get_rollout
from jaxmarl.viz.overcooked_v2_visualizer import OvercookedV2Visualizer
from helper.utils import get_recipe_identifier, mini_batch_pmap
from helper.plots import visualize_cross_play_matrix


def visualise_layout(layout: str):
    env = OvercookedV2(layout=layout)

    viz = OvercookedV2Visualizer()

    key_r = jax.random.PRNGKey(0)
    _, state = env.reset(key_r)

    state = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 0), state)

    viz = viz.render_sequence(state)

    img = jax.tree_util.tree_map(lambda x: x[0], viz)

    imageio.imsave(f"layout_viz/{layout}.png", img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--l", type=str, required=True)

    args = parser.parse_args()

    visualise_layout(args.l)
