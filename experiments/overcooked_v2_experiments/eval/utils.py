import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import jax
import numpy as np
import jax.numpy as jnp
import jaxmarl
from jaxmarl.environments.overcooked.layouts import overcooked_layouts
from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer
from jaxmarl.viz.overcooked_v2_visualizer import OvercookedV2Visualizer


def get_recipe_identifier(ingredients: List[int]) -> int:
    """
    Get the identifier for a recipe given the ingredients.
    """
    return f"{ingredients[0]}_{ingredients[1]}_{ingredients[2]}"


def resolve_old_overcooked_flags(config: Dict[str, Any]) -> Tuple[bool, bool]:
    old_overcooked = bool(config.get("OLD_OVERCOOKED", False))
    disable_auto = bool(config.get("DISABLE_OLD_OVERCOOKED_AUTO", False))
    if "alg" in config and isinstance(config["alg"], dict):
        old_overcooked = old_overcooked or bool(
            config["alg"].get("OLD_OVERCOOKED", False)
        )
        disable_auto = disable_auto or bool(
            config["alg"].get("DISABLE_OLD_OVERCOOKED_AUTO", False)
        )
    return old_overcooked, disable_auto


def resolve_eval_engine(
    layout: Any,
    env_kwargs: Dict[str, Any],
    old_overcooked: bool = False,
    disable_auto: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    kwargs = dict(env_kwargs)
    kwargs["layout"] = layout

    layout_name = layout if isinstance(layout, str) else None
    auto_old = (
        (not disable_auto)
        and isinstance(layout_name, str)
        and (layout_name in overcooked_layouts)
    )
    use_old = bool(old_overcooked or auto_old)

    if use_old:
        allowed = {"layout", "random_reset", "max_steps"}
        kwargs = {k: v for k, v in kwargs.items() if k in allowed}
        if isinstance(kwargs.get("layout"), str):
            layout_name = kwargs["layout"]
            if layout_name not in overcooked_layouts:
                raise ValueError(
                    f"Unknown overcooked(v1) layout '{layout_name}'. "
                    f"Available: {sorted(overcooked_layouts.keys())}"
                )
            kwargs["layout"] = overcooked_layouts[layout_name]
        return "overcooked", kwargs

    return "overcooked_v2", kwargs


def make_eval_env(
    layout: Any,
    env_kwargs: Dict[str, Any],
    old_overcooked: bool = False,
    disable_auto: bool = False,
):
    env_name, resolved_kwargs = resolve_eval_engine(
        layout=layout,
        env_kwargs=env_kwargs,
        old_overcooked=old_overcooked,
        disable_auto=disable_auto,
    )
    env = jaxmarl.make(env_name, **resolved_kwargs)
    return env, env_name, resolved_kwargs


def extract_global_full_obs(env, state, env_name: str):
    if env_name == "overcooked":
        obs = env.get_obs(state)
        return obs[env.agents[0]].astype(jnp.float32)
    return env.get_obs_default(state)[0].astype(jnp.float32)


def extract_pos_yx(state, env_name: str):
    if env_name == "overcooked":
        pos_x = state.agent_pos[:, 0]
        pos_y = state.agent_pos[:, 1]
        return pos_y, pos_x
    return state.agents.pos.y, state.agents.pos.x


def render_state_frame(state, env_name: str, agent_view_size: Optional[int] = None):
    if env_name == "overcooked":
        # For classic overcooked(v1), always render full-view equivalent.
        # Ignore OV2 partial-view configs (e.g. agent_view_size=2).
        avs = 5
        padding = max(1, avs - 2)
        h, w = int(state.maze_map.shape[0]), int(state.maze_map.shape[1])
        if (2 * padding) >= h or (2 * padding) >= w:
            padding = 1
        grid = np.asarray(state.maze_map[padding : h - padding, padding : w - padding, :])
        return OvercookedVisualizer._render_grid(
            grid,
            highlight_mask=None,
            agent_dir_idx=state.agent_dir_idx,
            agent_inv=state.agent_inv,
        )

    viz = OvercookedV2Visualizer()
    return np.asarray(viz._render_state(state, agent_view_size))
