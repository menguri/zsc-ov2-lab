import jax
import jax.numpy as jnp
from jaxmarl.environments.overcooked_v2.overcooked import (
    State,
    Agent,
    Position,
    DynamicObject,
    StaticObject,
)
from jaxmarl.environments.overcooked_v2.common import Direction
from jaxmarl.environments.overcooked_v2.layouts import overcooked_v2_layouts
from jaxmarl.environments.overcooked_v2.common import Actions

from overcooked_ai_py.mdp.actions import Direction as DirectionOld
from overcooked_ai_py.mdp.actions import Action as ActionOld

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState, ObjectState, SoupState


LAYOUT_SWAP_AGENT_DICT = {
    "cramped_room": True,
    "asymm_advantages": False,
    "coord_ring": False,
    "forced_coord": False,
    "counter_circuit": True,
}

MAP_ACTION = {
    ActionOld.STAY: Actions.stay,
    ActionOld.INTERACT: Actions.interact,
    DirectionOld.NORTH: Actions.up,
    DirectionOld.SOUTH: Actions.down,
    DirectionOld.EAST: Actions.right,
    DirectionOld.WEST: Actions.left,
}


def convert_action_to_overcooked_v2(action):
    return MAP_ACTION[action]


def convert_all_actions_to_overcooked_v2(actions, layout, perform_swap=False):
    if layout not in LAYOUT_SWAP_AGENT_DICT:
        raise ValueError(f"Unsupported layout {layout}")

    agent_swap = LAYOUT_SWAP_AGENT_DICT[layout] and perform_swap

    v2_actions = []
    for a in actions:
        action_v2 = convert_action_to_overcooked_v2(a)
        v2_actions.append(action_v2)

    if agent_swap:
        v2_actions = v2_actions[::-1]

    action_v2 = {
        f"agent_{i}": jnp.array(a, dtype=jnp.int32) for i, a in enumerate(v2_actions)
    }
    return action_v2


def convert_state_to_overcooked_v2(old_state, layout, perform_swap=False):
    """
    Converts an OvercookedState from the old environment to the new OvercookedV2 format.

    Args:
        old_state (OvercookedState): The old state object from the OvercookedGridworld.

    Returns:
        State: The new state object in OvercookedV2 format.
    """

    if layout not in LAYOUT_SWAP_AGENT_DICT:
        raise ValueError(f"Unsupported layout {layout}")

    agent_swap = LAYOUT_SWAP_AGENT_DICT[layout] and perform_swap

    layout = overcooked_v2_layouts[layout]
    static_objects = layout.static_objects
    pot_mask = static_objects[:, :] == StaticObject.POT

    def _obj_to_ing(obj):
        if obj is None:
            return DynamicObject.EMPTY, 0
        elif isinstance(obj, SoupState):
            ING_MAP = {
                "onion": 0,
                "tomato": 1,
            }

            ingredients = [ING_MAP[ing] for ing in obj.ingredients]
            encoded_ings = DynamicObject.get_recipe_encoding(jnp.array(ingredients))

            # print("obj", obj)
            pos = Position.from_tuple(obj.position)
            in_pot = pot_mask[pos.y, pos.x]

            if obj.is_ready:
                encoded_ings |= DynamicObject.COOKED

                if not in_pot:
                    encoded_ings |= DynamicObject.PLATE

            timer = 0
            if in_pot and obj.is_cooking:
                timer = obj.cook_time_remaining

                # print("timer", timer, obj)

            return encoded_ings, timer
        elif isinstance(obj, ObjectState):
            if obj.name == "onion":
                return DynamicObject.ingredient(0), 0
            elif obj.name == "tomato":
                return DynamicObject.ingredient(1), 0
            elif obj.name == "dish":
                return DynamicObject.PLATE, 0
            else:
                raise ValueError(f"Unknown object {obj.name}")
        else:
            raise ValueError(f"Unknown item {obj}, {type(obj)}")

    MAP_DIRECTION = {
        DirectionOld.NORTH: Direction.UP,
        DirectionOld.SOUTH: Direction.DOWN,
        DirectionOld.EAST: Direction.RIGHT,
        DirectionOld.WEST: Direction.LEFT,
    }

    agents = []
    for player in old_state.players:
        pos = Position.from_tuple(player.position)
        orientation = player.orientation
        new_dir = MAP_DIRECTION[orientation]

        obj = player.held_object
        inventory, _ = _obj_to_ing(obj)

        agent = Agent(
            pos=pos,
            dir=new_dir,
            inventory=inventory,
        )
        agents.append(agent)

    if agent_swap:
        assert len(agents) == 2
        agents = agents[::-1]

    agents = jax.tree_util.tree_map(lambda *x: jnp.array(x, dtype=jnp.int32), *agents)

    grid = jnp.stack(
        [
            static_objects,
            jnp.zeros_like(static_objects),  # ingredient channel
            jnp.zeros_like(static_objects),  # extra info channel
        ],
        axis=-1,
        dtype=jnp.int32,
    )

    for obj in old_state.objects.values():
        # print("obj1", obj)
        pos = Position.from_tuple(obj.position)
        ing, extra = _obj_to_ing(obj)

        grid = grid.at[pos.y, pos.x, 1:].set(jnp.array([ing, extra], dtype=jnp.int32))

    new_state = State(
        agents=agents,
        grid=grid,
        time=old_state.timestep,
        terminal=False,
        recipe=DynamicObject.get_recipe_encoding(jnp.array([0, 0, 0])),
    )

    return new_state
