import jax
import jax.numpy as jnp
import numpy as np

from jaxmarl.environments.overcooked_v2.common import StaticObject
from jaxmarl.environments.overcooked_v2.utils import compute_enclosed_spaces


def _sample_from_mask(rng, mask):
    flat = mask.reshape(-1)
    probs = flat.astype(jnp.float32)
    probs = probs / jnp.maximum(probs.sum(), 1.0)
    idx = jax.random.choice(rng, flat.size, p=probs)
    width = mask.shape[1]
    y = idx // width
    x = idx % width
    return jnp.array([y, x], dtype=jnp.int32)


def sample_blocked_state(rng, grid, partner_pos, no_block_prob=None):
    """partner가 도달 가능한 빈 타일에서 차단 좌표를 샘플링한다. [-1, -1]도 후보에 포함. 특정 맵에서는 그룹 필터링 적용.
    no_block_prob가 주어지면 [-1, -1] 선택 확률을 그 값으로 고정한다.
    """
    empty_mask = grid[..., 0] == StaticObject.EMPTY
    height, width = empty_mask.shape

    py, px = partner_pos
    in_bounds = (py >= 0) & (py < height) & (px >= 0) & (px < width)

    enclosed = compute_enclosed_spaces(empty_mask)

    def _get_partner_id():
        return enclosed[py, px]

    partner_id = jax.lax.cond(in_bounds, _get_partner_id, lambda: jnp.int32(-1))
    reachable_mask = empty_mask & (enclosed == partner_id)

    # ground-coord-simple 전용 ==============================================
    # # 그룹 필터링 적용 (grounded_coord_simple 맵만)
    # groups_left = jnp.array([[1,1], [1,2], [2,1], [2,2], [3,1], [3,2]], dtype=jnp.int32)
    # groups_right = jnp.array([[1,5], [3,5], [-1,-1], [-1,-1], [-1,-1], [-1,-1]], dtype=jnp.int32)  # 패딩으로 shape 맞춤
    # group_coords = jax.lax.cond(px <= 2, lambda: groups_left, lambda: groups_right)
    # # 그룹 좌표에 대한 마스크 생성
    # group_mask = jnp.zeros_like(empty_mask, dtype=bool)
    # ys, xs = group_coords[:, 0], group_coords[:, 1]
    # valid = (ys >= 0) & (xs >= 0)  # 유효 좌표만 True
    # group_mask = group_mask.at[ys, xs].set(valid)
    # # reachable_mask와 그룹 마스크 교집합
    # reachable_mask = reachable_mask & group_mask
    # ground-coord-simple 전용 ==============================================

    use_mask = jnp.where(jnp.any(reachable_mask), reachable_mask, empty_mask)
    has_any = jnp.any(use_mask)
    count = jnp.sum(use_mask)

    def _pick(key):
        key_pick, key_gate = jax.random.split(key)
        prob_no_block = jnp.where(
            no_block_prob is None,
            1.0 / (count + 1.0),
            jnp.clip(no_block_prob, 0.0, 1.0),
        )
        choose_no_block = jax.random.uniform(key_gate) < prob_no_block
        return jax.lax.cond(
            choose_no_block,
            lambda _k: jnp.array([-1, -1], dtype=jnp.int32),
            lambda _k: _sample_from_mask(_k, use_mask),
            key_pick,
        )

    def _fallback(_key):
        return jnp.array([-1, -1], dtype=jnp.int32)

    return jax.lax.cond(has_any, _pick, _fallback, rng)


def enumerate_reachable_positions(grid, partner_pos):
    """partner가 도달 가능한 빈 타일 좌표 목록을 반환한다. (N, 2)"""
    empty_mask = grid[..., 0] == StaticObject.EMPTY
    height, width = empty_mask.shape

    py, px = partner_pos
    in_bounds = (py >= 0) & (py < height) & (px >= 0) & (px < width)
    if not bool(in_bounds):
        return np.zeros((0, 2), dtype=np.int32)

    enclosed = compute_enclosed_spaces(empty_mask)
    partner_id = enclosed[py, px]
    reachable_mask = empty_mask & (enclosed == partner_id)

    coords = np.argwhere(np.asarray(reachable_mask))
    return coords.astype(np.int32)


def initialize_blocked_states(rng, env_state, enabled, num_envs, partner_pos, no_block_prob=None):
    """에피소드 시작 시 blocked_states_env 초기화."""
    if not enabled:
        return jnp.full((num_envs, 2), -1, dtype=jnp.int32)

    rng, rng_blocked = jax.random.split(rng)
    rng_blocked = jax.random.split(rng_blocked, num_envs)
    blocked_states_env = jax.vmap(lambda r, g, p: sample_blocked_state(r, g, p, no_block_prob))(
        rng_blocked, env_state.env_state.grid, partner_pos
    )
    return blocked_states_env


def resample_blocked_states(
    rng, env_state, episode_done, blocked_states_env, enabled, partner_pos, no_block_prob=None
):
    """에피소드 종료된 환경만 차단 좌표를 재샘플링."""
    if not enabled:
        return blocked_states_env

    rng, rng_blocked = jax.random.split(rng)
    rng_blocked = jax.random.split(rng_blocked, blocked_states_env.shape[0])
    sampled_blocked_env = jax.vmap(lambda r, g, p: sample_blocked_state(r, g, p, no_block_prob))(
        rng_blocked, env_state.env_state.grid, partner_pos
    )
    blocked_states_env = jnp.where(
        episode_done[:, None], sampled_blocked_env, blocked_states_env
    )
    return blocked_states_env


def expand_blocked_states(blocked_states_env, num_agents, is_ego):
    """env 단위 blocked_states를 actor 단위로 확장하고 ego는 마스킹."""
    blocked_states_actor = jnp.tile(blocked_states_env, (num_agents, 1))
    blocked_states_actor = jnp.where(
        is_ego[:, None],
        jnp.array([-1, -1], dtype=jnp.int32),
        blocked_states_actor,
    )
    return blocked_states_actor
