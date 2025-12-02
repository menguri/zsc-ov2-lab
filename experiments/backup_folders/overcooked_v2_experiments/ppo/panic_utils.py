"""
panic_utils.py

에피소드 단위 PANIC(액션 교란) 관련 유틸 함수 모음.

설계 개요:
- select_panic_partner_indices: 각 벡터화된 환경마다 교란 대상 에이전트 인덱스를 선택.
  * 2인 환경(OvercookedV2 self-play) 기준으로 Bernoulli(0.5) → {0,1} 균등.
  * 활성 아니면 -1 채움 (교란 비활성 표식).
- apply_panic_action_override: 활성 마스크(panic_active_mask)가 True인 환경에서 선택된 에이전트의 액션을
  균일 난수로 덮어쓰기. 덮어쓴 후 평탄화된 action 벡터 반환.

주의:
- panic_partner_indices 가 -1이면 덮어쓰지 않음.
- PPO 일관성을 위해 호출 측(ippo.py)에서 log_prob를 교란 후 재계산.
"""
from typing import Tuple, Dict
import jax
import jax.numpy as jnp


def select_panic_partner_indices(
    rng: jax.Array,
    num_envs: int,
    num_agents: int,
    panic_enabled: bool,
    panic_duration: int,
) -> jnp.ndarray:
    """환경별 PANIC 대상 에이전트 인덱스 선택.

    반환:
        shape (num_envs,) int32
        - 활성 아니면 모두 -1
        - 2인 환경이면 Bernoulli(0.5)로 0 또는 1
        - 그 외 에이전트 수는 균일 난수 선택(향후 확장 대비)
    """
    if not (panic_enabled and panic_duration > 0):
        return jnp.full((num_envs,), -1, dtype=jnp.int32)
    if num_agents == 2:
        bern = jax.random.bernoulli(rng, 0.5, (num_envs,))  # bool -> {False,True}
        return bern.astype(jnp.int32)  # {0,1}
    # fallback: uniform over all agents
    return jax.random.randint(rng, (num_envs,), 0, num_agents, dtype=jnp.int32)


def apply_panic_action_override(
    rng: jax.Array,
    actions_2d: jnp.ndarray,
    panic_active_mask: jnp.ndarray,
    panic_partner_indices: jnp.ndarray,
    num_actions: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """선택된 에이전트 액션을 균일 난수로 덮어쓰기.

    매개변수:
        actions_2d: shape (num_agents, num_envs)
        panic_active_mask: shape (num_envs,) 교란 활성 여부
        panic_partner_indices: shape (num_envs,) 각 환경 대상 인덱스 (또는 -1 비활성)
        num_actions: Discrete action space 크기

    반환:
        (updated_actions_2d, flat_actions)
    """
    num_agents, num_envs = actions_2d.shape
    random_actions = jax.random.randint(rng, (num_envs,), 0, num_actions)

    # 유효한 환경 (인덱스 >=0) & 활성 마스크
    valid_mask = panic_partner_indices >= 0
    override_mask = panic_active_mask & valid_mask  # shape (num_envs,)

    # one-hot 구축 (비활성 환경은 덮어쓰지 않도록 override_mask로 gating)
    safe_indices = jnp.where(valid_mask, panic_partner_indices, 0)
    one_hot = jax.nn.one_hot(safe_indices, num_agents, dtype=jnp.bool_)  # (num_envs, num_agents)

    # env-first view로 변환
    actions_env_major = actions_2d.T  # (num_envs, num_agents)

    # 교란 적용: (override_mask[:,None] & one_hot) 위치를 random_actions로 치환
    replacement = random_actions[:, None]
    mask_matrix = override_mask[:, None] & one_hot
    new_actions_env_major = jnp.where(mask_matrix, replacement, actions_env_major)

    updated_actions_2d = new_actions_env_major.T
    flat_actions = updated_actions_2d.reshape(num_agents * num_envs)
    return updated_actions_2d, flat_actions

__all__ = [
    "select_panic_partner_indices",
    "apply_panic_action_override",
    "accumulate_panic_step",
    "finalize_panic_episode",
    "aggregate_panic_metrics",
]


def accumulate_panic_step(
    panic_enabled: bool,                # PANIC 기능 활성 여부
    panic_duration: int,                # PANIC 지속 시간 (에피소드 로컬 스텝 단위)
    panic_active_mask: jnp.ndarray,     # shape (num_envs,) 현재 스텝에서 PANIC 창 내에 있는 환경들
    panic_partner_indices: jnp.ndarray, # shape (num_envs,) 각 환경 교란 대상 에이전트 인덱스 (-1 비활성)
    original_reward_first_agent: jnp.ndarray,  # shape (num_envs,) 첫 번째 에이전트의 원시 보상 (팀 공유 가정)
    panic_action_counts_ep: jnp.ndarray,       # shape (num_envs,) 에피소드 내 교란된 스텝 누적
    panic_reward_accum_ep: jnp.ndarray,        # shape (num_envs,) 에피소드 내 panic 창 구간 보상 누적
    panic_delivery_accum_ep: jnp.ndarray,      # shape (num_envs,) 에피소드 내 올바른 배달 횟수 누적
    panic_wrong_delivery_accum_ep: jnp.ndarray,# shape (num_envs,) 에피소드 내 잘못된 배달 횟수 누적
    delivery_reward_value: float,              # 올바른 배달 시 발생하는 DELIVERY_REWARD 상수값
    negative_delivery_reward_value: float,     # 잘못된 배달 시 발생하는 음수 리워드 (없는 경우 0)
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """한 스텝에서 panic 관련 per-episode 누적값 갱신.

    반환: (new_action_counts_ep, new_reward_accum_ep, new_delivery_accum_ep, new_wrong_delivery_accum_ep)
    """
    # PANIC 비활성 또는 duration <=0 인 경우 그대로 반환 (no-op)
    if not (panic_enabled and panic_duration > 0):
        return (
            panic_action_counts_ep,
            panic_reward_accum_ep,
            panic_delivery_accum_ep,
            panic_wrong_delivery_accum_ep,
        )
    # 교란된 환경 마스크: 활성 구간 & 대상 인덱스 존재
    override_env_mask = panic_active_mask & (panic_partner_indices >= 0)
    # 교란 스텝 카운트 증가
    new_action_counts_ep = panic_action_counts_ep + override_env_mask.astype(jnp.int32)
    # 팀 보상 누적 (panic 창 내부만)
    step_team_reward = original_reward_first_agent
    new_reward_accum_ep = panic_reward_accum_ep + jnp.where(
        panic_active_mask, step_team_reward, 0.0
    )
    # 올바른 배달: reward == +DELIVERY_REWARD
    delivery_mask = panic_active_mask & (original_reward_first_agent == delivery_reward_value)
    new_delivery_accum_ep = panic_delivery_accum_ep + delivery_mask.astype(jnp.int32)
    # 잘못된 배달: reward == -DELIVERY_REWARD (negative_rewards 활성 시)
    wrong_delivery_mask = panic_active_mask & (
        original_reward_first_agent == negative_delivery_reward_value
    )
    new_wrong_delivery_accum_ep = panic_wrong_delivery_accum_ep + wrong_delivery_mask.astype(jnp.int32)
    return (
        new_action_counts_ep,
        new_reward_accum_ep,
        new_delivery_accum_ep,
        new_wrong_delivery_accum_ep,
    )


def finalize_panic_episode(
    ended_mask: jnp.ndarray,                # shape (num_envs,) 에피소드 종료 여부
    panic_action_counts_ep: jnp.ndarray,    # per-episode 교란 스텝 카운트
    panic_reward_accum_ep: jnp.ndarray,     # per-episode panic reward 누적
    panic_delivery_accum_ep: jnp.ndarray,   # per-episode 올바른 배달 카운트
    panic_wrong_delivery_accum_ep: jnp.ndarray, # per-episode 잘못된 배달 카운트
    total_panic_actions: jnp.ndarray,       # 종료된 에피소드들의 교란 스텝 총합
    total_panic_reward: jnp.ndarray,        # 종료된 에피소드들의 panic reward 총합
    total_panic_deliveries: jnp.ndarray,    # 종료된 에피소드들의 올바른 배달 총합
    total_panic_wrong_deliveries: jnp.ndarray, # 종료된 에피소드들의 잘못된 배달 총합
    episodes_completed: jnp.ndarray,        # 종료된 에피소드 수
) -> Tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    """에피소드 종료 시 총계에 반영하고 per-episode 누적을 리셋."""
    total_panic_actions = total_panic_actions + jnp.where(ended_mask, panic_action_counts_ep, 0)
    total_panic_reward = total_panic_reward + jnp.where(ended_mask, panic_reward_accum_ep, 0.0)
    total_panic_deliveries = total_panic_deliveries + jnp.where(ended_mask, panic_delivery_accum_ep, 0)
    total_panic_wrong_deliveries = total_panic_wrong_deliveries + jnp.where(
        ended_mask, panic_wrong_delivery_accum_ep, 0
    )
    episodes_completed = episodes_completed + ended_mask.astype(jnp.int32)
    new_counts = jnp.where(ended_mask, 0, panic_action_counts_ep)
    new_reward = jnp.where(ended_mask, 0.0, panic_reward_accum_ep)
    new_deliveries = jnp.where(ended_mask, 0, panic_delivery_accum_ep)
    new_wrong_deliveries = jnp.where(ended_mask, 0, panic_wrong_delivery_accum_ep)
    return (
        new_counts,
        new_reward,
        new_deliveries,
        new_wrong_deliveries,
        total_panic_actions,
        total_panic_reward,
        total_panic_deliveries,
        total_panic_wrong_deliveries,
        episodes_completed,
    )


def aggregate_panic_metrics(
    panic_enabled: bool,                   # PANIC 활성 여부
    panic_duration: int,                   # 지속 시간
    total_panic_actions: jnp.ndarray,      # 누적 교란 스텝 수 (환경별)
    total_panic_reward: jnp.ndarray,       # 누적 panic reward (환경별)
    total_panic_deliveries: jnp.ndarray,   # 누적 올바른 배달 수 (환경별)
    total_panic_wrong_deliveries: jnp.ndarray, # 누적 잘못된 배달 수 (환경별)
    episodes_completed: jnp.ndarray,       # 종료된 에피소드 수 (환경별)
) -> Dict[str, jnp.ndarray]:
    """업데이트 단계에서 wandb 로깅용 aggregate 메트릭 계산."""
    if not (panic_enabled and panic_duration > 0):
        return {}
    total_actions = total_panic_actions.sum()
    total_reward = total_panic_reward.sum()
    total_deliveries = total_panic_deliveries.sum()
    total_wrong_deliveries = total_panic_wrong_deliveries.sum()
    episodes_finished = episodes_completed.sum()
    denom = jnp.maximum(episodes_finished, 1)
    return {
        "panic/episodes_finished": episodes_finished,
        "panic/total_actions": total_actions,
        "panic/total_reward": total_reward,
        "panic/total_deliveries": total_deliveries,
        "panic/total_wrong_deliveries": total_wrong_deliveries,
        "panic/actions_per_episode": total_actions / denom,
        "panic/reward_per_episode": total_reward / denom,
        "panic/deliveries_per_episode": total_deliveries / denom,
        "panic/wrong_deliveries_per_episode": total_wrong_deliveries / denom,
    }
