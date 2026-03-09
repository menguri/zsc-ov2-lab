import jax
import jax.numpy as jnp
from functools import partial


def _normalize_probs(probs: jnp.ndarray) -> jnp.ndarray:
    probs = jnp.asarray(probs, dtype=jnp.float32)
    denom = jnp.sum(probs)
    return jnp.where(denom > 0.0, probs / denom, jnp.zeros_like(probs))


def sample_penalty_count(
    rng,
    max_penalty_count: int,
    single_weight: float = 2.0,
    other_weight: float = 1.0,
):
    """Sample penalty-state count in [1, max_penalty_count] with weighted odds."""
    max_penalty_count = max(1, int(max_penalty_count))
    weights = jnp.full((max_penalty_count,), jnp.float32(other_weight))
    weights = weights.at[0].set(jnp.float32(single_weight))
    probs = _normalize_probs(weights)
    idx = jax.random.choice(rng, max_penalty_count, p=probs)
    return jnp.int32(idx + 1)


def sample_multi_targets_from_pool(
    rng,
    pool_env: jnp.ndarray,
    probs_env: jnp.ndarray,
    max_penalty_count: int,
    single_weight: float = 2.0,
    other_weight: float = 1.0,
):
    """Sample up to K unique targets from pool using PH1 probs and value-based dedup.

    Returns:
      targets: (K, ...)
      valid_mask: (K,)
      sampled_count: scalar int32 in [1, K]
    """
    k = max(1, int(max_penalty_count))
    pool_size = int(pool_env.shape[0])
    reduce_axes = tuple(range(1, int(pool_env.ndim)))

    none_target = jnp.full(pool_env.shape[1:], -1, dtype=pool_env.dtype)
    init_targets = jnp.broadcast_to(none_target, (k,) + pool_env.shape[1:])
    init_valid = jnp.zeros((k,), dtype=jnp.bool_)

    # Candidate probabilities come from PH1 v-gap distribution (exclude trailing "normal" bin).
    cand_probs = jnp.asarray(probs_env[:pool_size], dtype=jnp.float32)
    valid_candidate = ~jnp.all(pool_env == -1, axis=reduce_axes)
    cand_probs = cand_probs * valid_candidate.astype(jnp.float32)
    cand_probs = _normalize_probs(cand_probs)

    rng_count, rng_slots = jax.random.split(rng)
    sampled_count = sample_penalty_count(
        rng_count,
        max_penalty_count=k,
        single_weight=single_weight,
        other_weight=other_weight,
    )
    slot_keys = jax.random.split(rng_slots, k)

    def _sample_one(slot_idx, carry):
        targets, valid_mask, probs = carry

        def _do_sample(inner_carry):
            targets_cur, valid_cur, probs_cur = inner_carry
            p_sum = jnp.sum(probs_cur)

            def _pick(pick_carry):
                t_cur, v_cur, p_cur = pick_carry
                picked_idx = jax.random.choice(slot_keys[slot_idx], pool_size, p=p_cur)
                picked = pool_env[picked_idx]
                # Remove all candidates equal to selected state to avoid duplicates.
                same_state = jnp.all(pool_env == picked, axis=reduce_axes)
                p_next = jnp.where(same_state, 0.0, p_cur)
                p_next = _normalize_probs(p_next)
                t_next = t_cur.at[slot_idx].set(picked)
                v_next = v_cur.at[slot_idx].set(True)
                return t_next, v_next, p_next

            return jax.lax.cond(
                p_sum > 0.0,
                _pick,
                lambda c: c,
                operand=(targets_cur, valid_cur, probs_cur),
            )

        return jax.lax.cond(
            slot_idx < sampled_count,
            _do_sample,
            lambda c: c,
            operand=carry,
        )

    targets, valid_mask, _ = jax.lax.fori_loop(
        0, k, _sample_one, (init_targets, init_valid, cand_probs)
    )

    return targets, valid_mask, sampled_count


def calculate_v_spec(
    apply_fn,
    params,
    obs,
    done,
    hstate,
    partner_prediction,
    blocked_target,
    use_blocked_input: bool = True,
):
    """Network Value Head를 사용하여 V_spec 계산."""
    # obs: (Batch, H, W, C) -> (1, Batch, H, W, C)
    # done: (Batch,) -> (1, Batch)
    obs_time = obs[jnp.newaxis, ...]
    done_time = done[jnp.newaxis, ...]
    
    # blocked_target: (Batch, H, W, C) -> Add time dim
    # blocked_states logic in RNN handles (Time, ...) check? 
    # RNN checks `blocked_states_in.ndim >= 4`.
    # If using observation as block target: (Batch, H, W, C). Added dim -> (1, Batch, H, W, C).
    # If using coords: (Batch, 2). Added dim -> (1, Batch, 2).
    blocked_target_time = None
    if use_blocked_input and (blocked_target is not None):
        blocked_target_time = blocked_target[jnp.newaxis, ...]

    # partner_prediction: (Batch, ActionDim) -> (1, Batch, ActionDim)
    partner_prediction_time = None
    if partner_prediction is not None:
        if partner_prediction.ndim == 2:
            partner_prediction_time = partner_prediction[jnp.newaxis, ...]
        else:
            partner_prediction_time = partner_prediction

    # train=False 모드로 실행
    _, _, value, _ = apply_fn(
        params,
        hstate,
        (obs_time, done_time),
        partner_prediction=partner_prediction_time, 
        blocked_states=blocked_target_time,
        train=False
    )
    return value.squeeze()


@partial(
    jax.jit,
    static_argnames=[
        "apply_fn",
        "use_partner_pred",
        "use_blocked_input",
        "blocked_input_slots",
    ],
)
def compute_ph1_probs(
    apply_fn,
    params,
    batch_obs,
    batch_done,
    batch_hstate,
    candidate_targets,
    candidate_partner_pred=None,
    candidate_agent_idx=None,
    use_partner_pred: bool = True,
    use_blocked_input: bool = True,
    blocked_input_slots: int = 1,
    beta: float = 1.0,
    normal_prob: float = 0.5
):
    """
    V_gap 계산 및 Softmax 확률 분포 생성.

    Args:
        apply_fn: Network forward function (network.apply)
        params: Network parameters
        batch_obs: (B, ...) - V_gap 계산용 배치의 관측값
        candidate_targets: (N, H, W, C) (State pool)
        candidate_partner_pred: (B, ActionDim) (recommended) partner prediction for the *reference batch*.
            For backward compatibility, older callers may pass (N, ActionDim); this will be ignored.
        candidate_agent_idx: deprecated (ignored). Kept only for backward compatibility.
        use_partner_pred: partner prediction 사용 여부
        normal_prob: "None"(Normal) 타겟을 선택할 고정 확률 (0.0 ~ 1.0)
    Returns:
        probs: (N+1,) - 마지막은 None(Normal)에 대한 확률
        v_gaps: (N+1,) - 각 후보의 V_gap (마지막은 0.0)
    """
    batch_size = batch_obs.shape[0]
    blocked_input_slots = max(1, int(blocked_input_slots))
    
    # 1. Normal Value (Target = -1) 계산
    # candidate_targets: (N, H, W, C)
    target_shape = candidate_targets.shape[1:] 
    dummy_shape = (batch_size,) + target_shape

    def _pack_blocked_slots(single_target_batch):
        if (not use_blocked_input) or (blocked_input_slots <= 1):
            return single_target_batch
        packed = jnp.full(
            (batch_size, blocked_input_slots) + target_shape,
            -1,
            dtype=single_target_batch.dtype,
        )
        packed = packed.at[:, 0].set(single_target_batch)
        return packed

    normal_target_single = jnp.full(dummy_shape, -1, dtype=jnp.float32)
    normal_target = _pack_blocked_slots(normal_target_single)
    
    # 2. Compute V(s, normal) ONCE (shared across candidates)
    partner_pred_batch = None
    if use_partner_pred and candidate_partner_pred is not None:
        # Accept only batch-shaped partner predictions.
        # If an older caller passes (N_candidates, ActionDim), ignore it.
        if candidate_partner_pred.ndim == 2 and candidate_partner_pred.shape[0] == batch_size:
            partner_pred_batch = candidate_partner_pred

    v_normal = calculate_v_spec(
        apply_fn,
        params,
        batch_obs,
        batch_done,
        batch_hstate,
        partner_pred_batch,
        normal_target,
        use_blocked_input=use_blocked_input,
    )

    # 3. 각 후보별 V_gap 계산
    def _get_v_gap_single(candidate_k):
        # candidate_k shape: (...)
        # Expand: (H,W,C) -> (B, H,W,C)
        blocked_single = jnp.tile(
            candidate_k[None, ...], (batch_size,) + (1,) * candidate_k.ndim
        )
        blocked = _pack_blocked_slots(blocked_single)
        v_t = calculate_v_spec(
            apply_fn,
            params,
            batch_obs,
            batch_done,
            batch_hstate,
            partner_pred_batch,
            blocked,
            use_blocked_input=use_blocked_input,
        )
        return jnp.mean(v_normal - v_t)

    v_gaps = jax.vmap(_get_v_gap_single)(candidate_targets)
    
    # 3. Softmax Sampling (Candidates Only)
    # PH1/PH2 공통: V_gap이 작을수록(어려움이 낮을수록) 더 자주 샘플링.
    logits_cands = -beta * v_gaps
    probs_cands = jax.nn.softmax(logits_cands)
    
    # 4. Apply Normal Probability Mixing
    # Candidates total mass = (1.0 - normal_prob)
    probs_cands_scaled = probs_cands * (1.0 - normal_prob)
    
    # Combine: [Candidates..., Normal]
    probs = jnp.concatenate([probs_cands_scaled, jnp.array([normal_prob])])
    
    # For logging/debug (append 0.0 for Normal's V_gap)
    v_gaps_all = jnp.concatenate([v_gaps, jnp.array([0.0])])
    
    return probs, v_gaps_all

def sample_target_idx(rng, probs):
    """확률 분포에 따라 타겟 인덱스 샘플링 (마지막 인덱스는 None)"""
    return jax.random.choice(rng, len(probs), p=probs)
