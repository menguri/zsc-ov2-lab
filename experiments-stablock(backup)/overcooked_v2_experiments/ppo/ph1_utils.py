import jax
import jax.numpy as jnp
from functools import partial


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
    static_argnames=["apply_fn", "use_partner_pred", "use_blocked_input"],
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
    
    # 1. Normal Value (Target = -1) 계산
    # candidate_targets: (N, H, W, C)
    target_shape = candidate_targets.shape[1:] 
    dummy_shape = (batch_size,) + target_shape
        
    normal_target = jnp.full(dummy_shape, -1, dtype=jnp.float32)
    
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
        blocked = jnp.tile(candidate_k[None, ...], (batch_size,) + (1,) * candidate_k.ndim)
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
