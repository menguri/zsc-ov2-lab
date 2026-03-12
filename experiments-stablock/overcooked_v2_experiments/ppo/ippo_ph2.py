import copy
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from flax.training.train_state import TrainState

from .ippo_ph2_core import make_train as make_train_core
from .policy import PPOPolicy

_SHARED_PREDICTOR_KEYS = ("shared_predictor", "shared_state_predictor")


def _resolve_num_updates(model_cfg: Dict[str, Any]) -> int:
    if "NUM_UPDATES" in model_cfg:
        return int(model_cfg["NUM_UPDATES"])
    total_timesteps = int(model_cfg["TOTAL_TIMESTEPS"])
    num_steps = int(model_cfg["NUM_STEPS"])
    num_envs = int(model_cfg["NUM_ENVS"])
    return max(1, total_timesteps // max(1, num_steps * num_envs))


def _build_checkpoint_steps(config: Dict[str, Any], num_updates: int) -> np.ndarray:
    model_cfg = config["model"]
    num_steps = int(model_cfg["NUM_STEPS"])
    num_envs = int(model_cfg["NUM_ENVS"])
    steps_per_update = max(1, num_steps * num_envs)
    total_timesteps = int(model_cfg["TOTAL_TIMESTEPS"])
    ckpt_every_env_steps = int(config.get("PH1_EVAL_EVERY_ENV_STEPS", 0))
    ph1_eval_enabled = bool(config.get("PH1_EVAL_ENABLED", False)) and bool(
        config.get("PH1_ENABLED", False)
    )

    use_env_step_ckpt_schedule = ph1_eval_enabled and (ckpt_every_env_steps > 0)
    if use_env_step_ckpt_schedule:
        target_env_steps = np.arange(
            0,
            total_timesteps + ckpt_every_env_steps,
            ckpt_every_env_steps,
            dtype=np.int64,
        )
        if target_env_steps[-1] != total_timesteps:
            target_env_steps = np.append(target_env_steps, total_timesteps)
        ckpt_steps = np.unique(
            np.clip(
                np.ceil(target_env_steps / steps_per_update).astype(np.int32),
                0,
                num_updates,
            )
        )
        return ckpt_steps.astype(np.int32)

    num_checkpoints = int(config.get("NUM_CHECKPOINTS", 0))
    if num_checkpoints <= 0:
        return np.zeros((0,), dtype=np.int32)
    ckpt_steps = np.linspace(
        0,
        num_updates,
        num_checkpoints,
        endpoint=True,
        dtype=np.int32,
    )
    if ckpt_steps.shape[0] > 0:
        ckpt_steps[-1] = np.int32(num_updates)
    return ckpt_steps.astype(np.int32)


def _extract_recent_tilde_from_runner(runner_state):
    # runner_state[20] = ph1_pool_states: (Envs, Pool, H, W, C_full)
    if len(runner_state) <= 20:
        return jnp.zeros((0,), dtype=jnp.float32), jnp.array(False)
    pool_states = runner_state[20]
    if pool_states.ndim < 2:
        return jnp.zeros((0,), dtype=jnp.float32), jnp.array(False)
    env0 = pool_states[0]
    if env0.ndim < 1 or env0.shape[0] <= 0:
        return jnp.zeros((0,), dtype=jnp.float32), jnp.array(False)
    reduce_axes = tuple(range(1, int(env0.ndim)))
    if len(reduce_axes) == 0:
        valid = env0 >= 0.0
    else:
        valid = jnp.any(env0 >= 0.0, axis=reduce_axes)
    has_recent = jnp.any(valid)
    last_idx = jnp.maximum(jnp.sum(valid.astype(jnp.int32)) - 1, 0)
    recent = env0[last_idx]
    recent = jnp.where(has_recent, recent, jnp.zeros_like(env0[0]))
    return recent.astype(jnp.float32), has_recent


def _set_checkpoint(accum, params, ckpt_idx):
    return jax.tree_util.tree_map(
        lambda a, p: a.at[ckpt_idx].set(p),
        accum,
        params,
    )


def _extract_shared_predictor_subtree(tree, target_key="shared_predictor"):
    state_dict = serialization.to_state_dict(tree)

    def _find(node):
        if isinstance(node, dict):
            if target_key in node:
                return node[target_key]
            for child in node.values():
                out = _find(child)
                if out is not None:
                    return out
        return None

    return _find(state_dict)


def _replace_shared_predictor_subtree(tree, shared_subtree, target_key="shared_predictor"):
    if shared_subtree is None:
        return tree
    state_dict = serialization.to_state_dict(tree)

    def _replace(node):
        if not isinstance(node, dict):
            return False, node
        out = dict(node)
        replaced = False
        if target_key in out:
            out[target_key] = shared_subtree
            replaced = True
        for key, child in out.items():
            child_replaced, child_out = _replace(child)
            if child_replaced:
                out[key] = child_out
                replaced = True
        return replaced, out

    replaced, new_state_dict = _replace(state_dict)
    if not replaced:
        return tree
    return serialization.from_state_dict(tree, new_state_dict)


def _sync_train_state_shared_predictor(
    train_state: TrainState,
    shared_params_subtree,
    target_key="shared_predictor",
) -> TrainState:
    new_params = _replace_shared_predictor_subtree(
        train_state.params, shared_params_subtree, target_key=target_key
    )
    if new_params is train_state.params:
        return train_state
    return train_state.replace(params=new_params)


def _extract_shared_predictor_bundle(tree):
    return {
        key: _extract_shared_predictor_subtree(tree, target_key=key)
        for key in _SHARED_PREDICTOR_KEYS
    }


def _sync_train_state_shared_predictor_bundle(
    train_state: TrainState,
    shared_params_bundle,
) -> TrainState:
    out = train_state
    for key in _SHARED_PREDICTOR_KEYS:
        out = _sync_train_state_shared_predictor(
            out,
            shared_params_bundle.get(key),
            target_key=key,
        )
    return out


def _maybe_parse_resume_state(initial_train_state):
    # Optional resume layout:
    # - TrainState: spec only
    # - (spec_state, ind_state, *)
    # - {"spec": ..., "ind": ...}
    spec_state = None
    ind_state = None
    if initial_train_state is None:
        return spec_state, ind_state
    if isinstance(initial_train_state, TrainState):
        return initial_train_state, None
    if isinstance(initial_train_state, dict):
        return (
            initial_train_state.get("spec", None),
            initial_train_state.get("ind", None),
        )
    if isinstance(initial_train_state, (tuple, list)):
        if len(initial_train_state) >= 1:
            spec_state = initial_train_state[0]
        if len(initial_train_state) >= 2:
            ind_state = initial_train_state[1]
    return spec_state, ind_state


def make_train(
    config: Dict[str, Any],
    update_step_offset: Optional[int] = None,
    update_step_num_overwrite: Optional[int] = None,
    population_config: Optional[Dict[str, Any]] = None,
):
    cfg = copy.deepcopy(dict(config))
    shared_prediction = bool(cfg.get("SHARED_PREDICTION", False))
    model_cfg = cfg["model"]
    total_updates_cfg = _resolve_num_updates(model_cfg)
    total_updates = (
        total_updates_cfg
        if update_step_num_overwrite is None
        else int(update_step_num_overwrite)
    )
    total_updates = max(0, int(total_updates))
    initial_offset = 0 if update_step_offset is None else int(update_step_offset)

    # Use env-step checkpoint cadence even when online eval is disabled.
    # This keeps eval/ckpt_* snapshots available for offline viz/eval.
    ckpt_cadence_cfg = copy.deepcopy(cfg)
    if int(cfg.get("PH1_EVAL_EVERY_ENV_STEPS", 0)) > 0:
        ckpt_cadence_cfg["PH1_EVAL_ENABLED"] = True
    checkpoint_steps_np = _build_checkpoint_steps(ckpt_cadence_cfg, total_updates_cfg)
    checkpoint_steps = jnp.asarray(checkpoint_steps_np, dtype=jnp.int32)
    num_checkpoints = int(checkpoint_steps_np.shape[0])

    fixed_ind_prob = cfg.get("PH2_FIXED_IND_PROB", 0.5)
    if fixed_ind_prob is not None:
        fixed_ind_prob = float(np.clip(float(fixed_ind_prob), 0.0, 1.0))

    spec_cfg = copy.deepcopy(cfg)
    ind_cfg = copy.deepcopy(cfg)
    spec_cfg["PHASE_LOG_PREFIX"] = "phase1"
    ind_cfg["PHASE_LOG_PREFIX"] = "phase2"
    spec_cfg["PH2_MATCH_SCHEDULE"] = True
    ind_cfg["PH2_MATCH_SCHEDULE"] = True
    spec_cfg["PH2_ROLE"] = "spec"
    ind_cfg["PH2_ROLE"] = "ind"
    spec_cfg["PH2_FIXED_IND_PROB"] = fixed_ind_prob
    ind_cfg["PH2_FIXED_IND_PROB"] = fixed_ind_prob

    # PH2 trains without in-training evaluation, but still keeps checkpoint snapshots.
    spec_cfg["PH1_EVAL_ENABLED"] = bool(ckpt_cadence_cfg.get("PH1_EVAL_ENABLED", False))
    ind_cfg["PH1_EVAL_ENABLED"] = bool(ckpt_cadence_cfg.get("PH1_EVAL_ENABLED", False))

    # Blocked-target input policy:
    # - spec: configurable
    # - ind : always disabled (learner/population 모두 blocked target 미사용)
    spec_cfg["LEARNER_USE_BLOCKED_INPUT"] = bool(cfg.get("PH2_SPEC_USE_BLOCKED_INPUT", True))
    ind_cfg["LEARNER_USE_BLOCKED_INPUT"] = False
    # Population partner forward path must match each partner architecture.
    spec_cfg["POPULATION_USE_PARTNER_PRED_INPUT"] = True
    ind_cfg["POPULATION_USE_PARTNER_PRED_INPUT"] = True
    # spec_step에서 population은 ind 정책이므로 blocked input을 강제로 끈다.
    spec_cfg["POPULATION_USE_BLOCKED_INPUT"] = False
    ind_cfg["POPULATION_USE_BLOCKED_INPUT"] = bool(
        cfg.get("PH2_SPEC_USE_BLOCKED_INPUT", True)
    )
    spec_cfg["SHARED_PREDICTION"] = shared_prediction
    ind_cfg["SHARED_PREDICTION"] = shared_prediction

    # Build one-update trainers. We run a PH1-like update loop at wrapper level.
    spec_step = make_train_core(
        spec_cfg,
        update_step_num_overwrite=1,
        population_config=population_config,
    )
    ind_step = make_train_core(
        ind_cfg,
        update_step_num_overwrite=1,
        population_config=population_config,
    )
    spec_init = make_train_core(
        spec_cfg,
        update_step_num_overwrite=0,
        population_config=population_config,
    )
    ind_init = make_train_core(
        ind_cfg,
        update_step_num_overwrite=0,
        population_config=population_config,
    )

    def train(rng, population=None, initial_train_state=None):
        if population is not None:
            raise ValueError(
                "PH2-E3T does not support external population/BC partners."
            )

        original_seed = rng[0]
        log_seed_override = jnp.asarray(original_seed)

        spec_state, ind_state = _maybe_parse_resume_state(initial_train_state)

        rng, rng_spec_init, rng_ind_init, rng_loop = jax.random.split(rng, 4)
        spec_init_out = spec_init(rng_spec_init)
        ind_init_out = ind_init(rng_ind_init)

        def _set_runner_field(runner_state, idx, value):
            r = list(runner_state)
            r[idx] = value
            return tuple(r)

        spec_runner_state = spec_init_out["runner_state"]
        ind_runner_state = ind_init_out["runner_state"]

        if spec_state is None:
            spec_state = spec_runner_state[0]
        else:
            spec_runner_state = _set_runner_field(spec_runner_state, 0, spec_state)
        if ind_state is None:
            ind_state = ind_runner_state[0]
        else:
            ind_runner_state = _set_runner_field(ind_runner_state, 0, ind_state)

        # Shared-prediction mode sync:
        # initialize shared predictor state from spec, then inject into ind.
        if shared_prediction:
            shared_pred_params = _extract_shared_predictor_bundle(spec_state.params)
            ind_state = _sync_train_state_shared_predictor_bundle(
                ind_state, shared_pred_params
            )
            ind_runner_state = _set_runner_field(ind_runner_state, 0, ind_state)
        else:
            shared_pred_params = {}

        if int(initial_offset) != 0:
            spec_runner_state = _set_runner_field(
                spec_runner_state, 5, jnp.asarray(initial_offset, dtype=jnp.int32)
            )
            ind_runner_state = _set_runner_field(
                ind_runner_state, 5, jnp.asarray(initial_offset, dtype=jnp.int32)
            )

        if num_checkpoints > 0:
            spec_ckpt_accum = jax.tree_util.tree_map(
                lambda p: jnp.zeros((num_checkpoints,) + p.shape, dtype=p.dtype),
                spec_state.params,
            )
            ind_ckpt_accum = jax.tree_util.tree_map(
                lambda p: jnp.zeros((num_checkpoints,) + p.shape, dtype=p.dtype),
                ind_state.params,
            )
        else:
            spec_ckpt_accum = jax.tree_util.tree_map(
                lambda p: jnp.zeros((0,) + p.shape, dtype=p.dtype),
                spec_state.params,
            )
            ind_ckpt_accum = jax.tree_util.tree_map(
                lambda p: jnp.zeros((0,) + p.shape, dtype=p.dtype),
                ind_state.params,
            )

        recent0, has_recent0 = _extract_recent_tilde_from_runner(
            spec_init_out["runner_state"]
        )
        recent_ckpts = jnp.zeros((num_checkpoints,) + recent0.shape, dtype=jnp.float32)
        recent_has = jnp.zeros((num_checkpoints,), dtype=jnp.bool_)

        if num_checkpoints > 0 and int(checkpoint_steps_np[0]) == 0 and initial_offset == 0:
            spec_ckpt_accum = _set_checkpoint(spec_ckpt_accum, spec_state.params, 0)
            ind_ckpt_accum = _set_checkpoint(ind_ckpt_accum, ind_state.params, 0)
            recent_ckpts = recent_ckpts.at[0].set(recent0)
            recent_has = recent_has.at[0].set(has_recent0)

        policy_ind = PPOPolicy(
            params=ind_state.params,
            config=ind_cfg,
            stochastic=True,
            with_batching=True,
        )
        policy_spec = PPOPolicy(
            params=spec_state.params,
            config=spec_cfg,
            stochastic=True,
            with_batching=True,
        )

        run_steps = int(total_updates)
        curr_update = jnp.asarray(initial_offset, dtype=jnp.int32)
        last_spec_out = spec_init_out
        last_ind_out = ind_init_out

        if run_steps > 0:
            def _advance_one_step(carry):
                (
                    spec_runner_c,
                    ind_runner_c,
                    spec_ckpt_c,
                    ind_ckpt_c,
                    recent_ckpts_c,
                    recent_has_c,
                    curr_update_c,
                    rng_loop_c,
                    _last_spec_out_c,
                    _last_ind_out_c,
                ) = carry

                rng_loop_c, rng_spec_call, rng_ind_call = jax.random.split(rng_loop_c, 3)
                spec_state_c = spec_runner_c[0]
                ind_state_c = ind_runner_c[0]

                spec_out_c = spec_step(
                    rng_spec_call,
                    population=policy_ind,
                    initial_runner_state=spec_runner_c,
                    log_seed_override=log_seed_override,
                    population_params_override=ind_state_c.params,
                )
                next_spec_runner = spec_out_c["runner_state"]
                next_spec_state = next_spec_runner[0]

                ind_out_c = ind_step(
                    rng_ind_call,
                    population=policy_spec,
                    initial_runner_state=ind_runner_c,
                    log_seed_override=log_seed_override,
                    population_params_override=next_spec_state.params,
                )
                next_ind_runner = ind_out_c["runner_state"]
                next_ind_state = next_ind_runner[0]

                next_update = curr_update_c + jnp.int32(1)
                if num_checkpoints > 0:
                    hit_selector = checkpoint_steps == next_update
                    has_hit = jnp.any(hit_selector)
                    hit_idx = jnp.argmax(hit_selector)

                    spec_ckpt_c = jax.lax.cond(
                        has_hit,
                        lambda x: _set_checkpoint(x, next_spec_state.params, hit_idx),
                        lambda x: x,
                        spec_ckpt_c,
                    )
                    ind_ckpt_c = jax.lax.cond(
                        has_hit,
                        lambda x: _set_checkpoint(x, next_ind_state.params, hit_idx),
                        lambda x: x,
                        ind_ckpt_c,
                    )

                    recent_tilde_c, has_recent_c = _extract_recent_tilde_from_runner(
                        spec_out_c["runner_state"]
                    )
                    recent_ckpts_c = jax.lax.cond(
                        has_hit,
                        lambda x: x.at[hit_idx].set(recent_tilde_c),
                        lambda x: x,
                        recent_ckpts_c,
                    )
                    recent_has_c = jax.lax.cond(
                        has_hit,
                        lambda x: x.at[hit_idx].set(has_recent_c),
                        lambda x: x,
                        recent_has_c,
                    )

                return (
                    next_spec_runner,
                    next_ind_runner,
                    spec_ckpt_c,
                    ind_ckpt_c,
                    recent_ckpts_c,
                    recent_has_c,
                    next_update,
                    rng_loop_c,
                    spec_out_c,
                    ind_out_c,
                )

            def _advance_one_step_shared(carry):
                (
                    spec_runner_c,
                    ind_runner_c,
                    spec_ckpt_c,
                    ind_ckpt_c,
                    recent_ckpts_c,
                    recent_has_c,
                    curr_update_c,
                    rng_loop_c,
                    _last_spec_out_c,
                    _last_ind_out_c,
                    shared_params_c,
                ) = carry

                rng_loop_c, rng_spec_call, rng_ind_call = jax.random.split(rng_loop_c, 3)
                spec_state_c = spec_runner_c[0]
                ind_state_c = ind_runner_c[0]

                # Inject shared predictor into both train states before each role update.
                spec_state_c = _sync_train_state_shared_predictor_bundle(
                    spec_state_c, shared_params_c
                )
                ind_state_c = _sync_train_state_shared_predictor_bundle(
                    ind_state_c, shared_params_c
                )
                spec_runner_c = _set_runner_field(spec_runner_c, 0, spec_state_c)
                ind_runner_c = _set_runner_field(ind_runner_c, 0, ind_state_c)

                spec_out_c = spec_step(
                    rng_spec_call,
                    population=policy_ind,
                    initial_runner_state=spec_runner_c,
                    log_seed_override=log_seed_override,
                    population_params_override=ind_state_c.params,
                )
                next_spec_runner = spec_out_c["runner_state"]
                next_spec_state = next_spec_runner[0]
                shared_params_after_spec = _extract_shared_predictor_bundle(
                    next_spec_state.params
                )

                # Spec updated predictor is injected into ind before ind step.
                ind_state_step = _sync_train_state_shared_predictor_bundle(
                    ind_state_c, shared_params_after_spec
                )
                ind_runner_step = _set_runner_field(ind_runner_c, 0, ind_state_step)
                ind_out_c = ind_step(
                    rng_ind_call,
                    population=policy_spec,
                    initial_runner_state=ind_runner_step,
                    log_seed_override=log_seed_override,
                    population_params_override=next_spec_state.params,
                )
                next_ind_runner = ind_out_c["runner_state"]
                next_ind_state = next_ind_runner[0]

                # Final shared predictor after ind update; sync both states.
                shared_params_next = _extract_shared_predictor_bundle(
                    next_ind_state.params
                )
                next_spec_state = _sync_train_state_shared_predictor_bundle(
                    next_spec_state, shared_params_next
                )
                next_ind_state = _sync_train_state_shared_predictor_bundle(
                    next_ind_state, shared_params_next
                )
                next_spec_runner = _set_runner_field(next_spec_runner, 0, next_spec_state)
                next_ind_runner = _set_runner_field(next_ind_runner, 0, next_ind_state)

                next_update = curr_update_c + jnp.int32(1)
                if num_checkpoints > 0:
                    hit_selector = checkpoint_steps == next_update
                    has_hit = jnp.any(hit_selector)
                    hit_idx = jnp.argmax(hit_selector)

                    spec_ckpt_c = jax.lax.cond(
                        has_hit,
                        lambda x: _set_checkpoint(x, next_spec_state.params, hit_idx),
                        lambda x: x,
                        spec_ckpt_c,
                    )
                    ind_ckpt_c = jax.lax.cond(
                        has_hit,
                        lambda x: _set_checkpoint(x, next_ind_state.params, hit_idx),
                        lambda x: x,
                        ind_ckpt_c,
                    )

                    recent_tilde_c, has_recent_c = _extract_recent_tilde_from_runner(
                        spec_out_c["runner_state"]
                    )
                    recent_ckpts_c = jax.lax.cond(
                        has_hit,
                        lambda x: x.at[hit_idx].set(recent_tilde_c),
                        lambda x: x,
                        recent_ckpts_c,
                    )
                    recent_has_c = jax.lax.cond(
                        has_hit,
                        lambda x: x.at[hit_idx].set(has_recent_c),
                        lambda x: x,
                        recent_has_c,
                    )

                return (
                    next_spec_runner,
                    next_ind_runner,
                    spec_ckpt_c,
                    ind_ckpt_c,
                    recent_ckpts_c,
                    recent_has_c,
                    next_update,
                    rng_loop_c,
                    spec_out_c,
                    ind_out_c,
                    shared_params_next,
                )

            if shared_prediction:
                carry = (
                    spec_runner_state,
                    ind_runner_state,
                    spec_ckpt_accum,
                    ind_ckpt_accum,
                    recent_ckpts,
                    recent_has,
                    curr_update,
                    rng_loop,
                    last_spec_out,
                    last_ind_out,
                    shared_pred_params,
                )
                carry = _advance_one_step_shared(carry)

                if run_steps > 1:
                    def _scan_step_shared(carry_scan, _):
                        return _advance_one_step_shared(carry_scan), None

                    carry, _ = jax.lax.scan(
                        _scan_step_shared,
                        carry,
                        None,
                        length=run_steps - 1,
                    )

                (
                    spec_runner_state,
                    ind_runner_state,
                    spec_ckpt_accum,
                    ind_ckpt_accum,
                    recent_ckpts,
                    recent_has,
                    curr_update,
                    rng_loop,
                    last_spec_out,
                    last_ind_out,
                    shared_pred_params,
                ) = carry
            else:
                carry = (
                    spec_runner_state,
                    ind_runner_state,
                    spec_ckpt_accum,
                    ind_ckpt_accum,
                    recent_ckpts,
                    recent_has,
                    curr_update,
                    rng_loop,
                    last_spec_out,
                    last_ind_out,
                )
                carry = _advance_one_step(carry)

                if run_steps > 1:
                    def _scan_step(carry_scan, _):
                        return _advance_one_step(carry_scan), None

                    carry, _ = jax.lax.scan(
                        _scan_step,
                        carry,
                        None,
                        length=run_steps - 1,
                    )

                (
                    spec_runner_state,
                    ind_runner_state,
                    spec_ckpt_accum,
                    ind_ckpt_accum,
                    recent_ckpts,
                    recent_has,
                    curr_update,
                    rng_loop,
                    last_spec_out,
                    last_ind_out,
                ) = carry

            spec_state = spec_runner_state[0]
            ind_state = ind_runner_state[0]

        spec_runner = list(spec_runner_state)
        spec_runner[0] = spec_state
        spec_runner[1] = spec_ckpt_accum
        spec_runner[5] = jnp.asarray(curr_update, dtype=jnp.int32)

        # Keep legacy dual layout indices for main/offline eval compatibility.
        runner_state = tuple(
            spec_runner
            + [
                ind_state,      # -5
                ind_ckpt_accum, # -4
                spec_state.params,  # -3 (legacy placeholder)
                recent_ckpts,   # -2
                recent_has,     # -1
            ]
        )

        metrics = {
            "phase1": last_spec_out["metrics"],
            "phase2": last_ind_out["metrics"],
        }
        return {
            "runner_state": runner_state,
            "metrics": metrics,
            "is_ph2_dual": jnp.array(True, dtype=jnp.bool_),
        }

    return train
