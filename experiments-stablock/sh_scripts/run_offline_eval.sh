#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [[ -z "$REPO_ROOT" || ! -d "$REPO_ROOT/overcooked_v2_experiments" ]]; then
  REPO_ROOT="/home/mlic/mingukang/ex-overcookedv2/experiments-stablock"
fi

MODE="${1:-}"
RUN_BASE_DIR="${2:-}"
if [[ -z "$MODE" || -z "$RUN_BASE_DIR" ]]; then
  echo "Usage: $0 <ph1|ph2> <run_base_dir>"
  echo "Example: $0 ph1 /home/mlic/mingukang/ex-overcookedv2/experiments-stablock/runs/20260216-123000_xxxxx"
  exit 1
fi

if [[ "$MODE" != "ph1" && "$MODE" != "ph2" ]]; then
  echo "[ERROR] mode must be one of: ph1, ph2"
  exit 1
fi

if [[ ! "$RUN_BASE_DIR" = /* ]]; then
  RUN_BASE_DIR="$REPO_ROOT/$RUN_BASE_DIR"
fi
if [[ "$RUN_BASE_DIR" = /runs/* && ! -d "$RUN_BASE_DIR" ]]; then
  RUN_BASE_DIR="$REPO_ROOT$RUN_BASE_DIR"
fi
if [[ ! -d "$RUN_BASE_DIR" ]]; then
  echo "[ERROR] run_base_dir not found: $RUN_BASE_DIR"
  exit 1
fi
if [[ ! -d "$RUN_BASE_DIR/eval" ]]; then
  echo "[ERROR] snapshot directory not found: $RUN_BASE_DIR/eval"
  exit 1
fi

EVAL_EVERY_ENV_STEPS="${EVAL_EVERY_ENV_STEPS:-200000}"
VIDEO_EVERY_ENV_STEPS="${VIDEO_EVERY_ENV_STEPS:-1000000}"
LOG_VIDEO="${LOG_VIDEO:-False}"
VIZ_MAX_STEPS="${VIZ_MAX_STEPS:-400}"
DISABLE_JIT="${DISABLE_JIT:-True}"
EVAL_ANALYSIS="${EVAL_ANALYSIS:-False}"
EVAL_ENV_DEVICE="${EVAL_ENV_DEVICE:-cpu}" # cpu | gpu
EVAL_PLATFORM="${EVAL_PLATFORM:-cpu}"   # cpu | cuda
GPU_IDX="${GPU_IDX:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

env_device_lc="$(echo "$EVAL_ENV_DEVICE" | tr '[:upper:]' '[:lower:]')"
platform_lc="$(echo "$EVAL_PLATFORM" | tr '[:upper:]' '[:lower:]')"
if [[ "$env_device_lc" == "cuda" ]]; then
  env_device_lc="gpu"
fi
if [[ "$platform_lc" == "gpu" ]]; then
  platform_lc="cuda"
fi
if [[ "$env_device_lc" != "cpu" && "$env_device_lc" != "gpu" ]]; then
  echo "[ERROR] EVAL_ENV_DEVICE must be one of: cpu, gpu"
  exit 1
fi
if [[ "$platform_lc" != "cpu" && "$platform_lc" != "cuda" ]]; then
  echo "[ERROR] EVAL_PLATFORM must be one of: cpu, cuda"
  exit 1
fi

if [[ "$MODE" == "ph1" ]]; then
  PY_SCRIPT="$REPO_ROOT/overcooked_v2_experiments/ppo/ph1_offline_video.py"
else
  PY_SCRIPT="$REPO_ROOT/overcooked_v2_experiments/ppo/ph2_offline_eval.py"
fi

cmd=(
  "$PYTHON_BIN" "$PY_SCRIPT"
  --run-base-dir "$RUN_BASE_DIR"
  --eval-every-env-steps "$EVAL_EVERY_ENV_STEPS"
  --video-every-env-steps "$VIDEO_EVERY_ENV_STEPS"
  --log-video "$LOG_VIDEO"
  --viz-max-steps "$VIZ_MAX_STEPS"
  --disable-jit "$DISABLE_JIT"
  --env-device "$env_device_lc"
)
if [[ "$MODE" == "ph1" ]]; then
  cmd+=(--eval-analysis "$EVAL_ANALYSIS")
fi

# Memory-safe defaults for JAX eval
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"

if [[ "$platform_lc" == "cpu" ]]; then
  export JAX_PLATFORMS=cpu
elif [[ "$platform_lc" == "cuda" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_IDX"
  # Keep CPU backend visible when env interaction is pinned to CPU.
  if [[ "$env_device_lc" == "cpu" ]]; then
    export JAX_PLATFORMS=cuda,cpu
  else
    export JAX_PLATFORMS=cuda
  fi
fi

echo "[INFO] Start offline eval"
echo "[INFO] mode=$MODE run_base_dir=$RUN_BASE_DIR"
if [[ "$MODE" == "ph1" ]]; then
  echo "[INFO] eval_every_env_steps=$EVAL_EVERY_ENV_STEPS video_every_env_steps=$VIDEO_EVERY_ENV_STEPS log_video=$LOG_VIDEO viz_max_steps=$VIZ_MAX_STEPS disable_jit=$DISABLE_JIT eval_analysis=$EVAL_ANALYSIS env_device=$env_device_lc platform=$platform_lc gpu=$GPU_IDX"
else
  echo "[INFO] eval_every_env_steps=$EVAL_EVERY_ENV_STEPS video_every_env_steps=$VIDEO_EVERY_ENV_STEPS log_video=$LOG_VIDEO viz_max_steps=$VIZ_MAX_STEPS disable_jit=$DISABLE_JIT env_device=$env_device_lc platform=$platform_lc gpu=$GPU_IDX"
fi
(cd "$REPO_ROOT" && PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" "${cmd[@]}")
echo "[INFO] Done. Check outputs under: $RUN_BASE_DIR/eval"
