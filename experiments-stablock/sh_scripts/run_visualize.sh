#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [[ -z "$REPO_ROOT" || ! -d "$REPO_ROOT/overcooked_v2_experiments" ]]; then
    REPO_ROOT="/home/mlic/mingukang/ex-overcookedv2/experiments-stablock"
fi

# Default values
GPU_IDX=0
NUM_SEEDS=10
NUM_RECENT_TILDES=5
SEED=42
ALL_CKPT=false
CROSS=false
ALL=false
NO_VIZ=false
NO_RESET=false
LATENT_ANALYSIS=false
VALUE_ANALYSIS=false
PAIRING_POLICY=""
DIRECTORY=""
EVAL_ANALYSIS=false
EVAL_VIZ=false
EVAL_PHASE="auto"         # ph1 | ph2 | auto
EVAL_EVERY_ENV_STEPS=200000
EVAL_VIZ_MAX_STEPS=400
EVAL_DISABLE_JIT="False"
EVAL_ENV_DEVICE="cpu"
EVAL_PLATFORM="cuda"

usage() {
    echo "Usage: $0 --dir <directory> [options]"
    echo "Options (visualize mode):"
    echo "  --gpu <id>                GPU device ID (default: 0)"
    echo "  --dir <path>              Directory with checkpoints (required)"
    echo "  --num_seeds <n>           Number of evaluation seeds (default: 10)"
    echo "  --num_recent_tildes <n>   PH1 cross mode: number of recent tilde samples (default: 5)"
    echo "  --seed <n>                Random seed (default: 42)"
    echo "  --all_ckpt                Evaluate all checkpoints, not just final"
    echo "  --cross                   Run cross-play evaluation"
    echo "  --all                     Run both self-play and cross-play"
    echo "  --no_viz                  Skip video generation (only compute metrics)"
    echo "  --no_reset                Disable random reset and permutations"
    echo "  --pairing_policy <id>     Policy index for pairing in cross-play"
    echo "  --latent_analysis         Evaluate fixed e_t per episode (self-play only)"
    echo "  --value_analysis          Evaluate all e_t candidates per step (self-play only)"
    echo ""
    echo "Options (offline eval-analysis mode):"
    echo "  --eval-analysis           Run offline eval analysis csv generation"
    echo "                            (video generation is always disabled)"
    echo "  --eval-viz                Run offline eval video extraction from final snapshots"
    echo "                            (one 400-step video per mode: normal/recent/random)"
    echo "                            Output: <run_dir>/eval/video/*.gif"
    echo "  --phase <ph1|ph2|auto>    Eval mode for --eval-analysis (default: auto)"
    echo "  --eval-every-env-steps N  Step filter for --eval-analysis (default: 200000)"
    echo "  --eval-viz-max-steps N    Rollout length for eval (default: 400)"
    echo "  --eval-disable-jit <T/F>  Disable JIT during offline eval (default: False)"
    echo "  --eval-env-device <cpu|gpu>  Env interaction device (default: cpu)"
    echo "  --eval-platform <cpu|cuda>   Eval platform backend (default: cuda)"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)            GPU_IDX="$2";          shift 2 ;;
        --dir)            DIRECTORY="$2";        shift 2 ;;
        --num_seeds)      NUM_SEEDS="$2";        shift 2 ;;
        --num_recent_tildes) NUM_RECENT_TILDES="$2"; shift 2 ;;
        --seed)           SEED="$2";             shift 2 ;;
        --all_ckpt)       ALL_CKPT=true;         shift ;;
        --cross)          CROSS=true;            shift ;;
        --all)            ALL=true;              shift ;;
        --no_viz)         NO_VIZ=true;           shift ;;
        --no_reset)       NO_RESET=true;         shift ;;
        --pairing_policy) PAIRING_POLICY="$2";   shift 2 ;;
        --latent_analysis) LATENT_ANALYSIS=true; shift ;;
        --value_analysis) VALUE_ANALYSIS=true; shift ;;
        --eval-analysis)  EVAL_ANALYSIS=true;    shift ;;
        --eval-viz)       EVAL_VIZ=true;         shift ;;
        --phase)          EVAL_PHASE="$2";       shift 2 ;;
        --eval-every-env-steps) EVAL_EVERY_ENV_STEPS="$2"; shift 2 ;;
        --eval-viz-max-steps|--viz-max-steps) EVAL_VIZ_MAX_STEPS="$2"; shift 2 ;;
        --eval-disable-jit) EVAL_DISABLE_JIT="$2"; shift 2 ;;
        --eval-env-device) EVAL_ENV_DEVICE="$2"; shift 2 ;;
        --eval-platform) EVAL_PLATFORM="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "$DIRECTORY" ]; then
    echo "Error: --dir <directory> is required"
    exit 1
fi

if [[ "$EVAL_ANALYSIS" == true && "$EVAL_VIZ" == true ]]; then
    echo "Error: --eval-analysis and --eval-viz cannot be used together"
    exit 1
fi

if [[ "$EVAL_VIZ" == true ]]; then
    PHASE_LOWER="$(echo "$EVAL_PHASE" | tr '[:upper:]' '[:lower:]')"
    if [[ "$PHASE_LOWER" == "auto" ]]; then
        base_name="$(basename "$DIRECTORY" | tr '[:upper:]' '[:lower:]')"
        if [[ "$base_name" == *"_ph2"* || "$base_name" == *"ph2"* ]]; then
            PHASE_LOWER="ph2"
        else
            PHASE_LOWER="ph1"
        fi
    fi
    if [[ "$PHASE_LOWER" != "ph1" && "$PHASE_LOWER" != "ph2" ]]; then
        echo "Error: --phase must be one of ph1|ph2|auto"
        exit 1
    fi
    if [[ "$PHASE_LOWER" != "ph1" ]]; then
        echo "Error: --eval-viz currently supports ph1 runs only"
        exit 1
    fi

    echo "=== Eval Viz Configuration ==="
    echo "GPU: $GPU_IDX"
    echo "Directory: $DIRECTORY"
    echo "Phase: $PHASE_LOWER"
    echo "Viz max steps: $EVAL_VIZ_MAX_STEPS"
    echo "Disable JIT: $EVAL_DISABLE_JIT"
    echo "Env device: $EVAL_ENV_DEVICE"
    echo "Platform: $EVAL_PLATFORM"
    echo "Output video dir: $DIRECTORY/eval/video"
    echo "====================================="

    cmd=(./sh_scripts/run_offline_eval_viz.sh "$PHASE_LOWER" "$DIRECTORY")
    (
      cd "$REPO_ROOT" || exit 1
      env \
        VIZ_MAX_STEPS="$EVAL_VIZ_MAX_STEPS" \
        DISABLE_JIT="$EVAL_DISABLE_JIT" \
        EVAL_ENV_DEVICE="$EVAL_ENV_DEVICE" \
        EVAL_PLATFORM="$EVAL_PLATFORM" \
        GPU_IDX="$GPU_IDX" \
        PYTHON_BIN="python3" \
        "${cmd[@]}"
    )

    echo ""
    echo "Eval viz complete!"
    echo "Results saved in: $DIRECTORY/eval/video"
    exit 0
fi

if [[ "$EVAL_ANALYSIS" == true ]]; then
    PHASE_LOWER="$(echo "$EVAL_PHASE" | tr '[:upper:]' '[:lower:]')"
    if [[ "$PHASE_LOWER" == "auto" ]]; then
        base_name="$(basename "$DIRECTORY" | tr '[:upper:]' '[:lower:]')"
        if [[ "$base_name" == *"_ph2"* || "$base_name" == *"ph2"* ]]; then
            PHASE_LOWER="ph2"
        else
            PHASE_LOWER="ph1"
        fi
    fi
    if [[ "$PHASE_LOWER" != "ph1" && "$PHASE_LOWER" != "ph2" ]]; then
        echo "Error: --phase must be one of ph1|ph2|auto"
        exit 1
    fi

    echo "=== Eval Analysis Configuration ==="
    echo "GPU: $GPU_IDX"
    echo "Directory: $DIRECTORY"
    echo "Phase: $PHASE_LOWER"
    echo "Eval every env steps: $EVAL_EVERY_ENV_STEPS"
    echo "Viz max steps: $EVAL_VIZ_MAX_STEPS"
    echo "Disable JIT: $EVAL_DISABLE_JIT"
    echo "Env device: $EVAL_ENV_DEVICE"
    echo "Platform: $EVAL_PLATFORM"
    echo "Video generation: disabled (--eval-analysis mode)"
    echo "====================================="

    cmd=(./sh_scripts/run_offline_eval.sh "$PHASE_LOWER" "$DIRECTORY")
    (
      cd "$REPO_ROOT" || exit 1
      env \
        EVAL_EVERY_ENV_STEPS="$EVAL_EVERY_ENV_STEPS" \
        VIDEO_EVERY_ENV_STEPS="0" \
        LOG_VIDEO="False" \
        VIZ_MAX_STEPS="$EVAL_VIZ_MAX_STEPS" \
        DISABLE_JIT="$EVAL_DISABLE_JIT" \
        EVAL_ANALYSIS="True" \
        EVAL_ENV_DEVICE="$EVAL_ENV_DEVICE" \
        EVAL_PLATFORM="$EVAL_PLATFORM" \
        GPU_IDX="$GPU_IDX" \
        PYTHON_BIN="python3" \
        "${cmd[@]}"
    )

    echo ""
    echo "Eval analysis complete!"
    echo "Results saved in: $DIRECTORY/eval/offline_eval_analysis.csv"
    exit 0
fi

# Set GPU environment variable
export CUDA_VISIBLE_DEVICES=$GPU_IDX

# Disable WANDB for visualization
export WANDB_MODE=disabled

# JAX 플랫폼 설정: GPU 사용
export JAX_PLATFORMS=cuda

# Print configuration
echo "=== Visualization Configuration ==="
echo "GPU: $GPU_IDX"
echo "Directory: $DIRECTORY"
echo "Number of seeds: $NUM_SEEDS"
echo "Number of recent tildes: $NUM_RECENT_TILDES"
echo "Random seed: $SEED"
echo "All checkpoints: $ALL_CKPT"
echo "Cross-play: $CROSS"
echo "All modes: $ALL"
echo "No visualization: $NO_VIZ"
echo "No reset: $NO_RESET"
echo "Latent analysis: $LATENT_ANALYSIS"
echo "Value analysis: $VALUE_ANALYSIS"
[ -n "$PAIRING_POLICY" ] && echo "Pairing policy: $PAIRING_POLICY"
echo "===================================="

# PH1 cross mode: sample recent tilde{s} from eval snapshots and evaluate all cross/self pairings.
dir_base_lc="$(basename "$DIRECTORY" | tr '[:upper:]' '[:lower:]')"
if [[ "$CROSS" == true && "$dir_base_lc" == *"ph1"* ]]; then
    if [[ "$LATENT_ANALYSIS" == true || "$VALUE_ANALYSIS" == true ]]; then
        echo "Error: --latent_analysis/--value_analysis are not supported in PH1 recent-tilde cross mode"
        exit 1
    fi

    if [[ "$ALL_CKPT" == true ]]; then
        echo "[WARN] --all_ckpt is ignored in PH1 recent-tilde cross mode (final checkpoints only)."
    fi
    if [[ "$ALL" == true ]]; then
        echo "[WARN] --all is ignored in PH1 recent-tilde cross mode (self/cross are both evaluated in one pass)."
    fi

    echo "=== PH1 Recent-Tilde Cross Configuration ==="
    echo "GPU: $GPU_IDX"
    echo "Directory: $DIRECTORY"
    echo "Number of seeds: $NUM_SEEDS"
    echo "Number of recent tilde samples: $NUM_RECENT_TILDES"
    echo "No visualization: $NO_VIZ"
    echo "No reset: $NO_RESET"
    [ -n "$PAIRING_POLICY" ] && echo "Pairing policy: $PAIRING_POLICY"
    echo "============================================="

    PH1_ARGS=(
      --d "$DIRECTORY"
      --seed "$SEED"
      --num_seeds "$NUM_SEEDS"
      --num_recent_tildes "$NUM_RECENT_TILDES"
    )
    [ "$NO_VIZ" = true ] && PH1_ARGS+=( --no_viz )
    [ "$NO_RESET" = true ] && PH1_ARGS+=( --no_reset )
    [ -n "$PAIRING_POLICY" ] && PH1_ARGS+=( --pairing_policy "$PAIRING_POLICY" )

    cd "$REPO_ROOT" || exit 1
    export PYTHONPATH="$REPO_ROOT"
    python overcooked_v2_experiments/ppo/utils/ph1_recent_cross_eval.py "${PH1_ARGS[@]}"

    echo ""
    echo "PH1 recent-tilde cross evaluation complete!"
    echo "Results saved in: $DIRECTORY"
    exit 0
fi

# Build command arguments
ARGS=( --d "$DIRECTORY" --seed "$SEED" --num_seeds "$NUM_SEEDS" )

[ "$ALL_CKPT" = true ] && ARGS+=( --all_ckpt )
[ "$CROSS" = true ] && ARGS+=( --cross )
[ "$ALL" = true ] && ARGS+=( --all )
[ "$NO_VIZ" = true ] && ARGS+=( --no_viz )
[ "$NO_RESET" = true ] && ARGS+=( --no_reset )
[ -n "$PAIRING_POLICY" ] && ARGS+=( --pairing_policy "$PAIRING_POLICY" )
[ "$LATENT_ANALYSIS" = true ] && ARGS+=( --latent_analysis )
[ "$VALUE_ANALYSIS" = true ] && ARGS+=( --value_analysis )

# Change to experiments directory
cd "$REPO_ROOT" || exit 1

# PYTHONPATH를 experiments-stablock만으로 설정하여 다른 experiments 폴더 코드 사용 방지
export PYTHONPATH="$REPO_ROOT"

# Run visualization
python overcooked_v2_experiments/ppo/utils/visualize_ppo.py "${ARGS[@]}"

echo ""
echo "Visualization complete!"
echo "Results saved in: $DIRECTORY"





# # experiments 폴더에서 실행
# cd /home/mlic/mingukang/ex-overcookedv2/experiments

# # 기본 사용 (self-play, 10 seeds)
# ./run_visualize.sh --gpu 0 --dir runs/20251107-064612_a8lag2vo_test_time_simple_avs-2

# # Cross-play 평가 (20 seeds)
# ./run_visualize.sh --gpu 1 --dir runs/cramped_room_run --num_seeds 20 --cross

# # 비디오 생성 없이 메트릭만 계산 (500 seeds)
# ./run_visualize.sh --gpu 0 --dir runs/my_run --num_seeds 500 --no_viz --cross

# # 모든 체크포인트 평가 (self-play만)
# ./run_visualize.sh --gpu 2 --dir runs/my_run --all_ckpt

# # Self-play와 Cross-play 둘 다
# ./run_visualize.sh --gpu 0 --dir runs/my_run --all --num_seeds 100 --no_viz

# # 특정 정책과 페어링하여 평가
# ./run_visualize.sh --gpu 0 --dir runs/my_run --cross --pairing_policy 0 --num_seeds 50
