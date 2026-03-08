#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")" || exit 1

DRY_RUN=false
FORWARD_ARGS=()

# ---------------------------------------------------------------------
# Preset batch mode (used when CLI args are empty):
# - Auto-discover runs under AUTO_RUNS_DIR.
# - Include run dirs with date prefix >= AUTO_DATE_FROM_YYYYMMDD.
# - Or, if AUTO_AFTER_RUN is set, include run dirs strictly after that run
#   in sorted directory order.
# - Execute --eval-analysis in parallel batches of AUTO_BATCH_SIZE.
# ---------------------------------------------------------------------
AUTO_DISCOVER_PRESETS=false
AUTO_RUNS_DIR="../runs"
AUTO_DATE_FROM_YYYYMMDD=20260301
AUTO_AFTER_RUN=""
AUTO_BEFORE_RUN=""
AUTO_FROM_RUN="20260303-000027_31hadyvh_grounded_coord_simple_e3t_ph2"
AUTO_TO_RUN="20260303-192651_a0t7uvpt_grounded_coord_simple_e3t_ph2_ct1_e0p2_o20_s3"
AUTO_GPU_IDX=0
AUTO_BATCH_SIZE=3
AUTO_PRESET_EVAL_MODE="cross-play" # cross-play | eval-analysis | eval-viz
AUTO_PH1_CROSS_NUM_SEEDS=3
AUTO_PH1_CROSS_NUM_RECENT_TILDES=10
AUTO_PH2_CROSS_NUM_SEEDS=5

# Optional manual fallback commands (used only when AUTO_DISCOVER_PRESETS=false).
PRESET_FACTORY_COMMANDS=(
  "./run_visualize.sh --gpu 0 --dir runs/20260306-065629_nrvvrr3q_asymm_advantages_e3t_ph2_ct0_e0p2_o10_s2 --cross --num_seeds 5"
  "./run_visualize.sh --gpu 1 --dir runs/20260306-114435_zb151oi7_coord_ring_e3t_ph2_ct0_e0p2_o10_s2 --cross --num_seeds 5"
  "./run_visualize.sh --gpu 2 --dir runs/20260306-160646_r5awqode_cramped_room_e3t_ph2_ct0_e0p2_o10_s2 --cross --num_seeds 5"
  "./run_visualize.sh --gpu 3 --dir runs/20260306-201807_u99vy5ws_forced_coord_e3t_ph2_ct0_e0p2_o10_s2 --cross --num_seeds 5"
)

usage() {
  cat <<EOF
Usage:
  $0                            # run preset batch (auto-discovered by default)
  $0 <run_visualize.sh options...>

Factory options:
  --dry-run                      Print commands without executing
  --cross-play                   Preset batch mode selector
  --eval-analysis                Preset batch mode selector (default)
  --eval-viz                     Preset batch mode selector
  --after-run <run_dir_name>     Use runs strictly after this folder (sorted order)
  --before-run <run_dir_name>    Use runs up to this folder (inclusive, sorted order)

Preset batch defaults:
  runs dir:                      $AUTO_RUNS_DIR
  date from:                     $AUTO_DATE_FROM_YYYYMMDD
  after run:                     ${AUTO_AFTER_RUN:-<disabled>}
  before run:                    ${AUTO_BEFORE_RUN:-<disabled>}
  from run (inclusive):          ${AUTO_FROM_RUN:-<disabled>}
  to run (inclusive):            ${AUTO_TO_RUN:-<disabled>}
  gpu:                           $AUTO_GPU_IDX
  parallel batch size:           $AUTO_BATCH_SIZE
  preset eval mode:              $AUTO_PRESET_EVAL_MODE
  ph1 cross:                     num_seeds=$AUTO_PH1_CROSS_NUM_SEEDS num_recent_tildes=$AUTO_PH1_CROSS_NUM_RECENT_TILDES
  ph2 cross:                     num_seeds=$AUTO_PH2_CROSS_NUM_SEEDS

Forwarded run_visualize.sh options (examples):
  --cross --num_seeds N          Standard visualize/eval mode
  --latent_analysis              Latent analysis mode
  --value_analysis               Value analysis mode
  --eval-analysis                Offline eval analysis csv mode (video disabled)
                                 Equivalent output: <run_dir>/eval/offline_eval_analysis.csv
  --eval-viz                     Offline eval final-checkpoint video mode
                                 Equivalent output: <run_dir>/eval/video/*.gif

Examples:
  $0 --cross-play                # run preset batch with phase-aware cross-play
  $0 --eval-viz                  # run preset batch with eval-viz
  $0 --eval-analysis --after-run 20260216-104647_04d61n7i_grounded_coord_simple_e3t_ph1
  $0 --eval-analysis --before-run 20260219-114500_abcd1234_counter_circuit_e3t_ph1
  $0 --dry-run --eval-analysis   # print preset eval-analysis commands
  $0 --gpu 0 --dir runs/20260128-015425_9usv82dt_counter_circuit_fcp --eval-analysis
  $0 --gpu 0 --dir runs/20260210-143629_wsvfmd7x_counter_circuit_e3t_ph1 --cross --num_seeds 5 --no_viz
EOF
}

build_auto_preset_commands() {
  local runs_root="$AUTO_RUNS_DIR"
  local date_from="$AUTO_DATE_FROM_YYYYMMDD"
  local after_run="$AUTO_AFTER_RUN"
  local before_run="$AUTO_BEFORE_RUN"
  local from_run="$AUTO_FROM_RUN"
  local to_run="$AUTO_TO_RUN"
  local gpu_idx="$AUTO_GPU_IDX"
  local preset_mode="$AUTO_PRESET_EVAL_MODE"

  if [[ ! -d "$runs_root" ]]; then
    return 0
  fi

  local use_after=false
  local use_before=false
  local use_from=false
  local use_to=false
  local after_found=false
  local before_found=false
  local from_found=false
  local to_found=false
  if [[ -n "$after_run" ]]; then
    use_after=true
  fi
  if [[ -n "$before_run" ]]; then
    use_before=true
  fi
  if [[ -n "$from_run" ]]; then
    use_from=true
  fi
  if [[ -n "$to_run" ]]; then
    use_to=true
  fi

  local run_dir=""
  local run_base=""
  local run_date=""
  local run_base_lc=""
  while IFS= read -r run_dir; do
    run_base="$(basename "$run_dir")"
    if [[ "$use_from" == "true" ]]; then
      if [[ "$from_found" == "false" ]]; then
        if [[ "$run_base" == "$from_run" ]]; then
          from_found=true
        else
          continue
        fi
      fi
    elif [[ "$use_after" == "true" ]]; then
      if [[ "$after_found" == "false" ]]; then
        if [[ "$run_base" == "$after_run" ]]; then
          after_found=true
        fi
        continue
      fi
    else
      run_date="${run_base:0:8}"
      if [[ ! "$run_date" =~ ^[0-9]{8}$ ]] || (( run_date < date_from )); then
        continue
      fi
    fi

    run_base_lc="$(echo "$run_base" | tr '[:upper:]' '[:lower:]')"
    case "$preset_mode" in
      cross-play)
        if [[ "$run_base_lc" == *"ph1"* ]]; then
          echo "./run_visualize.sh --gpu $gpu_idx --dir runs/$run_base --cross --num_seeds $AUTO_PH1_CROSS_NUM_SEEDS --num_recent_tildes $AUTO_PH1_CROSS_NUM_RECENT_TILDES"
        elif [[ "$run_base_lc" == *"ph2"* ]]; then
          echo "./run_visualize.sh --gpu $gpu_idx --dir runs/$run_base --cross --num_seeds $AUTO_PH2_CROSS_NUM_SEEDS"
        fi
        ;;
      eval-viz)
        if [[ "$run_base_lc" == *"ph1"* ]]; then
          echo "./run_visualize.sh --gpu $gpu_idx --dir runs/$run_base --$preset_mode"
        fi
        ;;
      eval-analysis)
        echo "./run_visualize.sh --gpu $gpu_idx --dir runs/$run_base --$preset_mode"
        ;;
      *)
        echo "[WARN] unsupported AUTO_PRESET_EVAL_MODE: $preset_mode" >&2
        return 0
        ;;
    esac

    if [[ "$use_to" == "true" && "$run_base" == "$to_run" ]]; then
      to_found=true
      break
    fi
    if [[ "$use_before" == "true" && "$run_base" == "$before_run" ]]; then
      before_found=true
      break
    fi
  done < <(find "$runs_root" -mindepth 1 -maxdepth 1 -type d | sort)

  if [[ "$use_after" == "true" && "$after_found" == "false" ]]; then
    echo "[WARN] --after-run target not found: $after_run" >&2
  fi
  if [[ "$use_from" == "true" && "$from_found" == "false" ]]; then
    echo "[WARN] from-run target not found: $from_run" >&2
  fi
  if [[ "$use_before" == "true" && "$before_found" == "false" ]]; then
    echo "[WARN] --before-run target not found: $before_run" >&2
  fi
  if [[ "$use_to" == "true" && "$to_found" == "false" ]]; then
    echo "[WARN] to-run target not found: $to_run" >&2
  fi
}

run_preset_factory_commands() {
  local total=0
  local launched=0
  local completed=0
  local success=0
  local failed=0
  local batch_size="$AUTO_BATCH_SIZE"
  local -a commands=()
  local -a pids=()
  local -a pid_cmds=()

  if [[ ! -f "./run_visualize.sh" ]]; then
    echo "[ERROR] run_visualize.sh not found in $(pwd)"
    exit 1
  fi

  if [[ "$AUTO_DISCOVER_PRESETS" == "true" ]]; then
    mapfile -t commands < <(build_auto_preset_commands)
  else
    # Accept both styles:
    # 1) each entry is a full command string (recommended)
    # 2) accidentally tokenized entries (unquoted command in array)
    mapfile -t commands < <(normalize_manual_preset_commands)
  fi

  echo "=== Preset Factory Commands ==="
  echo "[INFO] auto_discover=$AUTO_DISCOVER_PRESETS runs_dir=$AUTO_RUNS_DIR date_from=$AUTO_DATE_FROM_YYYYMMDD after_run=${AUTO_AFTER_RUN:-<disabled>} before_run=${AUTO_BEFORE_RUN:-<disabled>} from_run=${AUTO_FROM_RUN:-<disabled>} to_run=${AUTO_TO_RUN:-<disabled>} batch_size=$batch_size mode=$AUTO_PRESET_EVAL_MODE"

  for cmd in "${commands[@]}"; do
    [[ -z "$cmd" ]] && continue
    total=$((total + 1))
    echo "[CMD] $cmd"
    if [[ "$DRY_RUN" != "true" ]]; then
      bash -lc "$cmd" &
      pids+=("$!")
      pid_cmds+=("$cmd")
      launched=$((launched + 1))

      if [[ ${#pids[@]} -ge "$batch_size" ]]; then
        for i in "${!pids[@]}"; do
          if wait "${pids[$i]}"; then
            success=$((success + 1))
          else
            failed=$((failed + 1))
            echo "[WARN] command failed: ${pid_cmds[$i]}"
          fi
          completed=$((completed + 1))
        done
        pids=()
        pid_cmds=()
      fi
    fi
  done

  if [[ "$DRY_RUN" != "true" && ${#pids[@]} -gt 0 ]]; then
    for i in "${!pids[@]}"; do
      if wait "${pids[$i]}"; then
        success=$((success + 1))
      else
        failed=$((failed + 1))
        echo "[WARN] command failed: ${pid_cmds[$i]}"
      fi
      completed=$((completed + 1))
    done
  fi

  echo "==============================="
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "[INFO] preset commands: total=$total (dry-run)"
  else
    echo "[INFO] preset commands: total=$total launched=$launched completed=$completed success=$success failed=$failed"
  fi
  if [[ "$total" -eq 0 ]]; then
    echo "[INFO] no preset commands found."
  fi
}

normalize_manual_preset_commands() {
  local -a raw=("${PRESET_FACTORY_COMMANDS[@]}")
  local token=""
  local has_split_tokens=false

  for token in "${raw[@]}"; do
    if [[ "$token" == "./run_visualize.sh" || "$token" == "run_visualize.sh" ]]; then
      has_split_tokens=true
      break
    fi
  done

  if [[ "$has_split_tokens" != "true" ]]; then
    printf '%s\n' "${raw[@]}"
    return
  fi

  local current_cmd=""
  for token in "${raw[@]}"; do
    [[ -z "$token" ]] && continue
    if [[ "$token" == "./run_visualize.sh" || "$token" == "run_visualize.sh" ]]; then
      if [[ -n "$current_cmd" ]]; then
        printf '%s\n' "$current_cmd"
      fi
      current_cmd="$token"
    else
      if [[ -z "$current_cmd" ]]; then
        current_cmd="$token"
      else
        current_cmd+=" $token"
      fi
    fi
  done
  if [[ -n "$current_cmd" ]]; then
    printf '%s\n' "$current_cmd"
  fi
}

if [[ $# -eq 0 ]]; then
  run_preset_factory_commands
  exit 0
fi

# If only factory flags are given, run preset batch mode.
factory_only=true
idx=1
while [[ $idx -le $# ]]; do
  arg="${!idx}"
  case "$arg" in
    --dry-run|--cross-play|--eval-analysis|--eval-viz)
      ;;
    --after-run)
      idx=$((idx + 1))
      if [[ $idx -gt $# ]]; then
        echo "[ERROR] --after-run requires a run directory name"
        exit 1
      fi
      ;;
    --before-run)
      idx=$((idx + 1))
      if [[ $idx -gt $# ]]; then
        echo "[ERROR] --before-run requires a run directory name"
        exit 1
      fi
      ;;
    *)
      factory_only=false
      break
      ;;
  esac
  idx=$((idx + 1))
done

if [[ "$factory_only" == "true" ]]; then
  boundary_overridden=false
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dry-run)
        DRY_RUN=true
        shift
        ;;
      --eval-analysis)
        AUTO_PRESET_EVAL_MODE="eval-analysis"
        shift
        ;;
      --cross-play)
        AUTO_PRESET_EVAL_MODE="cross-play"
        shift
        ;;
      --eval-viz)
        AUTO_PRESET_EVAL_MODE="eval-viz"
        shift
        ;;
      --after-run)
        if [[ "$boundary_overridden" == "false" ]]; then
          AUTO_AFTER_RUN=""
          AUTO_BEFORE_RUN=""
          boundary_overridden=true
        fi
        shift
        if [[ $# -eq 0 ]]; then
          echo "[ERROR] --after-run requires a run directory name"
          exit 1
        fi
        AUTO_AFTER_RUN="$1"
        shift
        ;;
      --before-run)
        if [[ "$boundary_overridden" == "false" ]]; then
          AUTO_AFTER_RUN=""
          AUTO_BEFORE_RUN=""
          boundary_overridden=true
        fi
        shift
        if [[ $# -eq 0 ]]; then
          echo "[ERROR] --before-run requires a run directory name"
          exit 1
        fi
        AUTO_BEFORE_RUN="$1"
        shift
        ;;
    esac
  done
  run_preset_factory_commands
  exit 0
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --eval|--training-eval|--eval_analysis)
      echo "[ERROR] $1 is removed from viz_factory.sh."
      echo "        Use run_visualize.sh --eval-analysis instead."
      echo "        Example: ./run_visualize.sh --gpu 0 --dir runs/<run_dir> --eval-analysis"
      exit 1
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      FORWARD_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ! -f "./run_visualize.sh" ]]; then
  echo "[ERROR] run_visualize.sh not found in $(pwd)"
  exit 1
fi

if [[ ${#FORWARD_ARGS[@]} -eq 0 ]]; then
  echo "[ERROR] no run_visualize.sh arguments given"
  usage
  exit 1
fi

cmd=(./run_visualize.sh "${FORWARD_ARGS[@]}")
echo "[CMD] ${cmd[*]}"
if [[ "$DRY_RUN" != "true" ]]; then
  "${cmd[@]}"
fi
