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
AUTO_DATE_FROM_YYYYMMDD=20260220
AUTO_AFTER_RUN="20260223-063120_cutvvj25_counter_circuit_e3t_ph2"
AUTO_BEFORE_RUN="20260223-234643_aoeewekf_counter_circuit_e3t_ph1"
AUTO_GPU_IDX=0
AUTO_BATCH_SIZE=3
AUTO_PRESET_EVAL_MODE="eval-analysis" # eval-analysis | eval-viz

# Optional manual fallback commands (used only when AUTO_DISCOVER_PRESETS=false).
PRESET_FACTORY_COMMANDS=(
  # "./run_visualize.sh --gpu 5 --dir runs/20260225-090922_nqb7lahb_grounded_coord_simple_e3t_ph1 --cross --num_seeds 3 --num_recent_tildes 10 --no_viz"
  # "./run_visualize.sh --gpu 6 --dir runs/20260225-121208_377o3otw_grounded_coord_simple_e3t_ph1 --cross --num_seeds 3 --num_recent_tildes 10 --no_viz"
  # "./run_visualize.sh --gpu 7 --dir runs/20260225-151100_ce3tscrx_counter_circuit_e3t_ph1 --cross --num_seeds 3 --num_recent_tildes 10 --no_viz"
  # "./run_visualize.sh --gpu 5 --dir runs/20260225-175719_udtk6oeb_counter_circuit_e3t_ph1 --cross --num_seeds 3 --num_recent_tildes 10 --no_viz"
  # "./run_visualize.sh --gpu 6 --dir runs/20260225-204306_xuwe4ki5_grounded_coord_simple_e3t_ph1 --cross --num_seeds 3 --num_recent_tildes 10 --no_viz"
  # "./run_visualize.sh --gpu 7 --dir runs/20260225-234304_ltu9h4y3_grounded_coord_simple_e3t_ph1 --cross --num_seeds 3 --num_recent_tildes 10 --no_viz"
  "./run_visualize.sh --gpu 5 --dir runs/20260226-024332_9ph7wcq6_counter_circuit_e3t_ph1 --cross --num_seeds 3 --num_recent_tildes 10 --no_viz"
  "./run_visualize.sh --gpu 6 --dir runs/20260226-053020_dnrcpb5u_counter_circuit_e3t_ph1 --cross --num_seeds 3 --num_recent_tildes 10 --no_viz"
  "./run_visualize.sh --gpu 7 --dir runs/20260226-090144_cn5qfm72_grounded_coord_simple_e3t_ph1 --cross --num_seeds 3 --num_recent_tildes 10 --no_viz"
  "./run_visualize.sh --gpu 5 --dir runs/20260226-115742_j3ni629d_grounded_coord_simple_e3t_ph1 --cross --num_seeds 3 --num_recent_tildes 10 --no_viz"
  "./run_visualize.sh --gpu 6 --dir runs/20260226-145719_5i1a87z0_counter_circuit_e3t_ph1 --cross --num_seeds 3 --num_recent_tildes 10 --no_viz"
  "./run_visualize.sh --gpu 7 --dir runs/20260226-174415_ueh0npkk_counter_circuit_e3t_ph1 --cross --num_seeds 3 --num_recent_tildes 10 --no_viz"
  "./run_visualize.sh --gpu 5 --dir runs/20260226-203018_t168x8so_grounded_coord_simple_e3t_ph1 --cross --num_seeds 3 --num_recent_tildes 10 --no_viz"
  "./run_visualize.sh --gpu 6 --dir runs/20260226-232657_sx5l64zm_grounded_coord_simple_e3t_ph1 --cross --num_seeds 3 --num_recent_tildes 10 --no_viz"
)

usage() {
  cat <<EOF
Usage:
  $0                            # run preset batch (auto-discovered by default)
  $0 <run_visualize.sh options...>

Factory options:
  --dry-run                      Print commands without executing
  --eval-analysis                Preset batch mode selector (default)
  --eval-viz                     Preset batch mode selector
  --after-run <run_dir_name>     Use runs strictly after this folder (sorted order)
  --before-run <run_dir_name>    Use runs up to this folder (inclusive, sorted order)

Preset batch defaults:
  runs dir:                      $AUTO_RUNS_DIR
  date from:                     $AUTO_DATE_FROM_YYYYMMDD
  after run:                     ${AUTO_AFTER_RUN:-<disabled>}
  before run:                    ${AUTO_BEFORE_RUN:-<disabled>}
  gpu:                           $AUTO_GPU_IDX
  parallel batch size:           $AUTO_BATCH_SIZE
  preset eval mode:              $AUTO_PRESET_EVAL_MODE

Forwarded run_visualize.sh options (examples):
  --cross --num_seeds N          Standard visualize/eval mode
  --latent_analysis              Latent analysis mode
  --value_analysis               Value analysis mode
  --eval-analysis                Offline eval analysis csv mode (video disabled)
                                 Equivalent output: <run_dir>/eval/offline_eval_analysis.csv
  --eval-viz                     Offline eval final-checkpoint video mode
                                 Equivalent output: <run_dir>/eval/video/*.gif

Examples:
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
  local gpu_idx="$AUTO_GPU_IDX"
  local preset_mode="$AUTO_PRESET_EVAL_MODE"

  if [[ ! -d "$runs_root" ]]; then
    return 0
  fi

  local use_after=false
  local use_before=false
  local after_found=false
  local before_found=false
  if [[ -n "$after_run" ]]; then
    use_after=true
  fi
  if [[ -n "$before_run" ]]; then
    use_before=true
  fi

  local run_dir=""
  local run_base=""
  local run_date=""
  local run_base_lc=""
  while IFS= read -r run_dir; do
    run_base="$(basename "$run_dir")"
    if [[ "$use_after" == "true" ]]; then
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
    if [[ "$preset_mode" == "eval-viz" && "$run_base_lc" != *"ph1"* ]]; then
      continue
    fi
    echo "./run_visualize.sh --gpu $gpu_idx --dir runs/$run_base --$preset_mode"

    if [[ "$use_before" == "true" && "$run_base" == "$before_run" ]]; then
      before_found=true
      break
    fi
  done < <(find "$runs_root" -mindepth 1 -maxdepth 1 -type d | sort)

  if [[ "$use_after" == "true" && "$after_found" == "false" ]]; then
    echo "[WARN] --after-run target not found: $after_run" >&2
  fi
  if [[ "$use_before" == "true" && "$before_found" == "false" ]]; then
    echo "[WARN] --before-run target not found: $before_run" >&2
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
  echo "[INFO] auto_discover=$AUTO_DISCOVER_PRESETS runs_dir=$AUTO_RUNS_DIR date_from=$AUTO_DATE_FROM_YYYYMMDD after_run=${AUTO_AFTER_RUN:-<disabled>} before_run=${AUTO_BEFORE_RUN:-<disabled>} batch_size=$batch_size mode=$AUTO_PRESET_EVAL_MODE"

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
    --dry-run|--eval-analysis|--eval-viz)
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

