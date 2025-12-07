# ------------------------------------------------------------------------------
# run_overcooked.sh — JAX-AHT OvercookedV2 PPO launcher (uv/conda 무관)
# ------------------------------------------------------------------------------

#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# 0) 기본값 설정 (환경변수로 덮어쓰기 가능)
# ==============================================================================

: "${CUDA_VISIBLE_DEVICES:=0}"                 # GPU 할당 (콤마 구분: 예 0,1)
: "${WANDB_PROJECT:=jax-aht}"                  # W&B 프로젝트명
: "${WANDB_ENTITY:=tatalintelli-university-of-seoul}"
: "${NUM_SEEDS:=10}"                           # 실험 시드 수
: "${NUM_ITERATIONS:=1}"

# 환경/실험 프리셋
: "${ENV_GROUP:=original}"                     # 예: original, grounded_coord_simple, test_time_wide
: "${LAYOUT:=cramped_room}"                   # ENV_GROUP=original 일 때 사용
: "${EXPERIMENT:=cnn}"                         # 예: cnn, rnn-op, rnn-sa, rnn-fcp, panic-sp

# Panic (partner random action) defaults
: "${PANIC_ENABLED:=0}"        # 1 => enable panic window
: "${PANIC_START_STEP:=50}"    # default start step within episode
: "${PANIC_DURATION:=30}"      # default duration (steps)

# E3T (Mixture Partner Policy) defaults
: "${E3T_EPSILON:=0.05}"       # 파트너 무작위 행동 확률

# JAX 메모리 설정
: "${XLA_PYTHON_CLIENT_PREALLOCATE:=false}"    # 메모리 선할당 방지
: "${XLA_PYTHON_CLIENT_MEM_FRACTION:=0.7}"     # 0.0~1.0 비율 (너무 낮으면 성능 저하 가능)

# FCP_DEVICE 설정 (기본값: cpu)
: "${FCP_DEVICE:=cpu}"

# XLA_FLAGS: 기본 CUDA data dir 설정
: "${XLA_FLAGS:=--xla_gpu_cuda_data_dir=${CUDA_HOME:-/usr/local/cuda-12.2}}"


# cuPTI 경로 (CUDA 12.2 기준)
if [ -d "/usr/local/cuda-12.2/extras/CUPTI/lib64" ]; then
  export LD_LIBRARY_PATH="/usr/local/cuda-12.2/lib64:/usr/local/cuda-12.2/extras/CUPTI/lib64:${LD_LIBRARY_PATH:-}"
fi

export XLA_PYTHON_CLIENT_PREALLOCATE
export XLA_PYTHON_CLIENT_MEM_FRACTION
export XLA_FLAGS

# 프로젝트 전용 Python 가상환경 bin 경로 우선
export PATH="/home/mlic/mingukang/ex-overcookedv2/overcookedv2/bin:$PATH"

# ==============================================================================
# GPU / CUDA 관련 환경 변수
# ==============================================================================

# CUDA_VISIBLE_DEVICES 기본값 재확인 (필요 시 사용자 지정 가능)
: "${CUDA_VISIBLE_DEVICES:=0}"

# JAX 플랫폼 설정: 비워두면 자동 감지, 강제 CPU는 --cpu 또는 JAX_PLATFORMS=cpu
: "${JAX_PLATFORMS:=}"

# CUDA 경로 설정 (필요 시 사용자 지정 가능)
: "${CUDA_HOME:=/usr/local/cuda-12.2}"

# PTX 경고 억제 토글 및 패턴
: "${SUPPRESS_PTX_WARN:=1}"

# '+ptx89 ... not a recognized feature for this target' 류의 소음 로그를 숨기기 위한 정규식
PTX_WARN_RE="(\+)?ptx[0-9]+.*not a recognized feature for this target|not a recognized feature for this target.*(\+)?ptx[0-9]+"

# PTX 경고 필터 함수: 토글에 따라 해당 라인을 제거하거나 그대로 통과
filter_ptx() {
  if [[ "${SUPPRESS_PTX_WARN}" == "1" ]]; then
    # PTX 경고 및 잔여 "(ignoring feature)" 라인을 제거
    grep -v -E "${PTX_WARN_RE}|\(ignoring feature\)\s*$" || true
  else
    cat
  fi
}

# 시스템 CUDA 라이브러리 사용 여부 (기본 0: jaxlib 내 번들 사용)
: "${USE_SYSTEM_CUDA_LIBS:=0}"
if [[ "${USE_SYSTEM_CUDA_LIBS}" == "1" ]]; then
  # LD_LIBRARY_PATH에 CUDA 경로 추가 (권장하지 않음: 충돌 가능)
  if [ -d "$CUDA_HOME/lib64" ]; then
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:${LD_LIBRARY_PATH:-}"
  fi
  # LD_LIBRARY_PATH 정리 (중복 제거)
  CLEANED_LD_LIBRARY_PATH=$(
    echo "${LD_LIBRARY_PATH:-}" \
      | tr ':' '\n' \
      | awk 'NF && !seen[$0]++' \
      | tr '\n' ':'
  )
  export LD_LIBRARY_PATH="$CLEANED_LD_LIBRARY_PATH"
  echo "[INFO] Using system CUDA libs (USE_SYSTEM_CUDA_LIBS=1)"
  echo "[INFO] Cleaned LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
else
  echo "[INFO] Skipping system CUDA libs (USE_SYSTEM_CUDA_LIBS=0)."
  echo "       We'll unset LD_LIBRARY_PATH and XLA_FLAGS at Python launch to avoid lib conflicts."
fi

# 공통 환경 변수 export
export JAX_PLATFORMS
export CUDA_HOME
export LD_LIBRARY_PATH

# GPU 설정 확인 메시지
echo "[INFO] JAX 플랫폼: JAX_PLATFORMS=$JAX_PLATFORMS"
echo "[INFO] CUDA 경로: CUDA_HOME=$CUDA_HOME"
echo "[INFO] LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-}"

# ==============================================================================
# 1) 스크립트 위치/경로 정리
# ==============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ==============================================================================
# 2) Weights & Biases API 키 로드
# ==============================================================================

WANDB_KEY_FILE="$SCRIPT_DIR/../wandb_info/wandb_api_key"
if [[ -f "$WANDB_KEY_FILE" ]]; then
  export WANDB_API_KEY="$(<"$WANDB_KEY_FILE")"
else
  echo "[ERR] W&B API key file not found: $WANDB_KEY_FILE"
  echo "      Create the file and put your API key in a single line."
  exit 1
fi

# ==============================================================================
# 3) 선택적 인자 파싱
#   예) ./run_overcooked.sh --env grounded_coord_simple --exp rnn-op \
#           --seeds 10 --tags 'zsc,aht' --notes 'ablation-1'
# ==============================================================================

NOTES=""
TAGS=""
SEEDS_EXPLICIT="0"
ITERATIONS_OVERRIDE=$NUM_ITERATIONS

FCP_DIR=""                    # FCP population 디렉토리(선택)
ENV_DEVICE=""                 # env를 CPU/GPU 어디에 둘지: cpu|gpu (기본: 자동)
CAST_OBS_BF16="0"             # 관측을 bf16으로 캐스팅하여 메모리 절감
MODEL_NUM_ENVS_OVERRIDE=""    # model.NUM_ENVS override
MODEL_NUM_STEPS_OVERRIDE=""   # model.NUM_STEPS override
CONF_PROFILE=""
CONF_THRESHOLD=""
CONF_COOLDOWN=""
CONF_TARGET=""
CONF_N_THRESHOLD=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)       export CUDA_VISIBLE_DEVICES="$2"; shift 2;;
    --seeds)      NUM_SEEDS="$2"; SEEDS_EXPLICIT="1"; shift 2;;
    --env)        ENV_GROUP="$2"; shift 2;;
    --layout)     LAYOUT="$2"; shift 2;;
    --exp|--experiment) EXPERIMENT="$2"; shift 2;;
    --project)    WANDB_PROJECT="$2"; shift 2;;
    --entity)     WANDB_ENTITY="$2"; shift 2;;
    --notes)      NOTES="$2"; shift 2;;
    --tags)       TAGS="$2"; shift 2;;  # "a,b" 또는 "a b"
    --iters|--iterations) ITERATIONS_OVERRIDE="$2"; shift 2;;
    --fcp)        FCP_DIR="$2"; shift 2;;  # 예: --fcp runs/fcp_populations/grounded_coord_simple
    --cpu)        export JAX_PLATFORMS=cpu; shift 1;;
    --env-device) ENV_DEVICE="$2"; shift 2;;     # cpu|gpu
    --bf16-obs)   CAST_OBS_BF16="1"; shift 1;;
    --nenvs)      MODEL_NUM_ENVS_OVERRIDE="$2"; shift 2;;
  --nsteps)     MODEL_NUM_STEPS_OVERRIDE="$2"; shift 2;;
  --panic)      PANIC_ENABLED=1; shift 1;;
  --panic-start) PANIC_START_STEP="$2"; shift 2;;
  --panic-duration) PANIC_DURATION="$2"; shift 2;;
    --conf-profile) CONF_PROFILE="$2"; shift 2;;
    --conf-threshold) CONF_THRESHOLD="$2"; shift 2;;
    --conf-steps) CONF_COOLDOWN="$2"; shift 2;;
    --conf-target) CONF_TARGET="$2"; shift 2;;
    --conf-n-threshold) CONF_N_THRESHOLD="$2"; shift 2;;
    --e3t-epsilon) E3T_EPSILON="$2"; shift 2;;
    --mem-frac)   XLA_PYTHON_CLIENT_MEM_FRACTION="$2"; shift 2;;
    --fcp-device) FCP_DEVICE="$2"; shift 2 ;;
    --)           shift; break;;
    *)            echo "[WARN] Unknown arg: $1"; shift 1;;
  esac
done

export WANDB_PROJECT
export WANDB_ENTITY

# ==============================================================================
# 4) 실행 정보 로깅
# ==============================================================================

RUN_NAME="${EXPERIMENT}"
CONF_PROFILE_PRETTY=""
if [[ -n "$CONF_PROFILE" ]]; then
  CONF_PROFILE_PRETTY=${CONF_PROFILE//_/-}
  # EXPERIMENT가 *-sp라면 suffix를 교체 (예: rnn-sp -> rnn-sp-uc)
  if [[ "$EXPERIMENT" == *"-sp" ]]; then
    RUN_NAME="${EXPERIMENT%-sp}-${CONF_PROFILE_PRETTY}"
  else
    RUN_NAME="${EXPERIMENT}-${CONF_PROFILE_PRETTY}"
  fi
fi

echo "==============================================================="
echo "  Run Name     : $RUN_NAME"
echo "  GPUs         : $CUDA_VISIBLE_DEVICES"

if [[ "$EXPERIMENT" == "rnn-fcp" || -n "$FCP_DIR" ]]; then
  if [[ "$SEEDS_EXPLICIT" == "1" ]]; then
    echo "  Seeds        : $NUM_SEEDS (explicit)"
  else
    echo "  Seeds        : (cfg default for FCP, 1)"
  fi
else
  echo "  Seeds        : $NUM_SEEDS"
fi

echo "  Env Group    : $ENV_GROUP"
echo "  Layout       : $LAYOUT"
echo "  Experiment   : $EXPERIMENT"
if [[ "$PANIC_ENABLED" == "1" ]]; then
  echo "  Panic        : ENABLED (start=$PANIC_START_STEP, duration=$PANIC_DURATION)"
else
  echo "  Panic        : disabled"
fi
echo "  W&B Project  : $WANDB_PROJECT"
echo "  W&B Entity   : $WANDB_ENTITY"

if [[ -n "$ITERATIONS_OVERRIDE" ]]; then
  echo "  Iterations   : $ITERATIONS_OVERRIDE (override)"
else
  echo "  Iterations   : (cfg default if defined)"
fi

[[ -n "$NOTES" ]]                  && echo "  Notes        : $NOTES"
[[ -n "$TAGS"  ]]                  && echo "  Extra Tags   : $TAGS"
[[ -n "$FCP_DIR" ]]                && echo "  FCP Pop Dir  : $FCP_DIR"
[[ -n "$ENV_DEVICE" ]]             && echo "  Env Device   : $ENV_DEVICE"
[[ "$CAST_OBS_BF16" == "1" ]]      && echo "  Obs DType    : bfloat16 (CAST_OBS_BF16)"
[[ -n "$MODEL_NUM_ENVS_OVERRIDE" ]]   && echo "  NUM_ENVS     : $MODEL_NUM_ENVS_OVERRIDE (override)"
[[ -n "$MODEL_NUM_STEPS_OVERRIDE" ]] && echo "  NUM_STEPS    : $MODEL_NUM_STEPS_OVERRIDE (override)"
[[ -n "$CONF_PROFILE" ]]             && echo "  Confidence   : profile=$CONF_PROFILE_PRETTY ($CONF_PROFILE)"
[[ -n "$CONF_THRESHOLD" ]]           && echo "  ConfThresh   : $CONF_THRESHOLD"
[[ -n "$CONF_COOLDOWN" ]]            && echo "  ConfCooldown : $CONF_COOLDOWN"
[[ -n "$CONF_TARGET" ]]              && echo "  ConfTarget   : $CONF_TARGET"

echo "==============================================================="

# ==============================================================================
# 5) 파이썬 진입 전 빠른 진단은 가상환경 활성화 이후로 이동
# ==============================================================================

export CUDA_VISIBLE_DEVICES

# ==============================================================================
# 0.5) 선택 GPU 메모리 상태 표시 및 경고
# ==============================================================================

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[INFO] nvidia-smi (selected GPUs: $CUDA_VISIBLE_DEVICES)"
  nvidia-smi \
    --query-gpu=index,name,memory.total,memory.used,memory.free \
    --format=csv,noheader \
    | awk -v sel="${CUDA_VISIBLE_DEVICES}" '
        BEGIN{ split(sel, a, ","); for(i in a) sel_idx[a[i]]=1 }
        {
          split($0, f, ", ");
          idx=f[1];
          if (idx in sel_idx)
            print "  GPU " idx ": " f[2] ", total=" f[3] ", used=" f[4] ", free=" f[5];
        }'

  FREE=$(
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
      | awk -v sel="${CUDA_VISIBLE_DEVICES}" '
          BEGIN{ split(sel, a, ","); for(i in a) sel_idx[a[i]]=1 }
          { if (NR-1 in sel_idx) print $1 }' \
      | head -n1
  )

  if [[ -n "$FREE" && "$FREE" -lt 512 ]]; then
    echo "[WARN] 선택된 GPU의 여유 메모리가 ${FREE} MiB 입니다 (< 512 MiB)."
    echo "       cuDNN 초기화 실패 가능성이 높습니다."
    echo "       다른 GPU를 지정하세요: --gpus <id> (예: --gpus 1)"
  fi
fi

# ==============================================================================
# 5.5) W&B 태그 구성 (실험/환경/레이아웃 + 사용자 태그)
# ==============================================================================

# 1) 기본 태그 리스트 구성
RAW_TAGS=("${EXPERIMENT}" "${ENV_GROUP}" "${LAYOUT}" "${ITERATIONS_OVERRIDE}")
if [[ -n "$CONF_PROFILE_PRETTY" ]]; then
  RAW_TAGS+=("${CONF_PROFILE_PRETTY}")
fi

# 2) --tags "a,b c" 같이 들어온 사용자 태그 파싱 (콤마/스페이스 모두 구분자)
if [[ -n "$TAGS" ]]; then
  IFS=', ' read -r -a EXTRA_TAGS <<< "$TAGS"
  for tag in "${EXTRA_TAGS[@]}"; do
    [[ -z "$tag" ]] && continue
    RAW_TAGS+=("$tag")
  done
fi

# 3) Hydra 문법용 문자열로 직렬화: ['t1','t2','t3']
serialize_tags() {
  local out="" sep=""
  for t in "$@"; do
    # 작은따옴표 이스케이프: a'b -> 'a'"'"'b'
    local esc=${t//\'/\'\"\'\"\'}
    out+="${sep}'${esc}'"
    sep=","
  done
  printf "%s" "$out"
}

TAGS_SERIALIZED=$(serialize_tags "${RAW_TAGS[@]}")
WANDB_TAGS_ARG="+wandb.tags=[${TAGS_SERIALIZED}]"

# ==============================================================================
# 6) 학습 실행 (Hydra 인자 구성 및 main 실행)
# ==============================================================================

PY_ARGS=(
  "+experiment=${EXPERIMENT}"
  "+env=${ENV_GROUP}"
  "+wandb.name=${RUN_NAME}"
  "+wandb.project=${WANDB_PROJECT}"
  "+wandb.entity=${WANDB_ENTITY}"
  "+wandb.notes=${NOTES}"
  "${WANDB_TAGS_ARG}"
)

# NUM_ITERATIONS override: 값이 존재하고 숫자이며, 1 초과인 경우만 적용
if [[ -n "${ITERATIONS_OVERRIDE:-}" && "${ITERATIONS_OVERRIDE}" =~ ^[0-9]+$ && "${ITERATIONS_OVERRIDE}" -gt 1 ]]; then
  echo "[INFO] Using NUM_ITERATIONS override from CLI: ${ITERATIONS_OVERRIDE}"
  PY_ARGS+=("NUM_ITERATIONS=${ITERATIONS_OVERRIDE}")
else
  echo "[INFO] NUM_ITERATIONS override not applied (must be integer > 1)."
fi

# NUM_SEEDS 처리: FCP 실험은 cfg 기본 1, --seeds로만 override
if [[ "$EXPERIMENT" == "rnn-fcp" || -n "$FCP_DIR" ]]; then
  if [[ "$SEEDS_EXPLICIT" == "1" ]]; then
    PY_ARGS+=("NUM_SEEDS=${NUM_SEEDS}")
  fi
else
  PY_ARGS+=("NUM_SEEDS=${NUM_SEEDS}")
fi

# FCP population 디렉토리 override
if [[ -n "$FCP_DIR" ]]; then
  PY_ARGS+=("+FCP=${FCP_DIR}")
  PY_ARGS+=("+FCP_DEVICE=${FCP_DEVICE}")
fi

# env 그룹이 original일 때만 layout override
if [[ "${ENV_GROUP}" == "original" ]]; then
  PY_ARGS+=("env.ENV_KWARGS.layout=${LAYOUT}")
fi

# 새 옵션 전달: ENV_DEVICE / CAST_OBS_BF16 / 배치 크기 오버라이드
if [[ -n "$ENV_DEVICE" ]]; then
  PY_ARGS+=("+ENV_DEVICE=${ENV_DEVICE}")
fi
if [[ "$CAST_OBS_BF16" == "1" ]]; then
  PY_ARGS+=("+CAST_OBS_BF16=True")
fi
if [[ -n "$MODEL_NUM_ENVS_OVERRIDE" ]]; then
  PY_ARGS+=("model.NUM_ENVS=${MODEL_NUM_ENVS_OVERRIDE}")
fi
if [[ -n "$MODEL_NUM_STEPS_OVERRIDE" ]]; then
  PY_ARGS+=("model.NUM_STEPS=${MODEL_NUM_STEPS_OVERRIDE}")
fi
if [[ -n "$CONF_PROFILE" ]]; then
  PY_ARGS+=("confidence=${CONF_PROFILE}")
  # CONF_PROFILE이 지정되면 confidence_trigger.enabled를 강제로 true로 설정
  PY_ARGS+=("confidence_trigger.enabled=true")
  # utils.py에서 suffix 생성을 위해 프로필 이름 전달
  PY_ARGS+=("+CONF_NAME=${CONF_PROFILE}")
fi
if [[ -n "$CONF_THRESHOLD" ]]; then
  PY_ARGS+=("confidence_trigger.entropy_threshold=${CONF_THRESHOLD}")
fi
if [[ -n "$CONF_COOLDOWN" ]]; then
  PY_ARGS+=("confidence_trigger.cooldown_steps=${CONF_COOLDOWN}")
fi
if [[ -n "$CONF_TARGET" ]]; then
  PY_ARGS+=("confidence_trigger.target=${CONF_TARGET}")
fi
if [[ -n "$CONF_N_THRESHOLD" ]]; then
  PY_ARGS+=("confidence_trigger.n_threshold=${CONF_N_THRESHOLD}")
fi

# E3T epsilon override
if [[ -n "$E3T_EPSILON" ]]; then
  PY_ARGS+=("E3T_EPSILON=${E3T_EPSILON}")
fi

# Panic overrides appended if enabled (Hydra keys defined in base.yaml)
if [[ "$PANIC_ENABLED" == "1" ]]; then
  PY_ARGS+=("panic.enabled=true" "panic.start_step=${PANIC_START_STEP}" "panic.duration=${PANIC_DURATION}")
fi

# ====================================================
# LD_LIBRARY_PATH / XLA_FLAGS를 해제하여 라이브러리 충돌 회피
cd ~/mingukang/ex-overcookedv2

# 1) uv env 활성화
source overcooked_v2/bin/activate

# 2) 파이썬 진단 (가상환경 활성화 후)
env -u LD_LIBRARY_PATH -u XLA_FLAGS python - <<'PY' 2>&1 | filter_ptx
import os
import jax
import sys

print("[DEBUG] Python executable:", sys.executable)
print("[DEBUG] Python version:", sys.version)
print("[DEBUG] CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("[DEBUG] JAX_PLATFORMS:", os.environ.get("JAX_PLATFORMS"))
print("[DEBUG] LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"))

try:
    devices = jax.devices()
    print("[JAX] devices:", devices)
except Exception as e:
    print("[JAX] Error initializing devices:", e)
PY

# 3) 실험 실행
cd experiments

env -u LD_LIBRARY_PATH -u XLA_FLAGS \
  python overcooked_v2_experiments/ppo/main.py "${PY_ARGS[@]}" 2>&1 | filter_ptx

