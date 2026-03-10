#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_PATH="${REPO_ROOT}/overcooked_v2"
UPGRADE_PIP=true
DRY_RUN=false

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Options:
  --venv PATH         Override virtualenv path (default: ${REPO_ROOT}/overcooked_v2)
  --skip-upgrade-pip  Skip 'python -m pip install --upgrade pip setuptools wheel'
  --dry-run           Print commands only
  -h, --help          Show this help
USAGE
}

run_cmd() {
  echo "+ $*"
  if [[ "${DRY_RUN}" != "true" ]]; then
    "$@"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv)
      VENV_PATH="$2"
      shift 2
      ;;
    --skip-upgrade-pip)
      UPGRADE_PIP=false
      shift
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
      echo "[ERROR] Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

PY_BIN="${VENV_PATH}/bin/python"
if [[ ! -x "${PY_BIN}" ]]; then
  echo "[ERROR] Python not found: ${PY_BIN}" >&2
  echo "        Create the venv first: python3 -m venv ${VENV_PATH}" >&2
  exit 1
fi

echo "[INFO] Repo: ${REPO_ROOT}"
echo "[INFO] Venv: ${VENV_PATH}"

if [[ "${UPGRADE_PIP}" == "true" ]]; then
  run_cmd "${PY_BIN}" -m pip install --upgrade pip setuptools wheel
fi

run_cmd "${PY_BIN}" -m pip install -e "${REPO_ROOT}/JaxMARL"
run_cmd "${PY_BIN}" -m pip install -e "${REPO_ROOT}/experiments-stablock"

echo "[INFO] Done. ex-overcookedv2 editable installs are bound to ${VENV_PATH}."
