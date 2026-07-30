#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<EOF
Usage:
  scripts/create_inference_venv.sh cpu
  scripts/create_inference_venv.sh cuda

Optional:
  VENV_DIR=/custom/path scripts/create_inference_venv.sh cuda

Defaults:
  cpu   -> ${ROOT_DIR}/.venv_vjepa2
  cuda  -> ${ROOT_DIR}/.venv_jepa_cuda
EOF
}

if [[ $# -ne 1 ]]; then
  usage
  exit 2
fi

PROFILE="$1"

case "${PROFILE}" in
  cpu)
    DEFAULT_VENV_DIR="${ROOT_DIR}/.venv_vjepa2"
    REQUIREMENTS_FILE="${ROOT_DIR}/requirements/vjepa2_inference.txt"
    ;;
  cuda | gpu)
    DEFAULT_VENV_DIR="${ROOT_DIR}/.venv_jepa_cuda"
    REQUIREMENTS_FILE="${ROOT_DIR}/requirements/vjepa2_inference_cuda.txt"
    ;;
  -h | --help | help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown inference venv profile: ${PROFILE}" >&2
    usage >&2
    exit 2
    ;;
esac

VENV_DIR="${VENV_DIR:-${DEFAULT_VENV_DIR}}"

python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install -r "${REQUIREMENTS_FILE}"

cat <<EOF
Created ${VENV_DIR}
Profile: ${PROFILE}
Requirements: ${REQUIREMENTS_FILE}

Activate with:
  source ${VENV_DIR}/bin/activate
EOF
