#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS_DIR="${ROOT_DIR}/weights_py"
VENV_DIR="${ROOT_DIR}/.venv"

ARIA_CONNECTIONS="${ARIA_CONNECTIONS:-16}"
ARIA_SPLITS="${ARIA_SPLITS:-16}"
ARIA_MIN_SPLIT_SIZE="${ARIA_MIN_SPLIT_SIZE:-1M}"

usage() {
  cat <<'EOF'
Download the smallest available non-manual model for each repo into weights_py/.

Usage:
  weights_py/download_weights.sh [repo ...]

Repos:
  vjepa2     V-JEPA 2.1 ViT-B/16, smallest V-JEPA checkpoint listed in vjepa2 README
  ijepa      I-JEPA ViT-H/14 IN1K checkpoint, smallest architecture class listed in ijepa README
  radjepa    RadJEPA Hugging Face repo, ViT-B/14
  all        All repos above

EchoJEPA is intentionally excluded because its weights are downloaded manually from Google Drive.

Examples:
  weights_py/download_weights.sh all
  weights_py/download_weights.sh vjepa2 radjepa

Environment:
  ARIA_CONNECTIONS=16       Connections per direct URL
  ARIA_SPLITS=16            Splits per direct URL
  ARIA_MIN_SPLIT_SIZE=1M    Minimum split size
EOF
}

want() {
  local target="$1"
  shift
  local group
  for group in "$@"; do
    if [[ "${group}" == "all" || "${group}" == "${target}" ]]; then
      return 0
    fi
  done
  return 1
}

need_cmd() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "Missing required command: ${cmd}" >&2
    exit 1
  fi
}

download_direct() {
  local repo_dir="$1"
  local url="$2"
  local output="$3"
  local dest_dir="${WEIGHTS_DIR}/${repo_dir}"
  local dest="${dest_dir}/${output}"

  mkdir -p "${dest_dir}"

  if [[ -s "${dest}" && ! -e "${dest}.aria2" ]]; then
    echo "Already downloaded: ${dest}"
    return 0
  fi

  echo "Downloading: ${url}"
  echo "Destination: ${dest}"
  aria2c \
    --continue=true \
    --max-connection-per-server="${ARIA_CONNECTIONS}" \
    --split="${ARIA_SPLITS}" \
    --min-split-size="${ARIA_MIN_SPLIT_SIZE}" \
    --auto-file-renaming=false \
    --allow-overwrite=true \
    --summary-interval=30 \
    --dir="${dest_dir}" \
    --out="${output}" \
    "${url}"

  if [[ ! -s "${dest}" || -e "${dest}.aria2" ]]; then
    echo "Download did not complete cleanly: ${dest}" >&2
    exit 1
  fi
}

download_hf() {
  local repo="$1"
  local output_dir="$2"
  local dest_dir="${WEIGHTS_DIR}/${output_dir}"
  local complete_marker="${dest_dir}/.download_complete"

  mkdir -p "${dest_dir}"

  if [[ -f "${complete_marker}" ]]; then
    echo "Already downloaded: ${dest_dir}"
    return 0
  fi

  echo "Downloading HF repo: ${repo}"
  echo "Destination: ${dest_dir}"
  "${VENV_DIR}/bin/hf" download "${repo}" --local-dir "${dest_dir}"
  touch "${complete_marker}"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$#" -eq 0 ]]; then
  usage
  exit 2
fi

need_cmd aria2c

if [[ ! -x "${VENV_DIR}/bin/hf" ]]; then
  echo "Missing Hugging Face CLI at ${VENV_DIR}/bin/hf. Create the workspace venv first." >&2
  exit 1
fi

mkdir -p "${WEIGHTS_DIR}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

if want vjepa2 "$@"; then
  download_direct \
    vjepa2 \
    https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt \
    vjepa2_1_vitb_dist_vitG_384.pt
fi

if want ijepa "$@"; then
  download_direct \
    ijepa \
    https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar \
    IN1K-vit.h.14-300e.pth.tar
fi

if want radjepa "$@"; then
  download_hf AIDElab-IITBombay/RadJEPA radjepa/hf-RadJEPA
fi
