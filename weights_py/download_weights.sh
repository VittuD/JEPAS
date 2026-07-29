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
Download non-EchoJEPA weights into weights_py/.

Usage:
  weights_py/download_weights.sh [group ...]

Groups:
  vjepa2-base       V-JEPA 2 direct PyTorch checkpoints
  vjepa2-21         V-JEPA 2.1 direct PyTorch checkpoints
  vjepa2-ac         V-JEPA 2 action-conditioned checkpoint
  vjepa2-probes     V-JEPA 2 evaluation probe checkpoints
  vjepa2-hf         V-JEPA 2 Hugging Face repos listed in the README
  ijepa             I-JEPA direct PyTorch checkpoints
  radjepa           RadJEPA Hugging Face repo
  all               Everything above

Examples:
  weights_py/download_weights.sh vjepa2-base ijepa radjepa
  weights_py/download_weights.sh all

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
  local url="$1"
  local output="$2"
  aria2c \
    --continue=true \
    --max-connection-per-server="${ARIA_CONNECTIONS}" \
    --split="${ARIA_SPLITS}" \
    --min-split-size="${ARIA_MIN_SPLIT_SIZE}" \
    --auto-file-renaming=false \
    --allow-overwrite=true \
    --summary-interval=30 \
    --dir="${WEIGHTS_DIR}" \
    --out="${output}" \
    "${url}"
}

download_hf() {
  local repo="$1"
  local output_dir="$2"
  "${VENV_DIR}/bin/hf" download "${repo}" --local-dir "${WEIGHTS_DIR}/${output_dir}"
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

if want vjepa2-base "$@"; then
  download_direct https://dl.fbaipublicfiles.com/vjepa2/vitl.pt vjepa2-vitl.pt
  download_direct https://dl.fbaipublicfiles.com/vjepa2/vith.pt vjepa2-vith.pt
  download_direct https://dl.fbaipublicfiles.com/vjepa2/vitg.pt vjepa2-vitg.pt
  download_direct https://dl.fbaipublicfiles.com/vjepa2/vitg-384.pt vjepa2-vitg-384.pt
fi

if want vjepa2-21 "$@"; then
  download_direct https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt vjepa2_1_vitb_dist_vitG_384.pt
  download_direct https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt vjepa2_1_vitl_dist_vitG_384.pt
  download_direct https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitg_384.pt vjepa2_1_vitg_384.pt
  download_direct https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitG_384.pt vjepa2_1_vitG_384.pt
fi

if want vjepa2-ac "$@"; then
  download_direct https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt vjepa2-ac-vitg.pt
fi

if want vjepa2-probes "$@"; then
  download_direct https://dl.fbaipublicfiles.com/vjepa2/evals/ssv2-vitl-16x2x3.pt ssv2-vitl-16x2x3.pt
  download_direct https://dl.fbaipublicfiles.com/vjepa2/evals/diving48-vitl-256.pt diving48-vitl-256.pt
  download_direct https://dl.fbaipublicfiles.com/vjepa2/evals/ek100-vitl-256.pt ek100-vitl-256.pt
  download_direct https://dl.fbaipublicfiles.com/vjepa2/evals/ssv2-vitg-384-64x2x3.pt ssv2-vitg-384-64x2x3.pt
  download_direct https://dl.fbaipublicfiles.com/vjepa2/evals/diving48-vitg-384-32x4x3.pt diving48-vitg-384-32x4x3.pt
  download_direct https://dl.fbaipublicfiles.com/vjepa2/evals/ek100-vitg-384.pt ek100-vitg-384.pt
fi

if want vjepa2-hf "$@"; then
  download_hf facebook/vjepa2-vitl-fpc64-256 hf-vjepa2-vitl-fpc64-256
  download_hf facebook/vjepa2-vith-fpc64-256 hf-vjepa2-vith-fpc64-256
  download_hf facebook/vjepa2-vitg-fpc64-256 hf-vjepa2-vitg-fpc64-256
  download_hf facebook/vjepa2-vitg-fpc64-384 hf-vjepa2-vitg-fpc64-384
fi

if want ijepa "$@"; then
  download_direct https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar ijepa-IN1K-vit.h.14-300e.pth.tar
  download_direct https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.16-448px-300e.pth.tar ijepa-IN1K-vit.h.16-448px-300e.pth.tar
  download_direct https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.h.14-900e.pth.tar ijepa-IN22K-vit.h.14-900e.pth.tar
  download_direct https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.g.16-600e.pth.tar ijepa-IN22K-vit.g.16-600e.pth.tar
fi

if want radjepa "$@"; then
  download_hf AIDElab-IITBombay/RadJEPA hf-RadJEPA
fi
