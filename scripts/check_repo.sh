#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-${ROOT_DIR}/.venv_vjepa2/bin/python}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-${TMPDIR:-/tmp}/jepas-pycache}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python environment not found: ${PYTHON}" >&2
  echo "Create it with: scripts/create_inference_venv.sh cpu" >&2
  exit 1
fi

cd "${ROOT_DIR}"

bash -n scripts/*.sh weights/*.sh
"${PYTHON}" -m compileall -q experiments
"${PYTHON}" -m unittest \
  experiments.common_resolution.test_render_comparison \
  experiments.shared.test_provenance \
  experiments.smoke.test_ijepa_lite_affinity_novelty_smoke

for manifest in manifests/*.json; do
  "${PYTHON}" -m json.tool "${manifest}" >/dev/null
done

if command -v node >/dev/null 2>&1; then
  node --check experiments/cross_media/review/static/review.js
fi

git diff --check
git diff --cached --check

echo "Repository checks passed."
