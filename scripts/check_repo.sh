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
"${PYTHON}" -m compileall -q experiments visualization
"${PYTHON}" -m marimo check --strict visualization/browser.py
"${PYTHON}" -m unittest \
  experiments.tests.test_compare \
  experiments.tests.test_artifacts \
  experiments.tests.test_browser \
  experiments.tests.test_extract \
  experiments.tests.test_provenance \
  experiments.tests.test_ijepa_lite

for manifest in manifests/*.json; do
  "${PYTHON}" -m json.tool "${manifest}" >/dev/null
done

git diff --check
git diff --cached --check

echo "Repository checks passed."
