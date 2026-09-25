#!/bin/bash
# Download checkpoints into $FAST/vjepa2 from the serial partition.
#
# lrd_all_serial runs on a login node (internet access, no 10-minute CPU limit, up to 4 h),
# which is what long downloads need: compute nodes have no internet, and the data mover
# (leonardo-dm) only allows file transfers, not commands such as wget.
#
# Usage (resumable: rerun the same command if it times out; complete files are skipped):
#   sbatch scripts/leonardo_download_weights.sbatch
#   FILES="vitg.pt vitg-384.pt" sbatch --export=ALL scripts/leonardo_download_weights.sbatch
#SBATCH --job-name=dl_weights
#SBATCH --account=IscrC_PRIMAL
#SBATCH --partition=lrd_all_serial
#SBATCH --ntasks=1 --cpus-per-task=2 --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=/leonardo_work/IscrC_PRIMAL/dvitturi/JEPAS/logs/download_%j.out
set -uo pipefail

FAST=/leonardo_scratch/fast/IscrC_PRIMAL/dvitturi/JEPAS
DEST="${DEST:-${FAST}/vjepa2}"
BASE_URL="${BASE_URL:-https://dl.fbaipublicfiles.com/vjepa2}"
FILES="${FILES:-vjepa2_1_vitl_dist_vitG_384.pt vjepa2_1_vitG_384.pt vitg.pt}"
mkdir -p "${DEST}"

failed=0
for f in ${FILES}; do
  remote=$(curl -sIL --max-time 60 "${BASE_URL}/${f}" | tr -d '\r' | awk 'tolower($1)=="content-length:"{n=$2} END{print n}')
  local_size=$(stat -c %s "${DEST}/${f}" 2>/dev/null || echo 0)
  if [[ -z "${remote}" ]]; then echo "$(date +%T) ${f}: cannot read remote size"; failed=1; continue; fi
  if [[ "${local_size}" == "${remote}" ]]; then echo "$(date +%T) ${f}: already complete (${remote} bytes)"; continue; fi
  echo "$(date +%T) ${f}: ${local_size}/${remote} bytes, downloading"
  wget -c --tries=10 --waitretry=10 --read-timeout=60 -nv -O "${DEST}/${f}" "${BASE_URL}/${f}"
  local_size=$(stat -c %s "${DEST}/${f}" 2>/dev/null || echo 0)
  if [[ "${local_size}" == "${remote}" ]]; then echo "$(date +%T) ${f}: OK (${local_size} bytes)"; else echo "$(date +%T) ${f}: INCOMPLETE ${local_size}/${remote}"; failed=1; fi
done
exit ${failed}
