#!/bin/bash
# Bulk token extraction + token metrics over every dataset and model.
#
# Smoke test:  sbatch --export=ALL,N=10 scripts/leonardo_bulk.sbatch
# Full run:    sbatch --export=ALL,N=1024 --qos=normal --time=12:00:00 scripts/leonardo_bulk.sbatch
#
# One node, all 4 GPUs: the 16 (dataset, model) jobs are pulled from a shared queue by
# 4 workers, one per GPU (CUDA_VISIBLE_DEVICES), each using CPUS/4 metric workers.
#
# Fair-comparison design: every model sees the SAME sampled inputs per dataset
# (sampling depends only on dataset, kind and seed, not on the model), at its own
# native resolution. Image models (ijepa, radjepa) run on image datasets; video
# models run on video datasets. A video model on an image would repeat the frame
# across tubelets, which token_metrics' 3D grid cannot treat as a real image.
#SBATCH --job-name=jepas_bulk
#SBATCH --account=IscrC_PRIMAL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --output=/leonardo_work/IscrC_PRIMAL/dvitturi/JEPAS/logs/bulk_%j.out

set -uo pipefail
cd /leonardo_work/IscrC_PRIMAL/dvitturi/JEPAS
source scripts/leonardo_env.sh

N="${N:-10}"
SEED="${SEED:-42}"
FRAMES=16
OUT="${FAST}/outputs/bulk_n${N}_seed${SEED}"
DS="${JEPAS_DATASETS_ROOT}"

IMAGE_MODELS=(ijepa radjepa)
VIDEO_MODELS=(vjepa2-vitl vjepa2-1-vitb echojepa)

declare -A INPUTS
INPUTS[imagenet]="${IMAGENET2012_VAL}"
INPUTS[rsna]="${DS}/medical/rsna-pneumonia-detection-challenge/stage_2_test_images_png"
INPUTS[diving48]="${DS}/video/diving48/rgb"
INPUTS[echonet]="${DS}/eidos-datasets/data-registry/medical/cardio/catdx/StanfordDataset/Videos"
INPUTS[kinetics]="/leonardo/prod/data/ai/kinetics/kinetics400/256/val"
# EPIC recordings are minutes to half an hour long, so cut seeded 10 s windows first.
EPIC_CLIPS="${FAST}/datasets_derived/epic-kitchens_clips10s_n${N}_seed${SEED}"
INPUTS[epic-kitchens]="${EPIC_CLIPS}"

python scripts/make_epic_clips.py --root /leonardo/prod/data/ai/epic-kitchens/epic-kitchens-100 \
  --out-dir "${EPIC_CLIPS}" --n "${N}" --seed "${SEED}" --workers "${SLURM_CPUS_PER_TASK}" || exit 1

NGPU=4
CPUS_PER_JOB=$(( SLURM_CPUS_PER_TASK / NGPU ))
LOGS="${OUT}/logs"
mkdir -p "${LOGS}"

# Job queue: "dataset kind model" lines, consumed atomically by the GPU workers.
QUEUE="${LOGS}/queue.txt"
COUNTER="${LOGS}/queue.counter"
: > "${QUEUE}"
for d in imagenet rsna; do for m in "${IMAGE_MODELS[@]}"; do echo "$d image $m" >> "${QUEUE}"; done; done
for d in diving48 echonet kinetics epic-kitchens; do for m in "${VIDEO_MODELS[@]}"; do echo "$d video $m" >> "${QUEUE}"; done; done
echo 0 > "${COUNTER}"

next_job() {
  flock "${COUNTER}" bash -c '
    i=$(cat "$0"); i=$((i + 1)); echo "$i" > "$0"; sed -n "${i}p" "$1"' "${COUNTER}" "${QUEUE}"
}

run_one() {  # dataset kind model  (GPU is fixed by CUDA_VISIBLE_DEVICES)
  local dataset="$1" kind="$2" model="$3"
  echo "=== ${dataset} / ${model} (${kind}, n=${N}, seed=${SEED}, gpu=${CUDA_VISIBLE_DEVICES})"
  # shellcheck disable=SC2086
  python experiments/extract_batch.py --model "${model}" --kind "${kind}" \
    --input ${INPUTS[$dataset]} --recursive --n "${N}" --seed "${SEED}" \
    --frames "${FRAMES}" --output-dir "${OUT}/${dataset}" || return 1
  python experiments/token_metrics.py --model "${model}" --seed "${SEED}" \
    --workers "${CPUS_PER_JOB}" \
    --embeddings "${OUT}/${dataset}/${model}/embeddings.h5" || return 1
  python experiments/token_metrics_null.py --model "${model}" --seed "${SEED}" \
    --workers "${CPUS_PER_JOB}" \
    --embeddings "${OUT}/${dataset}/${model}/embeddings.h5" \
    --out-dir "${OUT}/${dataset}/${model}/null_baselines" || return 1
}

worker() {  # gpu_id
  local gpu="$1" line
  export CUDA_VISIBLE_DEVICES="${gpu}"
  while line="$(next_job)"; [[ -n "${line}" ]]; do
    read -r dataset kind model <<< "${line}"
    if run_one "${dataset}" "${kind}" "${model}" > "${LOGS}/${dataset}__${model}.log" 2>&1; then
      echo "[gpu${gpu}] ok     ${dataset}/${model}"
    else
      echo "[gpu${gpu}] FAILED ${dataset}/${model} (see ${LOGS}/${dataset}__${model}.log)"
      echo "${dataset}/${model}" >> "${LOGS}/failed.txt"
    fi
  done
}

rm -f "${LOGS}/failed.txt"
for g in $(seq 0 $((NGPU - 1))); do worker "${g}" & done
wait

python experiments/token_metrics_aggregate.py "${OUT}" --out-dir "${OUT}/aggregate" \
  --null "${OUT}" || echo "aggregate failed"

if [[ -s "${LOGS}/failed.txt" ]]; then
  echo "Failed jobs:"; cat "${LOGS}/failed.txt"; exit 1
fi
echo "Done: ${OUT}"
