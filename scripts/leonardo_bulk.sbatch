#!/bin/bash
# Bulk token extraction + token metrics over every dataset and model.
#
# Smoke test:  sbatch --export=ALL,N=10 scripts/leonardo_bulk.sbatch
# Full run:    sbatch --export=ALL,N=1024 --qos=normal --time=12:00:00 scripts/leonardo_bulk.sbatch
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
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/leonardo_work/IscrC_PRIMAL/dvitturi/JEPAS/logs/bulk_%j.out

set -euo pipefail
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
INPUTS[epic-kitchens]="$(echo /leonardo/prod/data/ai/epic-kitchens/epic-kitchens-100/P*/videos)"

run() {  # dataset kind model...
  local dataset="$1" kind="$2"; shift 2
  local model
  for model in "$@"; do
    echo "=== ${dataset} / ${model} (${kind}, n=${N}, seed=${SEED})"
    # shellcheck disable=SC2086
    python experiments/extract_batch.py --model "${model}" --kind "${kind}" \
      --input ${INPUTS[$dataset]} --recursive --n "${N}" --seed "${SEED}" \
      --frames "${FRAMES}" --output-dir "${OUT}/${dataset}"
    python experiments/token_metrics.py --model "${model}" --seed "${SEED}" \
      --workers "${SLURM_CPUS_PER_TASK}" \
      --embeddings "${OUT}/${dataset}/${model}/embeddings.h5"
    python experiments/token_metrics_null.py --model "${model}" --seed "${SEED}" \
      --workers "${SLURM_CPUS_PER_TASK}" \
      --embeddings "${OUT}/${dataset}/${model}/embeddings.h5" \
      --out-dir "${OUT}/${dataset}/${model}/null_baselines"
  done
}

for d in imagenet rsna; do run "$d" image "${IMAGE_MODELS[@]}"; done
for d in diving48 echonet kinetics epic-kitchens; do run "$d" video "${VIDEO_MODELS[@]}"; done

python experiments/token_metrics_aggregate.py "${OUT}" --out-dir "${OUT}/aggregate" \
  --null "${OUT}"
echo "Done: ${OUT}"
