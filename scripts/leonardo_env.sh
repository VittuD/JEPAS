#!/usr/bin/env bash
# Source on Leonardo: `source scripts/leonardo_env.sh`
FAST=/leonardo_scratch/fast/IscrC_PRIMAL/dvitturi/JEPAS
module load python/3.11.7 profile/deeplrn imagenet/ilsvrc2012
source "${FAST}/venv_jepa_cuda/bin/activate"
# Static ffmpeg/ffprobe (no module on Leonardo); needed for video decoding.
export PATH="${FAST}/bin:${PATH}"
export JEPAS_DATASETS_ROOT="${FAST}/datasets"
# ImageNet: $IMAGENET2012_VAL (from the module) has n0*/ class folders.
