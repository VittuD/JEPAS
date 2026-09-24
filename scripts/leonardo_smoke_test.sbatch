#!/bin/bash
#SBATCH --job-name=jepas_smoke
#SBATCH --account=IscrC_PRIMAL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=/leonardo_scratch/fast/IscrC_PRIMAL/dvitturi/JEPAS/logs/smoke_%j.out

set -euo pipefail
cd /leonardo_work/IscrC_PRIMAL/dvitturi/JEPAS
source scripts/leonardo_env.sh

nvidia-smi -L
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available"
print("torch", torch.__version__, "device", torch.cuda.get_device_name(0))
x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
print("matmul ok", (x @ x).float().abs().mean().item())
PY

PYTHON="$(which python)" bash scripts/check_repo.sh
ls "$IMAGENET2012_VAL" | head -3
ls "$JEPAS_DATASETS_ROOT"
