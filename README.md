# JEPAs

Workspace for comparing JEPA-family model repositories and running local embedding extraction experiments.

External model code lives under `repos/` as Git submodules. Downloaded checkpoints and virtual environments are intentionally not tracked.

Contribution and extension conventions are documented in
[`CONTRIBUTING.md`](CONTRIBUTING.md).

Initialize source repositories after cloning:

```bash
git submodule update --init --recursive
```

## Layout

```text
experiments/        Local embedding extraction code and experiment outputs
manifests/          Local manifests and notes
repos/              External model repositories as submodules
weights/            Local checkpoints plus tracked download scripts/docs
requirements/       Reproducible environment inputs
scripts/            Repo setup helpers
```

`manifests/ijepa_lite_affinity_novelty.json` records the remote custom model's
inference-relevant Hydra configuration and checkpoint location without storing
the checkpoint itself.

## Environment

Create the lightweight CPU download/extraction environment:

```bash
scripts/create_weights_venv.sh
source .venv/bin/activate
```

Create the shared CPU inference environment for source-repo smoke tests:

```bash
scripts/create_vjepa2_venv.sh
source .venv_vjepa2/bin/activate
```

Or choose the inference profile explicitly:

```bash
scripts/create_inference_venv.sh cpu
source .venv_vjepa2/bin/activate

scripts/create_inference_venv.sh cuda
source .venv_jepa_cuda/bin/activate
```

Inference commands default to `--device auto`: the CPU environment selects CPU,
while the CUDA environment selects CUDA when the driver is available. Use
`--device cpu` or `--device cuda` to make the choice explicit. CUDA `auto`
precision uses BF16 when supported and otherwise FP16; CPU inference uses FP32.

## Weights

EchoJEPA weights are managed manually from Google Drive. The download venv includes `gdown` for that path. Other scripted weight options are documented in:

```bash
weights/OPTIONS.md
```

Start resumable non-Echo downloads with:

```bash
weights/download_weights.sh all
```

## Experiments

Experiment code should live under `experiments/` and import model code from the
source repositories under `repos/`.
Current smoke tests cover V-JEPA 2, I-JEPA, RadJEPA, EchoJEPA V-JEPA 2,
Neuro-JEPA, and the optional `ijepa_lite` affinity-novelty checkpoint.

The generic extractor accepts individual files or directories and writes both
token-level and mean-pooled representations:

```bash
.venv_vjepa2/bin/python experiments/extraction/extract_embeddings.py \
  --model ijepa \
  --input path/to/images \
  --output-dir experiments/outputs/extraction/ijepa
```

Aggregate comparisons are documented under [`experiments/`](experiments/README.md).
Use `experiments/cross_media/run.py` for the fixed `384x384` matrix and
`experiments/cross_media_native_resolution/run.py` for the same matrix at each
model's native spatial resolution.
