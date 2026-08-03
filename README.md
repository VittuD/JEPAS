# JEPAs

Minimal workspace for extracting and qualitatively comparing embeddings from
JEPA-family source repositories.

```bash
git submodule update --init --recursive
scripts/create_inference_venv.sh cpu   # or: cuda
```

Downloaded checkpoints live under `weights/` and are ignored. Available weight
sources and the resumable downloader are documented in `weights/OPTIONS.md`.

## Extract embeddings

```bash
.venv_vjepa2/bin/python experiments/extract.py \
  --model ijepa \
  --input experiments/inputs/web_examples/dog_image.jpg \
  --output-dir experiments/outputs/ijepa_dog
```

Each result contains one lossless, compressed `embeddings.h5` with original-dtype
`tokens` and FP32 mean-pooled `pooled` datasets, plus `input.jpg` and `run.log`.
The experiment root contains `manifest.json`.

Visualize an extraction independently:

```bash
.venv_vjepa2/bin/python experiments/visualize.py \
  experiments/outputs/ijepa_dog/dog_image/ijepa/embeddings.h5 \
  --out-dir experiments/outputs/ijepa_dog/dog_image/ijepa \
  --grid-shape 16x16 \
  --image experiments/outputs/ijepa_dog/dog_image/ijepa/input.jpg
```

After checking the visualization, cleanup is independently runnable and guarded:

```bash
.venv_vjepa2/bin/python experiments/cleanup.py \
  experiments/outputs/ijepa_dog
```

## Compare models

```bash
# Every default 2D/video model on every example at 384x384
.venv_vjepa2/bin/python experiments/compare.py

# One domain-appropriate input per model
.venv_vjepa2/bin/python experiments/compare.py --pairing matched

# Every input at each model's native spatial resolution
.venv_vjepa2/bin/python experiments/compare.py --resolution native

# Preserve embeddings.h5 instead of cleaning it after successful visualization
.venv_vjepa2/bin/python experiments/compare.py --keep-embeddings
```

Comparison runs resume each missing stage: extraction is skipped when HDF5 exists,
visualization is skipped when its declared artifacts validate, and HDF5 cleanup is
the default only after validation succeeds.

## Browse experiments

The tracked browser discovers visualized new-schema experiments without generating
per-run HTML or composite pages:

```bash
.venv_vjepa2/bin/python visualization/serve.py
# Open http://127.0.0.1:43871

# Optional overrides
.venv_vjepa2/bin/python visualization/serve.py \
  --port 49127 --outputs /path/to/experiment/outputs
```

Pass `--device cuda --precision auto` on a CUDA host. Use `--dry-run` to inspect
commands without loading checkpoints. See `experiments/README.md` for model,
input, and modality-adaptation details.

External implementations are pinned under `repos/`. The optional custom
`ijepa_lite` checkpoint is described by
`manifests/ijepa_lite_affinity_novelty.json` and can be selected explicitly.

Run the lightweight checks with:

```bash
scripts/check_repo.sh
```
