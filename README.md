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
  --output-dir experiments/outputs/extraction/ijepa
```

Each result contains `tokens.npy`, `pooled.npy`, the preprocessed input, and a
log. The output root contains one concise `manifest.json`.

## Compare models

```bash
# Every default 2D/video model on every example at 384x384
.venv_vjepa2/bin/python experiments/compare.py

# One domain-appropriate input per model
.venv_vjepa2/bin/python experiments/compare.py --pairing matched

# Every input at each model's native spatial resolution
.venv_vjepa2/bin/python experiments/compare.py --resolution native
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
