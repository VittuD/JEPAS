# Embedding Extraction

`extract_embeddings.py` is the generic source-backed extraction entry point. It
dispatches to the tested model scripts under `experiments/smoke/` and saves both
non-pooled tokens and their mean-pooled representation.

Single image:

```bash
.venv_vjepa2/bin/python experiments/extraction/extract_embeddings.py \
  --model ijepa \
  --input experiments/inputs/web_examples/dog_image.jpg \
  --output-dir experiments/outputs/extraction/ijepa
```

Directory of images:

```bash
.venv_vjepa2/bin/python experiments/extraction/extract_embeddings.py \
  --model radjepa \
  --input path/to/images \
  --recursive \
  --output-dir experiments/outputs/extraction/radjepa
```

Remote CUDA example:

```bash
.venv_jepa_cuda/bin/python experiments/extraction/extract_embeddings.py \
  --model vjepa2-vitl \
  --input path/to/video.mp4 \
  --output-dir experiments/outputs/extraction/vjepa2-vitl \
  --device cuda \
  --precision auto
```

Use `--dry-run --device cuda` to inspect CUDA commands without requiring a GPU.
Use `--weights` to override the default local checkpoint or model directory.

Model choices:

- `ijepa`
- `radjepa`
- `vjepa2-vitl`
- `vjepa2-vith`
- `vjepa2-1-vitb`
- `vjepa2-1-vitg`
- `echojepa`
- `neurojepa`
- `ijepa-lite-affinity-novelty`

V-JEPA and EchoJEPA accept images or videos. I-JEPA and RadJEPA accept images.
The custom I-JEPA Lite model also accepts images. Neuro-JEPA accepts `.npy`,
`.nii`, or `.nii.gz` volumes.

The I-JEPA Lite checkpoint is intentionally not downloaded. Its portable
default is `weights/ijepa_lite_affinity_novelty/last.pt`; override it when the
checkpoint is stored elsewhere:

```bash
export IJEPA_LITE_AFFINITY_NOVELTY_CHECKPOINT=weights/ijepa_lite_affinity_novelty/last.pt

.venv_jepa_cuda/bin/python experiments/extraction/extract_embeddings.py \
  --model ijepa-lite-affinity-novelty \
  --input path/to/chest_xray.png \
  --output-dir experiments/outputs/extraction/ijepa-lite-affinity-novelty \
  --device cuda --precision auto
```

Each input produces:

```text
<output-dir>/<input-name>/
  tokens.npy
  pooled.npy
  preview.jpg or preview.png
  extract.log
  result.json
```

The output root also contains `manifest.json`. Matching completed results are
skipped. Changed inputs, weights, source revisions, runtime, or extraction
settings require `--force` or a different output directory.
