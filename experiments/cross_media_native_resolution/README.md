# Cross-Media Native-Resolution Comparison

This experiment runs the same five-model by four-input matrix as
`experiments/cross_media`, but each model receives its catalogued native square
spatial resolution:

- I-JEPA: `224x224`
- RadJEPA: `224x224`
- V-JEPA 2 ViT-L: `256x256`
- V-JEPA 2.1 ViT-B: `384x384`
- EchoJEPA V-JEPA 2 ViT-L: `224x224`

The optional `ijepa-lite-affinity-novelty` model also runs at its native
`224x224` resolution when explicitly selected with
`--models ijepa-lite-affinity-novelty`. It is excluded from the default matrix
because its checkpoint remains on Leonardo.

The canonical crop and modality adaptations are unchanged. References are
resized separately for each resolution, video-model static maps use temporal
slice 0, and video models additionally produce temporal MP4 visualizations.

Run on CPU:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv_vjepa2/bin/python experiments/cross_media_native_resolution/run.py \
  --device cpu --precision fp32
```

Results are written under:

```text
experiments/outputs/cross_media_native_resolution/native/
```

The manifest records `resolution_mode: native`, the resolution used by each
model, exact commands, checkpoint and code provenance, tensor geometry, and
artifact fingerprints. Build the offline browser with:

```bash
.venv_vjepa2/bin/python experiments/cross_media/build_review.py \
  experiments/outputs/cross_media_native_resolution/native/manifest.json
```

Comparisons normalize panel display sizes for browsing, but token-grid
resolution remains model-specific. PCA and clustering are fitted independently
per result, so compare spatial organization rather than literal colors.
