# Cross-Media 384 Comparison

This experiment runs every 2D image/video JEPA against every catalogued image
and video at `384x384`.

For the same model/input matrix at each model's native spatial resolution, use
[`../cross_media_native_resolution/README.md`](../cross_media_native_resolution/README.md).

Models:

- I-JEPA
- RadJEPA
- V-JEPA 2 ViT-L
- V-JEPA 2.1 ViT-B
- EchoJEPA V-JEPA 2 ViT-L

The optional `ijepa-lite-affinity-novelty` ChestMNIST ViT-H/14 model is excluded
from the default matrix because its checkpoint is remote. On Leonardo, include
it explicitly with `--models ijepa-lite-affinity-novelty` (or list it alongside
the five default models). It participates as an image model, so video inputs
use frame 0.

Inputs:

- ImageNet-style dog image
- Chest X-ray
- Diving48 video
- EchoNet echocardiography video

Modality adaptation is intentionally simple:

- Image model plus video: extract frame 0 with FFmpeg.
- Video model plus image: repeat the image across 16 frames.
- Matching modalities: use the source image or sampled source video directly.

Every model row displays the same canonical square reference for an input.
For videos this is frame 0, and video token maps use temporal slice 0 (the first
tubelet) so the displayed frame and token field are temporally aligned.
Video-model results retain this static PNG and additionally write
`visualization/embedding_visualization.mp4`. Source videos provide sampled
reference frames; image-to-video adaptations repeat the canonical still in the
video's input panel.

Run:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv_vjepa2/bin/python experiments/cross_media/run.py
```

On a CUDA environment, pass `--device cuda --precision auto`. Runtime settings
are forwarded to every model/input extraction and recorded per result. Reusing
embeddings produced with a different runtime requires `--force` or another
output root.

Outputs are grouped by input and model:

```text
experiments/outputs/cross_media/384/<input>/<model>/
```

The `384/comparisons/` directory contains one all-model figure per input plus
`matrix_pca_rgb.png` and `matrix_pc1.png` for the complete 5-by-4 comparison.
`manifest.json`, `commands.sh`, per-result `result.json` files, script hashes,
checkpoint hashes, source revisions, artifact hashes, and command logs retain
the exact provenance.

Build the offline qualitative browser after the experiment completes:

```bash
.venv_vjepa2/bin/python experiments/cross_media/build_review.py \
  experiments/outputs/cross_media/384/manifest.json
```

Open `experiments/outputs/cross_media/384/review/index.html` directly in a
browser. It provides input-first, model-first, and full matrix views while
linking every tile back to its canonical result artifacts. The generated
`review/tiles/` maps are a small disposable browser cache; the experiment
manifest and per-result artifacts remain the source of truth.
Video-model detail views include a separate `Temporal video` link when the MP4
artifact is present.

`validate_results.py` checks that all models use one identical displayed
reference per input and that every video visualization selects temporal slice 0.

Neuro-JEPA is excluded because converting 2D images/videos into a 3D MRI volume
would not be a meaningful modality adaptation.
