# Experiments

The experiment code intentionally has four entry points/components:

```text
catalog.py       model and example-media definitions
extract.py       token and pooled embedding extraction
visualize.py     input, PCA RGB, PC1, KMeans-4, and temporal MP4 rendering
compare.py       matched/all-media comparisons at fixed or native resolution
models/          source-backed model adapters
```

Inputs and outputs remain under `experiments/inputs/` and
`experiments/outputs/` and are ignored by Git.

## Models

`extract.py --model` supports I-JEPA, RadJEPA, V-JEPA 2 ViT-L/H, V-JEPA 2.1
ViT-B/G, EchoJEPA, Neuro-JEPA, and the optional custom I-JEPA Lite model. The
comparison default is the smallest tested 2D/video variant from each family:
I-JEPA, RadJEPA, V-JEPA 2 ViT-L, V-JEPA 2.1 ViT-B, and EchoJEPA.

Model adapters load the source repository implementation and save non-pooled
tokens. `extract.py` computes the mean-pooled representation from those tokens.

## Comparison protocol

The catalogued examples are an ImageNet-style dog, a chest X-ray, a Diving48
video, and an EchoNet video.

- Image model on video: frame 0 is used.
- Video model on image: the adapter repeats the image across the requested frames.
- Video visualization: static output uses temporal slice 0; the MP4 uses one
  PCA/KMeans fit shared by every temporal slice.
- Inputs are center-cropped to square before resizing.

`--pairing matched` selects a domain-appropriate input for each model.
`--pairing all` creates the full model/input matrix. `--resolution 384` matches
input pixels, while `--resolution native` uses each model's training resolution.
Patch sizes, token counts, temporal training, and domains remain model-specific.

Comparison output contains one directory per input/model, per-input comparison
PNGs, a simple `index.html` linking the original PNG/MP4 artifacts, and one root
`manifest.json` with commands, checkpoints, source revisions, and geometry.

Neuro-JEPA is excluded from 2D comparisons because it consumes 3D MRI volumes,
but it remains available through `extract.py`.
