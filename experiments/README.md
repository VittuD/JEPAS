# Experiments

The experiment code intentionally has five entry points/components:

```text
catalog.py       model and example-media definitions
extract.py       token and pooled embedding extraction
visualize.py     input, PCA RGB, PC1, KMeans-4, and temporal MP4 rendering
cleanup.py       guarded, idempotent removal of visualized HDF5 embeddings
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

Model adapters load the source repository implementation and emit non-pooled
tokens without changing their dtype. `extract.py` stores them losslessly as the
`tokens` dataset in `embeddings.h5` and writes an FP32 mean over the token axis as
`pooled`. Both datasets use chunking, byte shuffle, and moderate gzip compression.

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

Comparison output uses this schema:

```text
experiments/outputs/<experiment>/
  manifest.json
  <input>/<model>/
    input.jpg
    embeddings.h5          # omitted after safe cleanup by default
    visualization.png
    visualization.mp4      # video models only
    metadata.json
    run.log
```

`compare.py` resumes extraction and visualization independently, then removes
`embeddings.h5` only after required artifacts and metadata validate. Pass
`--keep-embeddings` to retain it. `cleanup.py` applies the same idempotent guard
when cleanup is run separately. The permanent browser is served with
`python visualization/serve.py` on `127.0.0.1:43871`; legacy output layouts are
left untouched and are not discovered.

Neuro-JEPA is excluded from 2D comparisons because it consumes 3D MRI volumes,
but it remains available through `extract.py`.
