# Synthetic probe set

Status: design (v1). Written before implementation so the questions and the ways they
can fail are fixed before we see results.

## Why

The bulk runs (`BULK_N1024_FINDINGS.md`) give a stable ranking of models by how
spatially smooth their token maps are: vjepa2-1-vitb is smooth, vjepa2-vitl and
echojepa are rough, radjepa is smoother than ijepa. Two controls (three seeds, a
resolution control) show the ranking is not a sampling or grid-density artefact, and
the maps we inspected follow image content for vitb.

What none of that tells us is **what the smoothness is made of**, because on natural
data we do not know the true structure of the input:

- A model can look smooth because its tokens are dominated by position or a global
  component and ignore the input. Smoothness would then say nothing about content.
- A model can look rough because it is sensitive to individual patches, which may or
  may not be a defect.
- Our boundary/component metrics cannot separate "smooth because the input has large
  regions and the model follows them" from "smooth because the model does not react
  to the input".

The Gaussian null in `token_metrics_null.py` is a null in **feature space** (random
tokens at the model's geometry). It does not tell us what a model does when the
**input** has no structure, or has exactly known structure. A synthetic input set
does, because its ground truth is known by construction.

## What we test

Each question lists what we would conclude either way. Predictions are stated first
so they can be wrong.

### Q1. Do the token maps depend on the input at all?

Inputs with no spatial structure: `uniform` (one random colour), `gradient` (smooth,
no edges), `noise_pixel` (iid pixels, image only), and `noise_static` / `noise_temporal`
(video).

- If a model's boundary fraction / component ratio on *unstructured* inputs is about
  as low as on natural data, its smoothness on natural data is not evidence that it
  follows content (position or global component dominates).
- If it is high on unstructured inputs and low on `regions` / `shapes`, smoothness
  on natural data is content-driven.
- Prediction: vjepa2-1-vitb will be smoother than vitl on `uniform` too; the open
  question is whether it is *as* smooth on `noise_pixel` as on `regions`.

### Q2. What is each model's effective spatial resolution, and is the grid-density confound real?

`noise_blocks` (iid colour per block), `stripes` and `checker`, each with the number
of cells per image side swept over {2, 4, 8, 16, 32, 64}. Scale is defined in
**relative** units (cells per image), not pixels, because every model resizes to its
own resolution (224, 256 or 384).

- The metric-versus-scale curve per model shows at which structure size a model's
  maps start to follow the input. A model that follows 8x8 but not 32x32 has an
  effective resolution below its token grid.
- It is a direct, controlled version of the resolution control: the same pattern at
  the same relative scale through models with different grids.
- Prediction: curves cross the natural-data ordering at some scale; vitl's curve
  improves at 384 (as in the resolution control).

### Q3. Does the metric behave sensibly where the answer is known?

`regions` (k = 2..5 flat-colour Voronoi regions) and `shapes` (1-4 flat circles or
rectangles). The number of true regions and the true boundary length are known, so:

- `total_connected_components` should be close to the number of true regions for a
  model that follows them, and far above it for one that does not.
- `boundary_fraction` has a known lower bound (the true boundary length in tokens).
- This validates the metrics, not the models. If the metrics do not track known
  structure, that changes how much the natural-data results can be trusted.
- The explicit label-agreement score (ARI / boundary alignment) is **out of scope for
  v1** (masks are stored so it can be added, see "Not in v1").

### Q4. Temporal structure in the video models

`static_pattern` (frame repeated), `moving_shape` (constant velocity, 3 speeds),
`noise_temporal` vs `noise_static`, `flicker` (whole-frame colour changing over
time), `cut` (pattern A, then B halfway).

- `static_pattern` sets the temporal baseline: token maps should not change across
  tubelets, so any temporal boundary here is the model's own.
- `noise_temporal` vs `noise_static` separates "tokens change because the input
  changes" from "tokens change on their own".
- `flicker` and `cut` test whether the temporal component of the boundary metrics
  fires when (and only when) the input changes globally.

## Design

- **Deterministic and seeded.** Same seed, same files, byte for byte.
- **Lossless.** Images as PNG, videos as `ffv1` in Matroska. Compression would
  destroy the noise families and blur the sharp edges we rely on.
- **Generated at 384x384** (the largest native input among our models); each model's
  extractor resizes to its own resolution, exactly as for real data.
- **Scale in relative units.** Patterns are defined in cells/regions per image, so a
  "16-cell checkerboard" is the same pattern for a 224, 256 or 384 model. The one
  exception, `noise_pixel`, is flagged: bicubic downsampling smooths pixel noise by a
  model-dependent amount, so it is a qualitative probe, not a controlled one.
- **Videos are 32 frames at 16 fps (2 s),** so the extractors' uniform sampling to 16
  frames takes every other frame and velocities are per second.
- **One folder per family; family taken from the path.** `token_metrics_aggregate.py`
  labels datasets from the source path, so a single extraction job yields per-family
  rows (`syn-<family>`), with no change to the job structure.
- **Masks and parameters are stored, not used.** `masks/` (label maps for `regions`
  and `shapes`) and `manifest.jsonl` (family, scale, colours, speed per file) sit
  beside the inputs so later analysis can join on them. `masks/` is a sibling of the
  input directories so the recursive input scan does not pick the masks up as images.
- **Image models on the image set, video models on the video set,** as in the bulk
  runs.

### Families

| Set | Family | Varied parameter | Ground truth |
|---|---|---|---|
| image | `uniform` | random RGB | none (no structure) |
| image | `gradient` | linear angle / radial, two random colours | none (no edges) |
| image | `noise_pixel` | iid RGB per pixel | none; flagged |
| image | `noise_blocks` | cells per side in {2,4,8,16,32,64} | block grid |
| image | `stripes` | stripes per image in {2,4,8,16,32}, random orientation | exact period |
| image | `checker` | cells per side in {2,4,8,16,32} | exact grid |
| image | `regions` | k in {2,3,4,5} Voronoi regions, flat colours | region mask |
| image | `shapes` | 1-4 circles/rectangles on a flat background | object mask |
| video | `static_pattern` | a `regions` frame repeated | constant |
| video | `moving_shape` | speed in {0.1, 0.25, 0.5} image-widths/s | trajectory |
| video | `noise_temporal` | 16-cell block noise, new every frame | none (temporal) |
| video | `noise_static` | 16-cell block noise, same every frame | constant |
| video | `flicker` | whole-frame colour: smooth ramp / random per frame | temporal only |
| video | `cut` | `regions` frame A for 16 frames, then frame B | change at midpoint |

128 files per family, parameter values cycled evenly: 1024 images, 768 videos.

### Layout

```
$FAST/datasets_derived/synthetic_v1_seed<S>/
  image/<family>/<family>_<idx>.png
  video/<family>/<family>_<idx>.mkv
  masks/image/<family>/<family>_<idx>.png     # uint8 label maps (regions, shapes)
  manifest.jsonl                              # one record per file: kind, family, params
```

## Pipeline integration

- `scripts/make_synthetic_probe.py` generates the set (CPU only, minutes).
- `scripts/leonardo_bulk.sbatch` gets two datasets, `synthetic-image` and
  `synthetic-video`, selectable through `DATASETS`, with `--n` set to the number of
  files so every file is used. It is normally run as its own tagged run
  (`TAG=_synthetic`, `DATASETS="synthetic-image synthetic-video"`).
- `token_metrics_aggregate.infer_dataset` maps `.../synthetic_v1*/<kind>/<family>/...`
  to `syn-<family>`.
- `experiments/synthetic_scale_curves.py` joins per-sample `summary.csv` rows with
  `manifest.jsonl` to give metric-versus-parameter tables per model (Q2, Q4).

## What would make us revise the conclusions

- vjepa2-1-vitb equally smooth on `noise_pixel` / `noise_blocks` at fine scale as on
  `regions` -> its smoothness on natural data is not content-following; the
  smoothness ranking would need a "does the map depend on the input" companion metric.
- The metrics not tracking the known number of regions (Q3) -> the natural-data
  metrics are less interpretable than we assumed.
- A model rough on `uniform` -> its roughness is intrinsic, not driven by the image.

## Limits

- Synthetic inputs are out of distribution for these models. This is a probe and a
  calibration, not a benchmark, and behaviour on it need not carry over to natural
  images.
- It can invalidate our reading of the natural-data results; it cannot by itself
  confirm it.
- One generation seed in v1; family-level variability comes from the 128 files per
  family, not from repeated generation.

## Not in v1

- A label-agreement score (ARI / NMI between k-means labels and the region masks,
  boundary alignment). Masks are stored for it; we look at the v1 curves first.
- Video masks and moving-object tracking metrics.
- Textures and natural-image perturbations (e.g. blur or noise added to real images),
  which would sit between synthetic and natural.
