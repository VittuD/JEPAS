# Synthetic probe set

Status: v1 implemented (`experiments/synthetic_probe.py`, `scripts/make_synthetic_probe.py`,
`scripts/leonardo_bulk.sbatch`, `experiments/synthetic_scale_curves.py`) and smoke-tested
end to end on Leonardo at 4 files per family (job 58622211: 5/5 jobs OK, all 14 families
labelled in the aggregate). No full-size results yet. The questions and the ways they can
fail below were written before implementation, so they are fixed before we see results.

Run (Leonardo, `$WORK/JEPAS`):

```bash
TAG=_synthetic DATASETS="synthetic-image synthetic-video" \
  sbatch --export=ALL,N=1024,SEED=42 --account=IscrC_TBoneAI --qos=normal --time=01:00:00 scripts/leonardo_bulk.sbatch
# afterwards
python experiments/synthetic_scale_curves.py $FAST/outputs/bulk_n1024_seed42_synthetic \
  --manifest $FAST/datasets_derived/synthetic_v1_seed42/manifest.jsonl --out-dir $FAST/outputs/synthetic_curves_seed42
```

`N` only names the output directory; the synthetic sets always use every generated file.
`SYN_PER_FAMILY=<k>` generates a smaller set in its own directory (`..._pf<k>`) for smoke tests.

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

## Results (v1, seed 42, full size)

Job 58622496: generation (1024 images, 768 videos) plus 5/5 jobs OK in 8 min; 26,112 per-sample rows. Curves: `$FAST/outputs/synthetic_curves_seed42/curves.{md,csv}`. Maps: `$WORK/maps_synthetic/<family>/`. Boundary-fraction ratio to the Gaussian null, k-means k=3, mean per family; single seed, single generation, out-of-distribution inputs (see Limits). Real-data values for reference are from `BULK_N1024_FINDINGS.md`.

| Input | ijepa | radjepa | echojepa | vjepa2-1-vitb | vjepa2-vitl |
|---|---|---|---|---|---|
| uniform image / flicker (spatially uniform video) | 0.47 | 0.20 | 0.41-0.46 | 0.18-0.27 | 0.69-0.71 |
| gradient | 0.47 | 0.19 | | | |
| regions / static_pattern (flat regions) | 0.49-0.58 | 0.19-0.21 | 0.39-0.45 | 0.08-0.10 | 0.70-0.78 |
| noise (pixel / blocks / static) | 0.52 | 0.54 (pixel) | 0.54 | 0.60 | 0.86 |
| natural data (main runs) | 0.50-0.52 | 0.41-0.44 | 0.57-0.74 | 0.25-0.39 | 0.61-0.84 |

**Q1 (do the maps depend on the input?)**
- **vjepa2-1-vitb and radjepa: yes.** Both are very smooth on flat inputs (0.08-0.27) and rough on noise (0.54-0.60), and their natural-data values sit between the two. The risk that vitb's smoothness is content-independent (position or a global component) is not supported: on a two-colour diagonal image (static_pattern_0000) its PCA and k-means maps trace the diagonal edge exactly and are constant across tubelets.
- **vjepa2-vitl and ijepa: largely no, at the k-means boundary level.** vitl is 0.70 on a spatially uniform flicker video and 0.70-0.78 on flat regions, essentially the same as its 0.61-0.84 on natural video and only somewhat below its 0.86 on noise; adjacent cosine similarity is 0.55-0.69 even on a constant frame, vs 0.99 for vitb. ijepa is 0.47 on a uniform image and 0.49-0.58 on regions, vs 0.50-0.52 on natural images. For these two models the natural-data smoothness value is close to what they produce on an input with no structure, so it says little about the content. In the same diagonal-edge video, vitl's PCA map is speckle with no trace of the edge.
- **echojepa is in between**: 0.39-0.46 on flat/uniform inputs, 0.54-0.65 on noise, and its maps drift across tubelets even for a static input.
- Consequence for the bulk ranking: for vitl (video) and ijepa (image), roughness on natural data is largely intrinsic. The ranking still holds as a description of the token maps, but "vitl is rough" should not be read as "vitl encodes fine detail in the image".

**Q2 (effective resolution)**: for both image models the boundary ratio rises with the number of cells per side up to 16 (about one cell per token on a 16x16 grid) and then drops at 32 (checker: radjepa 0.17/0.26/0.43/0.66/0.19, ijepa 0.47/0.56/0.63/0.82/0.36 for 2/4/8/16/32 cells; noise_blocks and stripes peak at 16-32). Structure finer than the token grid is averaged out inside a token and the map looks smooth again, so the effective resolution is about the token grid. radjepa's PCA shows the checkerboard at 2-16 cells and moire at 32; ijepa shows it only faintly and is dominated by a border-versus-centre positional pattern (visible even on uniform inputs). The video probe has no scale sweep (noise fixed at 16 cells), so this is images only.

**Q3 (metrics on known structure)**: partly inconclusive. With a fixed k=3 clustering the ratio hardly changes with the true number of regions (radjepa 0.19-0.21 for 2-5 regions; ijepa 0.49-0.58), so the boundary/component metrics are not a region counter. radjepa's maps do follow the region and object masks visually (regions and shapes), ijepa's much less. The explicit label-agreement score (ARI / boundary alignment against the stored masks) is still the step that would settle this.

**Q4 (temporal)**: vitb responds to motion in the expected order (static 0.09, moving shape 0.09 / 0.12 / 0.15 at speeds 0.1 / 0.25 / 0.5) and slightly to a cut (0.10-0.16 vs 0.08-0.10 static); echojepa is flat with speed (0.47-0.49) but higher than static (0.39-0.45); vitl shows no difference between static, moving and cut (all 0.69-0.79). For noise, temporal vs static noise is 0.59 vs 0.60 (vitb), 0.65 vs 0.54 (echojepa), 0.95 vs 0.86 (vitl).

### What this changes
- Two models (vitl on video, ijepa on images) have a roughness floor that the input barely changes; comparisons of raw smoothness against them should be paired with the synthetic floor, e.g. report natural-data smoothness relative to the same model's value on flat inputs.
- vjepa2-1-vitb's smoothness is content-following; the earlier concern is answered for that model.
- Not yet done: label-agreement scoring (Q3), more seeds/generations, a scale sweep for video, and the same probe at the matched-resolution variants (vitb@256, vitl@384) to separate resolution from model.

## Position-locked patterns on noise (observed in the maps, 2026-09-26)

Maps: `$WORK/maps_synthetic/noise_static/synthetic-video_0384.png`, `noise_temporal/synthetic-video_0512.png` (k-means k=3, tubelets 0 / 4 / 7).

- **vjepa2-1-vitb (24x24 grid)** shows a regular lattice in the k-means map (rows and columns; the spectrum below puts its period at about 3 tokens, not 2), and a token-scale checkerboard in the PCA-RGB map. In `noise_temporal`, where every frame is different noise, tubelets 0, 4 and 7 show the *same* lattice. The tokens therefore carry a fixed pattern set by position, not by the input.
- **vjepa2-vitl (16x16)** shows a different fixed pattern (a centre-versus-border blob, clearest in `noise_temporal`); in `noise_static` it changes with the tubelet index.
- **echojepa** shows vertical (top/bottom) banding, stable across tubelets.
- Consequences: (1) the boundary ratio on noise (vitb 0.40 for this file, vitl 0.54-0.64) mixes two things, random roughness and regular structure; (2) the earlier "vitb follows the input" reading (Q1) holds for content (it traces the diagonal edge), but on unstructured input vitb is not featureless, so its low natural-data boundary ratio can partly reflect a regular lattice; (3) the mechanism is **not established**. A positional-encoding origin is plausible, not tested.

Tests (no new GPU runs; `experiments/position_lock.py`, on stored embeddings): position-locked variance fraction (chance about 1/samples), split-half cosine of the position-mean map across disjoint sample halves, and the share of spatial spectral power on the Nyquist lines (period-2 lattice) with the dominant spatial period. Run it on the noise families and on a real dataset for contrast, for the older models and the two largest ones (`_large`), to see whether the lattice is a V-JEPA 2.1 property or a size effect.

### Position-lock results (job 58679055 and the kinetics run, seed 42, 128 samples per group, existing models)

| model (grid) | input | pos_variance_fraction | split_half_cosine | nyquist_ratio | dominant_period |
|---|---|---|---|---|---|
| vjepa2-1-vitb (24x24) | noise_static / noise_temporal | 0.71 / 0.60 | 0.994 / 0.990 | 0.23 / 0.33 | 3 / 3 |
| vjepa2-vitl (16x16) | noise_static / noise_temporal | 0.24 / 0.22 | 0.952 / 0.944 | 0.88 / 0.80 | 16 / 16 |
| echojepa (14x14) | noise_static / noise_temporal | 0.29 / 0.26 | 0.963 / 0.956 | 0.77 / 0.76 | 14 / 14 |
| vjepa2-1-vitb | kinetics | 0.043 | 0.686 | 0.43 | 24 |
| vjepa2-vitl | kinetics | 0.088 | 0.844 | 0.78 | 16 |
| echojepa | kinetics | 0.168 | 0.922 | 0.74 | 14 |

Chance level for the variance fraction is 0.008.

- **All three models are position-locked on noise**: 22-71% of the token variance is a fixed per-position map (chance 0.8%), and that map is nearly identical between disjoint halves of the samples (cosine 0.94-0.99), for static and for per-frame noise alike. So the fixed pattern seen in the maps is a property of the model, not of one sample.
- **vjepa2-1-vitb has the strongest lock and a different shape.** 60-71% of its variance is fixed, and its dominant spatial period is 3 tokens (mid-frequency lattice). vitl and echojepa have a dominant period equal to their grid size (one cycle across the map), i.e. the low-frequency blob / banding seen in the maps.
- **Correction:** the "period 2" reading from the maps is not supported. No model concentrates power at the Nyquist lines (ratio below 1 everywhere); vitb's is at period 3.
- **On natural video the locked share is small** (0.04-0.17) because content dominates the variance, so the two input types are not directly comparable. vitb has the smallest share on kinetics (0.043) but a nonzero split-half cosine (0.69), so a fixed component is present there too, just much weaker relative to the content.
- **Reading for the bulk ranking:** vitb's low boundary ratio on natural data should not be attributed to content-following alone; a position-locked component that is very strong on noise is part of what the model outputs. It is weak relative to content on real video, which is why vitb's maps still trace real edges (Q1).
- Still open: the cause (positional encoding is one candidate, untested), and whether the larger models (vitG, vitg) show it. `experiments/position_lock.py --run <bulk_n1024_seed42_large>` answers the second.
