# Bulk token-smoothness run: n=1024, seed 42

Run: Leonardo job 58575628, `scripts/leonardo_bulk.sbatch`, 1 node / 4 A100, 38 min, 16/16 jobs OK, 0 skipped inputs.
Output: `$FAST/outputs/bulk_n1024_seed42/` (`aggregate/report.md`, `aggregates.csv`, `null_baselines/<model>/`).

## Design

- Same seeded 1024 samples per dataset for every model (sampling depends on dataset, kind and seed, not on the model). Native resolution, 16 frames.
- Image models (ijepa, radjepa) on imagenet and rsna. Video models (vjepa2-vitl, vjepa2-1-vitb, echojepa) on diving48, echonet, kinetics, epic-kitchens (10 s clips cut from the recordings).
- Clustering: k-means and diagonal GMM, k = 2, 3, 4.
- Each metric is compared with a null: i.i.d. Gaussian tokens at the model's exact (T, D, grid), 30 draws. `ratio` = real / null mean.
- 98,304 rows = 16 jobs x 6 method/k combinations x 1024 samples: nothing dropped.

## Findings (k=3, ratio vs noise, lower = smoother)

| Model | Boundary fraction | Connected components | Adjacent-token cosine sim. |
|---|---|---|---|
| vjepa2-1-vitb | 0.23-0.42 | 0.07-0.19 | 0.76-0.86 |
| radjepa | 0.35-0.41 | 0.17-0.20 | 0.70-0.74 |
| ijepa | 0.51-0.60 | 0.30-0.41 | ~0.57 |
| echojepa | 0.49-0.78 | 0.36-0.65 | 0.51-0.59 |
| vjepa2-vitl | 0.61-0.84 | 0.45-0.85 | 0.42-0.49 |

1. **The model ordering is stable across all datasets, and for both k-means and GMM.** Dataset matters much less than model. The one outlier is vjepa2-vitl on EchoNet with k-means (components ratio 0.85, nearly noise-like).
2. **vjepa2-1-vitb and radjepa are the smoothest; vjepa2-vitl is the roughest.** The ordering does not split image vs video: the two video families (V-JEPA 2 vs V-JEPA 2.1) are at opposite ends.
3. **Image models have much higher PCA concentration** (`pca_top3_sum` 0.28-0.44 vs 0.12-0.24 for video). Partly a token-count effect (256 tokens, D=768-1280).
4. **Silhouette is low everywhere (0.03-0.08).** Clusters are not well separated in feature space, so these metrics describe spatial organisation, not clean cluster structure.

## Caveats on the metrics

- **z-scores are unusable for `adjacent_cosine_distance` and `pca_top3_sum`** (|z| in the thousands): the null in high dimension has adjacent cosine distance ~1.0 with a tiny sd. Use the ratio, or raw values within a model/dataset. Only the boundary-fraction and component-count z-scores are informative (roughly -7 to +2).
- **Components z is near 0** (0.6-2.5) even where the ratio is ~0.5, because the null sd is wide. Boundary fraction is the metric that separates real from noise most reliably.
- **Raw values are not comparable across models** with different (T, D, grid): use the null-normalised table.
- **vjepa2-1-vitb has 4608 tokens vs 2048 (vitl) / 1568 (echojepa).** Its extreme smoothness (components ratio down to 0.07) could partly come from the larger grid or from a token-layout difference rather than better features. The null should absorb the count, but this is the result we would most want to confirm independently.
- Neuro-JEPA is excluded. Values are from one seed and one run; no confidence intervals across seeds.

## Why the next step is visual inspection

The metrics are all indirect: they are computed on cluster labels of token features. Two conclusions rest on that indirection and are surprising enough to check by eye before writing them up:

- **vjepa2-1-vitb: is it genuinely smooth, or degenerate?** Very high adjacent cosine similarity (up to 0.86) plus very few components can mean coherent object/region maps, or tokens dominated by a global/positional component (all tokens alike), which also minimises boundaries. The maps distinguish these: coherent regions vs one blob or a position-only gradient.
- **vjepa2-vitl: is it noisy, or high-frequency?** Near-noise cluster maps could be real texture-level detail (semantically fine) or artefact. Also whether the EchoNet outlier is visible.

Concretely: render, for the same sample from the same dataset, the input, PCA-RGB, PC1 and k-means map for all models side by side, on a few samples per dataset, including a low-ratio sample and a high-ratio sample per model (chosen from the per-sample metric distribution, not cherry-picked).

## Existing visualisation code we can reuse

- `experiments/visualize.py` renders input, PCA-RGB, PC1 and KMeans-4 (static 2x2 panel) and optional temporal frames from one `embeddings.h5`. It supports `--sample-index` for batch files, `--grid-shape`, `--slice-index` (which temporal slice), `--image` / `--reference-video` for the background, and `--animate`. It applies `layer_norm` per token and does its own PCA/KMeans, so it is independent of `token_metrics.py`. Its k-means is k=4, not our 2/3/4.
- `experiments/compare.py` already drives extraction + `visualize.py` per (input, model) with resumability, and `visualization/browser.py` (marimo) browses `experiments/outputs/<experiment>/<input>/<model>/` using `visualization.png` + `metadata.json`. That layout is for single hand-picked examples, not batch outputs.
- Reusable helpers: `_token_pca_maps`, `_kmeans_map`, `_select_2d_tokens`.

Gaps for our use:

1. **Sample matching.** `visualize.py` takes an integer index into one h5. Because all models share the seeded sample list, index i should be the same input across models within a dataset; this must be checked against the manifests/sample paths before relying on it.
2. **Choosing which samples to show.** Needs per-sample metric values from `token_metrics/` to select low/median/high examples per model. `visualize.py` has no such selector.
3. **Side-by-side multi-model figure.** `visualize.py` writes one figure per embedding. A small driver script would call its helpers for each model on the same input and lay them out in one grid (rows = models, columns = input / PCA / k-means).
4. **Reference input.** For video, `--reference-video` is needed to show the frame under the map; for EPIC clips and Kinetics that means the original files, which the bulk run does not copy next to the embeddings.
5. **Static vs temporal.** For video models `_select_2d_tokens` picks one temporal slice, which hides temporal smoothness (which our boundary metric includes). Use `--animate` or a slice strip.

## Proposed next steps

1. Verify per-dataset sample alignment across models (compare paths in manifests / `embeddings.h5`).
2. Write a driver (e.g. `experiments/compare_bulk_maps.py`) around the `visualize.py` helpers: choose samples by metric percentile, render a model grid per sample. Run on Leonardo (embeddings are there) or pull only the few needed samples' tokens.
3. Decide the vjepa2-1-vitb question from the maps. If degenerate, add a token-diversity metric (e.g. participation ratio or cosine to the mean token) so smoothness is not rewarded for collapse.
4. Then: more seeds (cheap, 38 min per run) for confidence intervals, and consider reporting only boundary-fraction and ratio-based metrics.

## Update: alignment verified, driver built, first maps inspected

- **Alignment:** `sources` in every `embeddings.h5` is identical, index by index, across all models of a dataset (checked for all 6 datasets, 1024 unique inputs each). Sample index i is the same input for every model.
- **Driver:** `experiments/compare_bulk_maps.py <bulk_dir> <dataset> --out-dir D --pick {low,median,high,random} --count N` (or `--indices`). It reuses the bulk run's clustering seed and PCA dim from `token_metrics/manifest.json`; its recomputed boundary fraction matches `summary.csv` (a mismatch prints a WARNING). Run it via `srun`, not on the login node (the 10-min CPU limit kills it).
- **vjepa2-1-vitb is not collapsed.** On kinetics[901] (smoothest across models) its PCA/k-means maps follow the scene: hat, sky, sea, sand, hands. On kinetics[69] (hair, no layout) it is noisy like the others, though still smoother in k-means (bf 0.34 vs 0.61-0.63). Its smoothness is content-dependent, not degenerate.
- **vjepa2-vitl** shows weak spatial structure even on the smooth sample and near-random maps on the rough one; echojepa sits between (it picks up the bottom text band on kinetics[901]).
- **New confound to control:** vjepa2-1-vitb has a 24x24 grid per tubelet vs 16x16 (vjepa2-vitl) and 14x14 (echojepa): a finer grid means adjacent tokens cover smaller, more similar image patches, which raises adjacent similarity and lowers boundary fraction regardless of the model. The Gaussian null does not remove this (it removes token count, not pixel sampling density). Control: run vjepa2-1-vitb at the vitl resolution (or vitl at vitb's) and compare.

## Update: three seeds (0, 1, 42)

Runs: `bulk_n1024_seed{0,1,42}`, each 1024 samples per dataset, ~38 min, 16/16 OK, 0 skipped inputs. Combined with `experiments/token_metrics_seeds.py` (output: `$FAST/outputs/seed_summary_n1024/seed_summary.{csv,md}`). Different seeds draw different samples, so the spread is sampling variability of the per-configuration mean; with 3 seeds the sd is a rough guide.

- **Ordering is identical in every seed on every dataset** (boundary-fraction ratio, k-means k=3).
- **Seed sd is small:** boundary-fraction ratio sd 0.001-0.027 (largest: radjepa on rsna, 0.027, on a mean of 0.44); adjacent cosine similarity sd < 0.003.
- **Correction:** vjepa2-vitl is not the roughest everywhere. On Diving48 the order is vjepa2-1-vitb < vjepa2-vitl < echojepa; on echonet, epic-kitchens and kinetics it is vjepa2-1-vitb < echojepa < vjepa2-vitl. Only these hold on every dataset: vjepa2-1-vitb is the smoothest video model, radjepa is smoother than ijepa.
- **Ratio caveat:** `adjacent_cosine_similarity` is tabulated as a raw mean, because its null is ~0 and real/null gives ratios of order 1e4.

Boundary-fraction ratio, mean over 3 seeds (sd in brackets), k-means k=3:

| dataset | echojepa | vjepa2-1-vitb | vjepa2-vitl | ijepa | radjepa |
|---|---|---|---|---|---|
| diving48 | 0.660 (0.020) | 0.319 (0.009) | 0.613 (0.004) | | |
| echonet | 0.572 (0.007) | 0.246 (0.003) | 0.835 (0.002) | | |
| epic-kitchens | 0.737 (0.010) | 0.394 (0.009) | 0.767 (0.008) | | |
| kinetics | 0.712 (0.006) | 0.318 (0.002) | 0.735 (0.007) | | |
| imagenet | | | | 0.515 (0.003) | 0.412 (0.015) |
| rsna | | | | 0.504 (0.001) | 0.437 (0.027) |
| coco | | | | 0.507 (0.005) | 0.416 (0.015) |

### COCO (val2017) added

COCO val2017 was added as a third image dataset (`DATASETS=coco` into each existing `bulk_n1024_seed{0,1,42}`; ijepa and radjepa only, ~1 min per job, no failures, no skipped inputs). It reproduces ImageNet within about one sd (boundary-fraction ratio ijepa 0.507 vs 0.515, radjepa 0.416 vs 0.412; components ratio 0.301 vs 0.317 and 0.203 vs 0.202), and radjepa < ijepa in every seed. So the image-model result does not depend on the dataset, but COCO adds little new information on this metric. The seed summary now covers 108 configurations.

## Resolution control (grid density)

Question: is vjepa2-1-vitb's smoothness (24x24 grid at its native 384) just a finer token grid? Control, seed 42, n=1024, same samples, four video datasets, each run with its own null baseline: vjepa2-1-vitb at 256 (grid 8x16x16) and vjepa2-vitl at 384 (grid 8x24x24). Jobs 58617827 / 58617829, 16/16 OK, 0 skipped inputs. Compared with `experiments/token_metrics_variants.py` (`$FAST/outputs/resolution_control/variants.md`).

Boundary-fraction ratio (k-means k=3; lower = smoother; single seed, seed sd is ~0.01-0.02):

| dataset | vitb@384 (main) | vitb@256 | vitl@256 (main) | vitl@384 |
|---|---|---|---|---|
| diving48 | 0.314 | 0.339 | 0.611 | 0.573 |
| echonet | 0.249 | 0.298 | 0.836 | 0.718 |
| epic-kitchens | 0.385 | 0.413 | 0.762 | 0.766 |
| kinetics | 0.315 | 0.355 | 0.729 | 0.726 |

- **The lead is not a grid-density artefact.** At the matched 16x16 grid vitb is 0.30-0.41 vs vitl 0.61-0.84; at the matched 24x24 grid it is 0.25-0.39 vs 0.57-0.77. The gap between the two models is 0.2-0.5, an order of magnitude larger than the resolution effect on either.
- **Resolution does matter a little for vitb:** going 384 -> 256 raises its boundary-fraction ratio by 0.03-0.05 (components ratio 0.077 -> 0.088 on diving48, similar elsewhere; adjacent cosine similarity 0.796 -> 0.788). Its ordering versus the other models is unchanged.
- **vitl at 384 is slightly smoother on Diving48 (0.611 -> 0.573) and clearly on EchoNet (0.836 -> 0.718; components 0.85 -> 0.565)**, and unchanged on EPIC and Kinetics. So the EchoNet outlier for vitl (near-noise at 256) is partly a resolution effect, consistent with EchoNet's small native frames being upscaled. This is the only place the control changes a conclusion: report vitl on EchoNet with that caveat.
- Not controlled: patch size (both 16), training data and objective. Only the token grid was varied. One seed only.

### Maps at matched resolution (seed 42)

Figures: `$WORK/maps_res_control/` (echonet: 635, 792 = roughest on average; 652, 870 = random; kinetics: 871 = random). Rows per figure: input, then PCA-RGB and k-means k=3 for vitb@256, vitb@384, vitl@256, vitl@384 (`compare_bulk_maps.py` on a symlink view of the runs). Only echonet[652] and kinetics[871] were inspected so far.

- **vitb is spatially coherent at both resolutions and tracks the image.** echonet[652]: at 256 and 384 alike the PCA map separates the ultrasound sector from the black background, with a distinct sector edge, and k-means finds chamber / wall / outside regions (bf 0.24 / 0.26). kinetics[871]: wall, couch and floor are separate regions at both sizes (bf 0.19 / 0.12). The 384 maps are finer-grained versions of the 256 maps, not different ones.
- **vitl at 256 is near-noise even where the input is trivially uniform.** echonet[652]: it does not separate the black background from tissue in the PCA map (bf 0.57, cc 356); kinetics[871] shows only a faint dark-couch patch (bf 0.56).
- **vitl at 384 recovers part of the structure.** echonet[652]: the sector interior and outside become distinguishable in PCA, k-means has a central blob, but within-region speckle remains (bf 0.48). kinetics[871]: a central region appears (bf 0.44). This matches the resolution effect in the table above, and shows it is a partial recovery, not parity with vitb.
- Consequence for interpretation: the smoothness gap reflects real differences in the maps (vitb's tokens vary at the scale of scene regions; vitl's vary at the scale of individual patches), not a metric artefact of the grid or of degenerate tokens. It does not say which is better: fine per-patch variation can be useful for dense tasks, and the maps were not tied to any label. Two samples out of the twelve rendered; not a systematic result.
