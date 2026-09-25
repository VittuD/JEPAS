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
