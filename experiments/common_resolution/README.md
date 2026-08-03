# Common-Resolution Visualization

This experiment extracts non-pooled tokens from every 2D image/video JEPA at
one square input resolution, then runs the shared visualization pipeline.

The default is `384x384`, the highest native spatial resolution among the local
checkpoints (V-JEPA 2.1). I-JEPA and RadJEPA interpolate their checkpoint
position grids bicubically. V-JEPA 2 and EchoJEPA use the source models' RoPE
path at the runtime grid.

```bash
.venv_vjepa2/bin/python experiments/common_resolution/run.py
```

On a CUDA environment, pass `--device cuda --precision auto`. Runtime settings
are forwarded to every extractor and recorded per result. Reusing embeddings
produced with a different runtime requires `--force` or another output root.

Results are written under:

```text
experiments/outputs/common_resolution/384/<model>/
```

The experiment-level `384/comparison.png` places all models in one figure with
consistent columns for the model input, token PCA RGB, token PC1, and KMeans.
PCA signs and cluster labels are fitted independently per model, so compare
spatial organization rather than literal colors.

Video rows display the first sampled frame and visualize temporal slice 0, so
the reference frame corresponds to the first token tubelet. Video-model result
directories also contain `visualization/embedding_visualization.mp4`, while the
existing static `embedding_visualization.png` continues to be produced.

Reproducibility records are stored beside the outputs:

- `384/manifest.json` records the environment and all model results.
- `384/commands.sh` contains the exact extraction and rendering commands.
- `384/<model>/result.json` records the input/checkpoint sources, source-code
  revision, script hashes, tensor geometry, and artifact fingerprints.
- `extract.log`, `visualize.log`, and `comparison.log` retain command output
  whenever that stage is computed.

Pass `--hash-checkpoints` once to include SHA-256 hashes for the large checkpoint
files. Existing hashes are retained on later resumable runs when file metadata
still matches.

The run is resumable. Existing embeddings and figures are skipped; pass
`--force` to recompute them. Use `--models ijepa radjepa` to run a subset.

The custom `ijepa-lite-affinity-novelty` model is available but excluded from
the default run because its checkpoint exists only on Leonardo. Select it
explicitly with `--models ijepa-lite-affinity-novelty`; its matched input is the
catalogued chest X-ray.

At 384 pixels, patch-14 models have a `27x27` spatial grid and cover 378 pixels,
while patch-16 models have a `24x24` grid and cover all 384 pixels. Matching
input resolution therefore does not match token count. Neuro-JEPA is excluded
because it consumes a 3D MRI volume.
