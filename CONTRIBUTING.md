# Contributing

Keep this repository small and source-backed. Do not commit weights, virtual
environments, input datasets, or generated outputs.

## Add a model

1. Add the upstream implementation under `repos/` as a pinned submodule.
2. Add one adapter under `experiments/models/` that loads the source model,
   accepts `--device` and `--precision`, and saves non-pooled tokens.
3. Register checkpoint, modality, resolution, patch geometry, and adapter
   arguments once in `experiments/catalog.py`.
4. Add a focused CPU-safe test for unusual checkpoint keys or tensor geometry.
5. Verify checkpoint deserialization and a real forward pass on the intended
   host before describing the model as tested.

## Add an experiment

Prefer a small script that calls `extract_one()` and existing visualization
functions. Add shared infrastructure only when two active experiments need it.
Record the command, checkpoint, source revision, runtime, and token geometry.
Store repository paths relative to the checkout root.

Generated experiments must use `experiments/outputs/<experiment>/<input>/<model>/`.
The experiment root owns `manifest.json`; each result owns `input.jpg`,
`embeddings.h5`, `visualization.png`, optional numbered temporal-frame PNGs,
`metadata.json`, and `run.log`. HDF5 metadata must record token/pooled shapes and dtypes, model,
input, checkpoint, and extraction settings. Visualization metadata must declare
its required static/temporal artifacts and carry forward extraction metadata so the
HDF5 file can be safely removed. Keep browsing logic in the tracked marimo
notebook and its schema scanner; do not add generated HTML or composite pages.

## Check changes

```bash
scripts/check_repo.sh
git diff --check
git status --short
```
