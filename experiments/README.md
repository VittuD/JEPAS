# Experiments

This folder is for local embedding extraction experiments.

The intended pattern is:

1. Keep external model implementations in `repos/`.
2. Keep downloaded checkpoints in `weights/` or `weights_py/`.
3. Put local scripts, manifests, outputs, and notebooks here.
4. Treat generated embeddings as experiment artifacts unless they are small metadata files.

Suggested output layout:

```text
experiments/
  extract_embeddings.py
  inputs/
  outputs/
```
