# Experiments

Each experiment has its own folder, entry point, documentation, and output
namespace.

```text
experiments/
  smoke/               checkpoint loading and token extraction checks
  extraction/          generic embedding extraction interface
  visualization/       model-independent token visualization
  common_resolution/   matched spatial-resolution comparison
  cross_media/         every 2D/video model crossed with every media input
  cross_media_native_resolution/
                       cross-media matrix at each model's native resolution
  shared/              reusable experiment helpers
  inputs/              local/generated inputs (gitignored)
  outputs/             active generated results (contents gitignored)
  outputs_old/         dated local result snapshots (gitignored)
```

Start with:

- [`extraction/README.md`](extraction/README.md) to extract token and pooled
  embeddings from files or directories.
- [`smoke/README.md`](smoke/README.md) to test a model checkpoint.
- [`visualization/README.md`](visualization/README.md) to visualize saved tokens.
- [`common_resolution/README.md`](common_resolution/README.md) to compare all
  applicable models at `384x384`.
- [`cross_media/README.md`](cross_media/README.md) to run every applicable model
  against every image and video.
- [`cross_media_native_resolution/README.md`](cross_media_native_resolution/README.md)
  to run the same matrix at each model's native spatial resolution.

The shared CPU environment is created with:

```bash
scripts/create_inference_venv.sh cpu
```

The CUDA blueprint uses:

```bash
scripts/create_inference_venv.sh cuda
```

All inference entry points accept `--device {auto,cpu,cuda}` and
`--precision {auto,fp32,bf16,fp16}`. No experiment runner creates or modifies an
environment.

External source implementations stay under `repos/`, downloaded checkpoints
stay under `weights/`, and generated experiment data remains outside Git.
The custom `ijepa_lite` affinity-novelty model is an optional catalog entry:
its source is pinned as a submodule, while its existing remote checkpoint is
referenced in place and never copied by these scripts.

`experiments/outputs/` is the active workspace used by the runners.
`experiments/outputs_old/` is reserved for dated local snapshots when clearing
the active workspace. Inputs and archived outputs are never moved or deleted by
an experiment runner.
