# Contributing

Keep changes reproducible, source-backed, and small enough to review. Model
weights, input datasets, virtual environments, and generated experiment
outputs do not belong in Git.

## Setup

```bash
git submodule update --init --recursive
scripts/create_inference_venv.sh cpu
```

Use `scripts/create_inference_venv.sh cuda` only on a CUDA host. The CPU and
CUDA requirement files should expose the same Python packages, differing only
where the PyTorch build requires it.

## Repository Boundaries

- `repos/`: upstream implementations as pinned Git submodules. Do not vendor or
  casually patch upstream source here.
- `weights/`: local checkpoints plus the tracked downloader and its docs.
- `manifests/`: tracked metadata for custom or externally managed models.
- `experiments/smoke/`: direct source-repository checkpoint loading and forward
  passes.
- `experiments/extraction/`: the generic embedding extraction interface.
- `experiments/<name>/`: self-contained aggregate or visualization experiments.
- `experiments/shared/`: helpers used by more than one experiment.
- `experiments/inputs/` and `experiments/outputs/`: local generated data,
  ignored by Git.

## Adding A Model

1. Add the implementation under `repos/` as a submodule pinned to a tested
   revision.
2. Keep checkpoint downloads under `weights/`. For access-controlled or
   remote-only checkpoints, add a manifest and allow the path to be overridden
   with an environment variable.
3. Add a smoke adapter that uses the source implementation, loads the
   checkpoint on CPU, supports `--device` and `--precision`, and exposes
   non-pooled tokens. Print the pooled shape as a basic diagnostic.
4. Register the model in `experiments/shared/catalog.py` and, when applicable,
   `experiments/extraction/extract_embeddings.py`.
5. Make models with unavailable-by-default checkpoints opt-in so existing
   aggregate commands remain runnable.
6. Document native resolution, patch or tubelet geometry, preprocessing,
   checkpoint source, and any positional-embedding adaptation.
7. Add focused tests for checkpoint key selection and tensor geometry without
   requiring the full checkpoint.

Do not report a model as tested based only on file presence. A completed model
validation includes checkpoint deserialization and a source-backed forward
pass on the intended host.

## Adding An Experiment

Create `experiments/<name>/README.md` and an explicit entry point such as
`run.py`. Write generated files only below `experiments/outputs/<name>/`.

An experiment should record enough provenance to reproduce each result:

- exact command and working directory;
- input and checkpoint paths and fingerprints;
- source-repository revision;
- runtime device and precision;
- preprocessing and tensor/grid geometry;
- hashes of the experiment and visualization code.

Store paths inside the repository relative to its root. Keep absolute paths
only for external inputs that cannot be represented within the checkout.
Generated command scripts should discover the checkout root rather than embed
the path of the machine that produced them.

Keep modality conversions explicit. For temporal visualizations, ensure the
displayed reference frame corresponds to the token slice being visualized.

## Validation

Run the dependency-free repository checks from the project root:

```bash
scripts/check_repo.sh
```

Then run the narrowest relevant smoke test. Use `--dry-run --device cuda` for
command validation when the current machine must not execute CUDA inference.
Do not install dependencies or download checkpoints as part of a test script.

Before committing, inspect both tracked and staged changes:

```bash
git status --short
git diff --check
git diff --cached --check
git diff --cached --stat
```

Stage only the source, documentation, manifests, submodule pointers, and
environment blueprints required by the change.
