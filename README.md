# JEPAs

Workspace for comparing JEPA-family model repositories and running local embedding extraction experiments.

External model code lives under `repos/` as Git submodules. Downloaded checkpoints and virtual environments are intentionally not tracked.

## Layout

```text
experiments/        Local embedding extraction code and experiment outputs
manifests/          Local manifests and notes
repos/              External model repositories as submodules
weights/            Manually managed weights, ignored by Git
weights_py/         Script-managed weights, ignored except scripts/docs
requirements/       Reproducible environment inputs
scripts/            Repo setup helpers
```

## Environment

Create the lightweight CPU download/extraction environment:

```bash
scripts/create_weights_venv.sh
source .venv/bin/activate
```

## Weights

EchoJEPA weights are managed manually. Other weight options are documented in:

```bash
weights_py/OPTIONS.md
```

Start resumable non-Echo downloads with:

```bash
weights_py/download_weights.sh vjepa2-base ijepa radjepa
```

## Experiments

Experiment code should live under `experiments/` and import model code from the source repositories under `repos/`.
