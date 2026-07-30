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

Create the shared CPU inference environment for source-repo smoke tests:

```bash
scripts/create_vjepa2_venv.sh
source .venv_vjepa2/bin/activate
```

Or choose the inference profile explicitly:

```bash
scripts/create_inference_venv.sh cpu
source .venv_vjepa2/bin/activate

scripts/create_inference_venv.sh cuda
source .venv_jepa_cuda/bin/activate
```

The CUDA blueprint only installs a GPU-enabled PyTorch wheel. The experiment
scripts still default to CPU execution until they are given explicit device
flags.

## Weights

EchoJEPA weights are managed manually from Google Drive. The download venv includes `gdown` for that path. Other scripted weight options are documented in:

```bash
weights_py/OPTIONS.md
```

Start resumable non-Echo downloads with:

```bash
weights_py/download_weights.sh all
```

## Experiments

Experiment code should live under `experiments/` and import model code from the source repositories under `repos/`.
Current smoke tests cover V-JEPA 2, I-JEPA, RadJEPA, EchoJEPA V-JEPA 2, and Neuro-JEPA.
