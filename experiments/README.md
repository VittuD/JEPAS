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

## Smoke Tests

The smoke tests instantiate source-repo model code on CPU, load local checkpoints, run one random image, and print non-pooled token shapes plus a mean-pooled summary where available.

## V-JEPA 2

Create the CPU inference environment:

```bash
scripts/create_vjepa2_venv.sh
```

Run a random-image smoke test against the smallest scripted V-JEPA 2 checkpoint:

```bash
.venv_vjepa2/bin/python experiments/vjepa2_random_image_smoke.py \
  --checkpoint weights_py/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt
```

The script creates one random image, repeats it across the video input expected by the selected V-JEPA 2 checkpoint, and prints token and pooled embedding shapes.

## I-JEPA

The same CPU inference environment is sufficient for I-JEPA random-image inference:

```bash
.venv_vjepa2/bin/python experiments/ijepa_random_image_smoke.py \
  --checkpoint weights_py/ijepa/IN1K-vit.h.14-300e.pth.tar
```

The script prints the non-pooled patch-token representation and a mean-pooled summary.

## RadJEPA

The same CPU inference environment also supports RadJEPA after installing the shared requirements:

```bash
.venv_vjepa2/bin/python experiments/radjepa_random_image_smoke.py \
  --model-dir weights_py/radjepa/hf-RadJEPA
```

RadJEPA's provided HF model returns a mean-pooled image embedding.

## Neuro-JEPA

Neuro-JEPA uses a 3D MRI volume backbone. Put the gated Hugging Face download here:

```text
weights_py/neurojepa/Neuro-JEPA/model.safetensors
```

Then run a random-volume smoke test:

```bash
.venv_vjepa2/bin/python experiments/neurojepa_random_volume_smoke.py
```

The script creates one random `[1, 1, 96, 108, 96]` volume and prints the non-pooled 3D patch-token representation plus a mean-pooled summary.

## EchoJEPA V-JEPA 2

EchoJEPA weights are downloaded manually. For the V-JEPA 2 variant, put the checkpoint here:

```text
weights_py/echojepa/vitl-vmix22m-pt220-c55.pt
```

Then run:

```bash
.venv_vjepa2/bin/python experiments/echojepa_vjepa2_random_image_smoke.py
```

This script intentionally targets the EchoJEPA V-JEPA 2 ViT-L path, not the V-JEPA 2.1 checkpoints.
