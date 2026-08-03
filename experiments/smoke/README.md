# Model Smoke Tests

These scripts load local checkpoints through each source repository and expose
non-pooled tokens plus a mean-pooled summary.

```bash
.venv_vjepa2/bin/python experiments/smoke/ijepa_random_image_smoke.py
.venv_vjepa2/bin/python experiments/smoke/radjepa_random_image_smoke.py
.venv_vjepa2/bin/python experiments/smoke/vjepa2_random_image_smoke.py --variant vjepa2-vitl
.venv_vjepa2/bin/python experiments/smoke/vjepa2_random_image_smoke.py --variant vjepa2-vith
.venv_vjepa2/bin/python experiments/smoke/vjepa2_random_image_smoke.py --variant vjepa2-1-vitb
.venv_vjepa2/bin/python experiments/smoke/vjepa2_random_image_smoke.py --variant vjepa2-1-vitg
.venv_vjepa2/bin/python experiments/smoke/echojepa_vjepa2_random_image_smoke.py
.venv_vjepa2/bin/python experiments/smoke/neurojepa_random_volume_smoke.py
```

## I-JEPA Lite affinity-novelty model

The custom ChestMNIST model uses the exact `ijepa_lite` source revision from
the training run (`388e73b571db4cc6132f38ba56ea6331ae4610ec`). Its checkpoint
remains on Leonardo and is not managed or downloaded by this repository.

Set the checkpoint path for the current system, then run:

```bash
export IJEPA_LITE_AFFINITY_NOVELTY_CHECKPOINT=weights/ijepa_lite_affinity_novelty/last.pt

.venv_jepa_cuda/bin/python \
  experiments/smoke/ijepa_lite_affinity_novelty_smoke.py \
  --image path/to/chest_xray.png \
  --device cuda --precision auto
```

Without the environment variable, the adapter uses
`weights/ijepa_lite_affinity_novelty/last.pt`. The manifest retains the
original Leonardo location as an external provenance URI, not as a runtime
default.

The adapter reconstructs the configured `224x224` ViT-H/14 encoder with 1280
channels, 32 blocks, learned patch-only positional embeddings, and no CLS
token. It selects the EMA target encoder by default, matching
`IJEPAModel.get_eval_encoder()` for this run. Pass `--encoder context` only when
an explicit online-versus-target comparison is intended.

At non-native square resolutions, the learned `16x16` positional grid is
interpolated bicubically. When a resolution is not divisible by patch size 14,
the source patchifier uses the largest covered square grid; at 384 this is
`27x27`, covering 378 pixels. Checkpoint loading uses memory mapping so
optimizer, predictor, and masker tensors in the 10.6 GB training checkpoint are
not materialized eagerly.

All scripts accept output paths for saving token arrays and a representative
preprocessed image or slice. They generate deterministic random inputs when an
input path is omitted. EchoJEPA defaults to the VMix checkpoint; pass
`--checkpoint weights/echojepa/vitl-scratch-pt-210-c25.pt` to test the
scratch checkpoint. Run `--help` on a script for all model-specific flags.

Every script accepts `--device` and `--precision`. Checkpoints are loaded on CPU
before the model is moved to the requested device. CUDA `auto` precision selects
BF16 when supported and otherwise FP16. Saved NumPy embeddings are always
converted to FP32.
