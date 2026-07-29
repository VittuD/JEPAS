# Weight Download Options

This folder is for Python-managed downloaded weights. Use the workspace venv:

```bash
cd /home/davide/Desktop/JEPAS
source .venv/bin/activate
```

Use the download script for resumable direct downloads and HF downloads:

```bash
weights_py/download_weights.sh --help
weights_py/download_weights.sh vjepa2-base ijepa radjepa
```

Direct `.pt` and `.pth.tar` files use `aria2c` with segmented resume support. Hugging Face repos use `hf download` with `HF_XET_HIGH_PERFORMANCE=1` by default.

## V-JEPA 2 (`repos/vjepa2`)

The README provides direct PyTorch checkpoint URLs, PyTorch Hub loaders, and Hugging Face model repos.

### Direct PyTorch checkpoints

```bash
curl -L https://dl.fbaipublicfiles.com/vjepa2/vitl.pt -o weights_py/vjepa2-vitl.pt
curl -L https://dl.fbaipublicfiles.com/vjepa2/vith.pt -o weights_py/vjepa2-vith.pt
curl -L https://dl.fbaipublicfiles.com/vjepa2/vitg.pt -o weights_py/vjepa2-vitg.pt
curl -L https://dl.fbaipublicfiles.com/vjepa2/vitg-384.pt -o weights_py/vjepa2-vitg-384.pt
```

### V-JEPA 2.1 direct PyTorch checkpoints

```bash
curl -L https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt -o weights_py/vjepa2_1_vitb_dist_vitG_384.pt
curl -L https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt -o weights_py/vjepa2_1_vitl_dist_vitG_384.pt
curl -L https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitg_384.pt -o weights_py/vjepa2_1_vitg_384.pt
curl -L https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitG_384.pt -o weights_py/vjepa2_1_vitG_384.pt
```

### Action-conditioned checkpoint

```bash
curl -L https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt -o weights_py/vjepa2-ac-vitg.pt
```

### Evaluation probe checkpoints

```bash
curl -L https://dl.fbaipublicfiles.com/vjepa2/evals/ssv2-vitl-16x2x3.pt -o weights_py/ssv2-vitl-16x2x3.pt
curl -L https://dl.fbaipublicfiles.com/vjepa2/evals/diving48-vitl-256.pt -o weights_py/diving48-vitl-256.pt
curl -L https://dl.fbaipublicfiles.com/vjepa2/evals/ek100-vitl-256.pt -o weights_py/ek100-vitl-256.pt
curl -L https://dl.fbaipublicfiles.com/vjepa2/evals/ssv2-vitg-384-64x2x3.pt -o weights_py/ssv2-vitg-384-64x2x3.pt
curl -L https://dl.fbaipublicfiles.com/vjepa2/evals/diving48-vitg-384-32x4x3.pt -o weights_py/diving48-vitg-384-32x4x3.pt
curl -L https://dl.fbaipublicfiles.com/vjepa2/evals/ek100-vitg-384.pt -o weights_py/ek100-vitg-384.pt
```

### PyTorch Hub loader names

```python
import torch

processor = torch.hub.load("facebookresearch/vjepa2", "vjepa2_preprocessor")

vjepa2_vit_large = torch.hub.load("facebookresearch/vjepa2", "vjepa2_vit_large")
vjepa2_vit_huge = torch.hub.load("facebookresearch/vjepa2", "vjepa2_vit_huge")
vjepa2_vit_giant = torch.hub.load("facebookresearch/vjepa2", "vjepa2_vit_giant")
vjepa2_vit_giant_384 = torch.hub.load("facebookresearch/vjepa2", "vjepa2_vit_giant_384")

vjepa2_1_vit_base_384 = torch.hub.load("facebookresearch/vjepa2", "vjepa2_1_vit_base_384")
vjepa2_1_vit_large_384 = torch.hub.load("facebookresearch/vjepa2", "vjepa2_1_vit_large_384")
vjepa2_1_vit_giant_384 = torch.hub.load("facebookresearch/vjepa2", "vjepa2_1_vit_giant_384")
vjepa2_1_vit_gigantic_384 = torch.hub.load("facebookresearch/vjepa2", "vjepa2_1_vit_gigantic_384")

vjepa2_encoder, vjepa2_ac_predictor = torch.hub.load("facebookresearch/vjepa2", "vjepa2_ac_vit_giant")
```

Local checkout caveat: `repos/vjepa2/src/hub/backbones.py` currently sets `VJEPA_BASE_URL = "http://localhost:8300"` for testing. Remote `torch.hub.load("facebookresearch/vjepa2", ...)` should use the GitHub version, but local hub loading from this checkout will not download public weights unless that base URL is changed back to `https://dl.fbaipublicfiles.com/vjepa2`.

### Hugging Face repos

Base models listed in the README:

```bash
hf download facebook/vjepa2-vitl-fpc64-256 --local-dir weights_py/hf-vjepa2-vitl-fpc64-256
hf download facebook/vjepa2-vith-fpc64-256 --local-dir weights_py/hf-vjepa2-vith-fpc64-256
hf download facebook/vjepa2-vitg-fpc64-256 --local-dir weights_py/hf-vjepa2-vitg-fpc64-256
hf download facebook/vjepa2-vitg-fpc64-384 --local-dir weights_py/hf-vjepa2-vitg-fpc64-384
```

Additional official HF classification/probe repos visible in the Facebook V-JEPA 2 collection:

```bash
hf download facebook/vjepa2-vitg-fpc64-384-ssv2 --local-dir weights_py/hf-vjepa2-vitg-fpc64-384-ssv2
hf download facebook/vjepa2-vitl-fpc16-256-ssv2 --local-dir weights_py/hf-vjepa2-vitl-fpc16-256-ssv2
hf download facebook/vjepa2-vitg-fpc32-384-diving48 --local-dir weights_py/hf-vjepa2-vitg-fpc32-384-diving48
hf download facebook/vjepa2-vitl-fpc32-256-diving48 --local-dir weights_py/hf-vjepa2-vitl-fpc32-256-diving48
```

## EchoJEPA (`repos/EchoJEPA`)

Echo-specific checkpoints are provided through a Google Drive folder, not Torch Hub or Hugging Face model repos in the README.

Google Drive folder:

```text
https://drive.google.com/drive/folders/1RFEXMe8TTcABMBz4H_qtiLB43K9jD_lf?usp=sharing
```

Echo-specific checkpoint names from the README:

```text
vitl-vmix22m-pt220-c55.pt
vitl-scratch-pt-210-c25.pt
vjepa21_vitl_mimic_pt100.pt
vjepa21_vitl_mimic_pt117.pt
vjepa2_1_vitb_mimic_pt169_c60.pt
```

The EchoJEPA checkout also carries V-JEPA 2 upstream initializer options:

```bash
curl -L https://dl.fbaipublicfiles.com/vjepa2/vitl.pt -o weights_py/echo-init-vjepa2-vitl.pt
curl -L https://dl.fbaipublicfiles.com/vjepa2/vith.pt -o weights_py/echo-init-vjepa2-vith.pt
curl -L https://dl.fbaipublicfiles.com/vjepa2/vitg.pt -o weights_py/echo-init-vjepa2-vitg.pt
curl -L https://dl.fbaipublicfiles.com/vjepa2/vitg-384.pt -o weights_py/echo-init-vjepa2-vitg-384.pt
```

Local `repos/EchoJEPA/hubconf.py` exposes only the V-JEPA 2 upstream Torch Hub loaders:

```python
torch.hub.load("facebookresearch/vjepa2", "vjepa2_vit_large")
torch.hub.load("facebookresearch/vjepa2", "vjepa2_vit_huge")
torch.hub.load("facebookresearch/vjepa2", "vjepa2_vit_giant")
torch.hub.load("facebookresearch/vjepa2", "vjepa2_vit_giant_384")
torch.hub.load("facebookresearch/vjepa2", "vjepa2_ac_vit_giant")
```

## I-JEPA (`repos/ijepa`)

The README provides direct PyTorch checkpoint URLs only. There is no local `hubconf.py` and no Hugging Face model repo listed in the README.

```bash
curl -L https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar -o weights_py/ijepa-IN1K-vit.h.14-300e.pth.tar
curl -L https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.16-448px-300e.pth.tar -o weights_py/ijepa-IN1K-vit.h.16-448px-300e.pth.tar
curl -L https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.h.14-900e.pth.tar -o weights_py/ijepa-IN22K-vit.h.14-900e.pth.tar
curl -L https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.g.16-600e.pth.tar -o weights_py/ijepa-IN22K-vit.g.16-600e.pth.tar
```

## RadJEPA (`repos/RadJEPA`)

The README provides Hugging Face Transformers loading. The HF file listing includes `jepa_encoder.pth.tar`.

```bash
hf download AIDElab-IITBombay/RadJEPA --local-dir weights_py/hf-RadJEPA
```

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "AIDElab-IITBombay/RadJEPA",
    trust_remote_code=True,
)
```

No direct `dl.fbaipublicfiles.com` URL or Torch Hub loader is listed for RadJEPA in the local README.
