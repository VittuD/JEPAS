# Weight Download Options

This folder is for Python-managed downloaded weights. Use the workspace venv:

```bash
cd /home/davide/Desktop/JEPAS
source .venv/bin/activate
```

EchoJEPA weights are downloaded manually from Google Drive, so the script handles the other repos only.

## Smallest Model Per Repo

Run:

```bash
weights_py/download_weights.sh all
```

This downloads:

| Repo | Selected model | Destination |
| --- | --- | --- |
| `repos/vjepa2` | V-JEPA 2.1 ViT-B/16, 80M params, smallest V-JEPA checkpoint listed in the README | `weights_py/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt` |
| `repos/ijepa` | I-JEPA ViT-H/14 IN1K, one of the smallest I-JEPA architecture options listed in the README | `weights_py/ijepa/IN1K-vit.h.14-300e.pth.tar` |
| `repos/RadJEPA` | RadJEPA ViT-B/14 Hugging Face repo | `weights_py/radjepa/hf-RadJEPA/` |

The script uses:

- `aria2c` with segmented resume support for direct `.pt` / `.pth.tar` files.
- `hf download` with `HF_XET_HIGH_PERFORMANCE=1` for Hugging Face repos.

Completed direct downloads are skipped when the destination file exists and no `.aria2` partial-control file remains. Completed HF downloads are marked with `.download_complete` in the target folder.

## Individual Repos

```bash
weights_py/download_weights.sh vjepa2
weights_py/download_weights.sh ijepa
weights_py/download_weights.sh radjepa
```

## EchoJEPA

Echo-specific checkpoints are provided through a Google Drive folder, not Torch Hub or Hugging Face model repos in the README.

Google Drive folder:

```text
https://drive.google.com/drive/folders/1RFEXMe8TTcABMBz4H_qtiLB43K9jD_lf?usp=sharing
```

EchoJEPA V-JEPA 2 checkpoint to use first:

```text
vitl-vmix22m-pt220-c55.pt
```

Expected local path:

```text
weights_py/echojepa/vitl-vmix22m-pt220-c55.pt
```

Reproducible `gdown` command from the download venv:

```bash
mkdir -p weights_py/echojepa
.venv/bin/gdown --continue 'https://drive.google.com/uc?id=1T_ubAMpDMEByH7V6TJu9iT5Yf9IxlDBB' \
  -O weights_py/echojepa/vitl-vmix22m-pt220-c55.pt
```

Other EchoJEPA V-JEPA 2 checkpoint listed in the folder:

```text
vitl-scratch-pt-210-c25.pt
```

Its Drive file ID is:

```text
1y899SZlVL10kGfEPPNXXH4M3aWaP5ujc
```

Smallest EchoJEPA V-JEPA 2.1 checkpoint listed in the README, intentionally not used for the first EchoJEPA test:

```text
vjepa2_1_vitb_mimic_pt169_c60.pt
```
