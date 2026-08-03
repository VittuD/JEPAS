# Weight Download Options

This folder holds local model weights. Git tracks only this documentation and
the downloader; checkpoint files and downloaded model directories stay local.
Use the workspace venv:

```bash
source .venv/bin/activate
```

EchoJEPA weights are downloaded manually from Google Drive, so the script handles the other repos only.

## Smallest Model Per Repo

Run:

```bash
weights/download_weights.sh all
```

This downloads:

| Repo | Selected model | Destination |
| --- | --- | --- |
| `repos/vjepa2` | V-JEPA 2.1 ViT-B/16, 80M params, smallest V-JEPA checkpoint listed in the README | `weights/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt` |
| `repos/ijepa` | I-JEPA ViT-H/14 IN1K, one of the smallest I-JEPA architecture options listed in the README | `weights/ijepa/IN1K-vit.h.14-300e.pth.tar` |
| `repos/RadJEPA` | RadJEPA ViT-B/14 Hugging Face repo | `weights/radjepa/hf-RadJEPA/` |

The script uses:

- `aria2c` with segmented resume support for direct `.pt` / `.pth.tar` files.
- `hf download` with `HF_XET_HIGH_PERFORMANCE=1` for Hugging Face repos.

Completed direct downloads are skipped when the destination file exists and no `.aria2` partial-control file remains. Completed HF downloads are marked with `.download_complete` in the target folder.

## Individual Repos

```bash
weights/download_weights.sh vjepa2
weights/download_weights.sh ijepa
weights/download_weights.sh radjepa
```

## Additional V-JEPA Checkpoints

The default `vjepa2` target downloads the smallest listed V-JEPA 2.1 checkpoint, which is distilled:

```text
weights/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt
```

For non-distilled V-JEPA 2 checkpoints:

```bash
weights/download_weights.sh vjepa2-vitl vjepa2-vith
```

This downloads:

| Model | Destination |
| --- | --- |
| V-JEPA 2 ViT-L/16 | `weights/vjepa2/vitl.pt` |
| V-JEPA 2 ViT-H/16 | `weights/vjepa2/vith.pt` |

For the smallest non-distilled V-JEPA 2.1 checkpoint:

```bash
weights/download_weights.sh vjepa2-1-vitg
```

This downloads:

```text
weights/vjepa2/vjepa2_1_vitg_384.pt
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
weights/echojepa/vitl-vmix22m-pt220-c55.pt
```

Reproducible `gdown` command from the download venv:

```bash
mkdir -p weights/echojepa
.venv/bin/gdown --continue 'https://drive.google.com/uc?id=1T_ubAMpDMEByH7V6TJu9iT5Yf9IxlDBB' \
  -O weights/echojepa/vitl-vmix22m-pt220-c55.pt
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
