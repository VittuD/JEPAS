#!/usr/bin/env python3
"""Smoke test I-JEPA source inference on one random image."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
IJEPA_DIR = ROOT_DIR / "repos" / "ijepa"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT_DIR / "weights_py" / "ijepa" / "IN1K-vit.h.14-300e.pth.tar",
        help="Path to an I-JEPA checkpoint.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def clean_backbone_key(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        cleaned[key.replace("module.", "")] = value
    return cleaned


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(IJEPA_DIR))

    from src.models.vision_transformer import vit_huge

    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)

    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)

    model = vit_huge(img_size=[args.image_size], patch_size=14)
    model.eval()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint.get("target_encoder") or checkpoint.get("encoder")
    if state_dict is None:
        raise KeyError(f"Checkpoint has no target_encoder or encoder key. Keys: {sorted(checkpoint)}")

    load_msg = model.load_state_dict(clean_backbone_key(state_dict), strict=True)
    del checkpoint

    image = torch.randn(1, 3, args.image_size, args.image_size)
    embedding = model(image)
    pooled = embedding.mean(dim=1)

    print(f"checkpoint={args.checkpoint}")
    print(
        f"missing_keys={len(load_msg.missing_keys)} "
        f"unexpected_keys={len(load_msg.unexpected_keys)}"
    )
    print(f"embedding_shape={tuple(embedding.shape)}")
    print(f"pooled_shape={tuple(pooled.shape)}")
    print(f"pooled_mean={pooled.mean().item():.6f} pooled_std={pooled.std().item():.6f}")


if __name__ == "__main__":
    main()
