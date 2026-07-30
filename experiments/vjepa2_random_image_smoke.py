#!/usr/bin/env python3
"""Smoke test V-JEPA 2 source inference on one random image."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
VJEPA2_DIR = ROOT_DIR / "repos" / "vjepa2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=("vjepa2-vitl", "vjepa2-vith", "vjepa2-1-vitb", "vjepa2-1-vitg"),
        default="vjepa2-1-vitb",
        help="Source model variant to instantiate.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to a V-JEPA 2 checkpoint. Defaults to the expected local path for --variant.",
    )
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def clean_backbone_key(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        cleaned[key.replace("module.", "").replace("backbone.", "")] = value
    return cleaned


def default_checkpoint(variant: str) -> Path:
    if variant == "vjepa2-vitl":
        return ROOT_DIR / "weights_py" / "vjepa2" / "vitl.pt"
    if variant == "vjepa2-vith":
        return ROOT_DIR / "weights_py" / "vjepa2" / "vith.pt"
    if variant == "vjepa2-1-vitb":
        return ROOT_DIR / "weights_py" / "vjepa2" / "vjepa2_1_vitb_dist_vitG_384.pt"
    if variant == "vjepa2-1-vitg":
        return ROOT_DIR / "weights_py" / "vjepa2" / "vjepa2_1_vitg_384.pt"
    raise ValueError(f"Unknown variant: {variant}")


def default_image_size(variant: str) -> int:
    if variant == "vjepa2-vitl":
        return 256
    if variant == "vjepa2-vith":
        return 256
    if variant == "vjepa2-1-vitb":
        return 384
    if variant == "vjepa2-1-vitg":
        return 384
    raise ValueError(f"Unknown variant: {variant}")


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(VJEPA2_DIR))

    from src.hub.backbones import (
        vjepa2_1_vit_base_384,
        vjepa2_1_vit_giant_384,
        vjepa2_vit_huge,
        vjepa2_vit_large,
    )

    checkpoint_path = args.checkpoint or default_checkpoint(args.variant)
    image_size = args.image_size or default_image_size(args.variant)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)

    if args.variant == "vjepa2-vitl":
        model, _predictor = vjepa2_vit_large(pretrained=False)
    elif args.variant == "vjepa2-vith":
        model, _predictor = vjepa2_vit_huge(pretrained=False)
    elif args.variant == "vjepa2-1-vitb":
        model, _predictor = vjepa2_1_vit_base_384(pretrained=False)
    elif args.variant == "vjepa2-1-vitg":
        model, _predictor = vjepa2_1_vit_giant_384(pretrained=False)
    else:
        raise ValueError(f"Unknown variant: {args.variant}")
    model.eval()

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = (
        checkpoint.get("ema_encoder")
        or checkpoint.get("target_encoder")
        or checkpoint.get("encoder")
    )
    if state_dict is None:
        raise KeyError(f"Checkpoint has no target_encoder or encoder key. Keys: {sorted(checkpoint)}")

    load_msg = model.load_state_dict(clean_backbone_key(state_dict), strict=False)
    del checkpoint

    image = torch.randn(1, 3, image_size, image_size)
    video = image.unsqueeze(2).repeat(1, 1, args.frames, 1, 1)

    embedding = model(video)
    pooled = embedding.mean(dim=1)

    print(f"variant={args.variant}")
    print(f"checkpoint={checkpoint_path}")
    print(
        f"missing_keys={len(load_msg.missing_keys)} "
        f"unexpected_keys={len(load_msg.unexpected_keys)}"
    )
    print(f"embedding_shape={tuple(embedding.shape)}")
    print(f"pooled_shape={tuple(pooled.shape)}")
    print(f"pooled_mean={pooled.mean().item():.6f} pooled_std={pooled.std().item():.6f}")


if __name__ == "__main__":
    main()
