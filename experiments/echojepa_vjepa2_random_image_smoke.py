#!/usr/bin/env python3
"""Smoke test EchoJEPA V-JEPA 2 source inference on one random image."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
ECHOJEPA_DIR = ROOT_DIR / "repos" / "EchoJEPA"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT_DIR / "weights_py" / "echojepa" / "vitl-vmix22m-pt220-c55.pt",
        help="Path to the manually downloaded EchoJEPA V-JEPA 2 checkpoint.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def clean_backbone_key(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        cleaned[
            key.replace("module.", "")
            .replace("backbone.", "")
            .replace("encoder.", "")
        ] = value
    return cleaned


def select_encoder_state_dict(checkpoint: dict) -> dict[str, torch.Tensor]:
    for key in ("target_encoder", "encoder", "context_encoder"):
        if key in checkpoint:
            return checkpoint[key]
    if "state_dict" in checkpoint:
        return select_encoder_state_dict(checkpoint["state_dict"])
    raise KeyError(f"No encoder state dict found. Keys: {sorted(checkpoint)}")


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(ECHOJEPA_DIR))

    from src.models.vision_transformer import vit_large

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"{args.checkpoint} does not exist. Put the EchoJEPA V-JEPA 2 checkpoint there, "
            "or pass --checkpoint /path/to/vitl-vmix22m-pt220-c55.pt."
        )

    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)

    model = vit_large(
        img_size=args.image_size,
        patch_size=16,
        num_frames=args.frames,
        tubelet_size=2,
        uniform_power=True,
        use_rope=True,
        use_sdpa=True,
    )
    model.eval()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = clean_backbone_key(select_encoder_state_dict(checkpoint))
    load_msg = model.load_state_dict(state_dict, strict=False)
    del checkpoint

    image = torch.randn(1, 3, args.image_size, args.image_size)
    video = image.unsqueeze(2).repeat(1, 1, args.frames, 1, 1)

    embedding = model(video)
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
