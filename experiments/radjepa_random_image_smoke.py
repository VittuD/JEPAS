#!/usr/bin/env python3
"""Smoke test RadJEPA local HF model inference on one random image."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


ROOT_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT_DIR / "weights_py" / "radjepa" / "hf-RadJEPA",
        help="Local Hugging Face RadJEPA model directory.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_dir.exists():
        raise FileNotFoundError(args.model_dir)

    sys.path.insert(0, str(args.model_dir))
    from modeling_radjepa import RadJEPAConfig, RadJEPAEncoder

    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)

    config = RadJEPAConfig.from_pretrained(str(args.model_dir), local_files_only=True)
    model = RadJEPAEncoder(config)
    checkpoint = torch.load(
        args.model_dir / "jepa_encoder.pth.tar",
        map_location="cpu",
        weights_only=True,
    )
    if "encoder" in checkpoint:
        state_dict = checkpoint["encoder"]
    elif "state_dict" in checkpoint and "encoder" in checkpoint["state_dict"]:
        state_dict = checkpoint["state_dict"]["encoder"]
    else:
        raise RuntimeError(f"Encoder weights not found. Keys: {sorted(checkpoint)}")

    state_dict = {
        key.replace("module.", "").replace("encoder.", ""): value
        for key, value in state_dict.items()
    }
    load_msg = model.model.load_state_dict(state_dict, strict=True)
    del checkpoint
    model.eval()

    image = torch.randn(1, 3, args.image_size, args.image_size)
    tokens = model.model.forward_features(image)
    embedding = tokens.mean(dim=1)

    print(f"model_dir={args.model_dir}")
    print(
        f"missing_keys={len(load_msg.missing_keys)} "
        f"unexpected_keys={len(load_msg.unexpected_keys)}"
    )
    print(f"tokens_shape={tuple(tokens.shape)}")
    print(f"embedding_shape={tuple(embedding.shape)}")
    print(f"embedding_mean={embedding.mean().item():.6f} embedding_std={embedding.std().item():.6f}")


if __name__ == "__main__":
    main()
