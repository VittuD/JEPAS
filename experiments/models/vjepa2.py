#!/usr/bin/env python3
"""Extract V-JEPA 2 tokens from one image or video."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch


ROOT_DIR = Path(__file__).resolve().parents[2]
VJEPA2_DIR = ROOT_DIR / "repos" / "vjepa2"
sys.path.insert(0, str(ROOT_DIR))

from experiments.device import (  # noqa: E402
    add_runtime_arguments,
    inference_context,
    resolve_runtime,
    seed_inference,
)


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
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Optional image path repeated across the video input.",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Optional video path sampled into --frames frames with ffmpeg.",
    )
    parser.add_argument(
        "--output-embedding",
        type=Path,
        default=None,
        help="Optional .npy path for non-pooled token embeddings.",
    )
    parser.add_argument(
        "--output-preprocessed-image",
        type=Path,
        default=None,
        help="Optional image path for the square preprocessed image/frame used for visualization.",
    )
    parser.add_argument("--seed", type=int, default=0)
    add_runtime_arguments(parser)
    return parser.parse_args()


def clean_backbone_key(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        cleaned[key.replace("module.", "").replace("backbone.", "")] = value
    return cleaned


def default_checkpoint(variant: str) -> Path:
    if variant == "vjepa2-vitl":
        return ROOT_DIR / "weights" / "vjepa2" / "vitl.pt"
    if variant == "vjepa2-vith":
        return ROOT_DIR / "weights" / "vjepa2" / "vith.pt"
    if variant == "vjepa2-1-vitb":
        return ROOT_DIR / "weights" / "vjepa2" / "vjepa2_1_vitb_dist_vitG_384.pt"
    if variant == "vjepa2-1-vitg":
        return ROOT_DIR / "weights" / "vjepa2" / "vjepa2_1_vitg_384.pt"
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


def load_image(path: Path, image_size: int) -> tuple[torch.Tensor, Any]:
    import numpy as np
    from PIL import Image

    image = Image.open(path).convert("RGB")
    side = min(image.size)
    left = (image.width - side) // 2
    top = (image.height - side) // 2
    image = image.crop((left, top, left + side, top + side))
    image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
    tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().div(255.0)
    mean = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
    return (tensor - mean) / std, image


def load_video(path: Path, image_size: int, frames: int) -> tuple[torch.Tensor, Any]:
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    duration = max(float(probe.stdout.strip() or "1"), 1.0)
    fps = frames / duration
    with tempfile.TemporaryDirectory(prefix="jepas_frames_") as tmp:
        frame_pattern = Path(tmp) / "frame_%04d.jpg"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-vf",
            f"fps={fps:.8f},scale={image_size}:{image_size}:force_original_aspect_ratio=increase,crop={image_size}:{image_size}",
            "-frames:v",
            str(frames),
            str(frame_pattern),
        ]
        subprocess.run(cmd, check=True)
        frame_paths = sorted(Path(tmp).glob("frame_*.jpg"))
        if not frame_paths:
            raise RuntimeError(f"ffmpeg did not extract frames from {path}")
        loaded_pairs = [load_image(frame_path, image_size) for frame_path in frame_paths]
        while len(loaded_pairs) < frames:
            loaded_pairs.append(loaded_pairs[-1])
        tensors = [item[0] for item in loaded_pairs[:frames]]
        preview = loaded_pairs[0][1]
        return torch.stack(tensors, dim=1).unsqueeze(0), preview


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

    runtime = resolve_runtime(args.device, args.precision)
    seed_inference(args.seed, runtime)

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
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = (
        checkpoint.get("ema_encoder")
        or checkpoint.get("target_encoder")
        or checkpoint.get("encoder")
    )
    if state_dict is None:
        raise KeyError(f"Checkpoint has no target_encoder or encoder key. Keys: {sorted(checkpoint)}")

    load_msg = model.load_state_dict(clean_backbone_key(state_dict), strict=False)
    del checkpoint, state_dict
    model = model.to(runtime.device).eval()

    if args.image is not None and args.video is not None:
        raise ValueError("Pass only one of --image or --video.")
    if args.video is not None:
        video, preprocessed_image = load_video(args.video, image_size, args.frames)
    elif args.image is not None:
        image_tensor, preprocessed_image = load_image(args.image, image_size)
        image = image_tensor.unsqueeze(0)
        video = image.unsqueeze(2).repeat(1, 1, args.frames, 1, 1)
    else:
        image = torch.randn(1, 3, image_size, image_size)
        video = image.unsqueeze(2).repeat(1, 1, args.frames, 1, 1)
        preprocessed_image = None

    video = video.to(runtime.device, non_blocking=runtime.device.type == "cuda")
    with inference_context(runtime):
        embedding = model(video)
    pooled = embedding.mean(dim=1)

    if args.output_embedding is not None:
        import numpy as np

        args.output_embedding.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output_embedding, embedding.detach().float().cpu().numpy())
    if args.output_preprocessed_image is not None and preprocessed_image is not None:
        args.output_preprocessed_image.parent.mkdir(parents=True, exist_ok=True)
        preprocessed_image.save(args.output_preprocessed_image)

    print(f"variant={args.variant}")
    print(f"checkpoint={checkpoint_path}")
    print(f"image={args.image or ('none' if args.video is not None else 'random')}")
    print(f"video={args.video or 'repeated-image'}")
    print(f"device={runtime.device.type}")
    print(f"precision={runtime.precision}")
    print(
        f"missing_keys={len(load_msg.missing_keys)} "
        f"unexpected_keys={len(load_msg.unexpected_keys)}"
    )
    print(f"embedding_shape={tuple(embedding.shape)}")
    print(f"pooled_shape={tuple(pooled.shape)}")
    pooled_stats = pooled.detach().float()
    print(
        f"pooled_mean={pooled_stats.mean().item():.6f} "
        f"pooled_std={pooled_stats.std().item():.6f}"
    )
    if args.output_embedding is not None:
        print(f"output_embedding={args.output_embedding}")
    if args.output_preprocessed_image is not None and preprocessed_image is not None:
        print(f"output_preprocessed_image={args.output_preprocessed_image}")


if __name__ == "__main__":
    main()
