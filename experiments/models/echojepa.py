#!/usr/bin/env python3
"""Extract EchoJEPA V-JEPA 2 tokens from one image or video."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch


ROOT_DIR = Path(__file__).resolve().parents[2]
ECHOJEPA_DIR = ROOT_DIR / "repos" / "EchoJEPA"
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
        "--checkpoint",
        type=Path,
        default=ROOT_DIR / "weights" / "echojepa" / "vitl-vmix22m-pt220-c55.pt",
        help="Path to the manually downloaded EchoJEPA V-JEPA 2 checkpoint.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--image", type=Path, default=None, help="Optional image path.")
    parser.add_argument("--video", type=Path, default=None, help="Optional video path sampled with ffmpeg.")
    parser.add_argument(
        "--input-list",
        type=Path,
        default=None,
        help=(
            "Text file of newline-separated image or video paths for batched "
            "extraction. Mutually exclusive with --image/--video. Requires "
            "--list-kind. Loads the model once and runs sub-batches of "
            "--batch-size, writing one (N, T, D) token array."
        ),
    )
    parser.add_argument(
        "--list-kind",
        choices=("image", "video"),
        default=None,
        help="Kind of every path in --input-list.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Sub-batch size for --input-list forward passes (video tokens are large).",
    )
    parser.add_argument(
        "--load-workers",
        type=int,
        default=24,
        help=(
            "Thread pool size for loading each sub-batch (PIL decode / ffmpeg "
            "frame extraction), which is otherwise sequential CPU/IO work "
            "the GPU sits idle for. ffmpeg itself is capped to 2 threads per "
            "call, so load-workers * 2 is the real concurrency ceiling -- "
            "default targets roughly half the container's cores, not all of "
            "them, out of courtesy to other hssh users on shared CPU."
        ),
    )
    parser.add_argument("--output-embedding", type=Path, default=None, help="Optional .npy token output.")
    parser.add_argument(
        "--output-preprocessed-image",
        type=Path,
        default=None,
        help="Optional image path for the square preprocessed image/frame.",
    )
    parser.add_argument("--seed", type=int, default=0)
    add_runtime_arguments(parser)
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
    with tempfile.TemporaryDirectory(prefix="jepas_echo_frames_") as tmp:
        frame_pattern = Path(tmp) / "frame_%04d.jpg"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-threads",
            "2",
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
    sys.path.insert(0, str(ECHOJEPA_DIR))

    from src.models.vision_transformer import vit_large

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"{args.checkpoint} does not exist. Put the EchoJEPA V-JEPA 2 checkpoint there, "
            "or pass --checkpoint /path/to/vitl-vmix22m-pt220-c55.pt."
        )

    runtime = resolve_runtime(args.device, args.precision)
    seed_inference(args.seed, runtime)

    model = vit_large(
        img_size=args.image_size,
        patch_size=16,
        num_frames=args.frames,
        tubelet_size=2,
        uniform_power=True,
        use_rope=True,
        use_sdpa=True,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = clean_backbone_key(select_encoder_state_dict(checkpoint))
    load_msg = model.load_state_dict(state_dict, strict=False)
    del checkpoint, state_dict
    model = model.to(runtime.device).eval()

    if args.input_list is not None:
        if args.image is not None or args.video is not None:
            raise ValueError("Pass only one of --image/--video or --input-list.")
        if args.list_kind is None:
            raise ValueError("--input-list requires --list-kind image|video.")
        paths = [
            Path(line) for line in args.input_list.read_text().splitlines() if line.strip()
        ]
        if not paths:
            raise ValueError(f"No paths found in {args.input_list}")

        def load_one(p: Path) -> torch.Tensor:
            if args.list_kind == "video":
                return load_video(p, args.image_size, args.frames)[0]
            image_tensor = load_image(p, args.image_size)[0].unsqueeze(0)
            return image_tensor.unsqueeze(2).repeat(1, 1, args.frames, 1, 1)

        chunks = []
        num_batches = -(-len(paths) // args.batch_size)
        with ThreadPoolExecutor(max_workers=args.load_workers) as load_pool:
            for start in range(0, len(paths), args.batch_size):
                chunk_paths = paths[start : start + args.batch_size]
                batch = torch.cat(list(load_pool.map(load_one, chunk_paths)), dim=0).to(
                    runtime.device, non_blocking=runtime.device.type == "cuda"
                )
                with inference_context(runtime):
                    batch_embedding = model(batch)
                chunks.append(batch_embedding.detach().half().cpu())
                print(f"batch {start // args.batch_size + 1}/{num_batches}: {len(chunk_paths)} {args.list_kind}s")
        embedding = torch.cat(chunks, dim=0)
        pooled = embedding.float().mean(dim=1)
        preprocessed_image = None
    else:
        if args.image is not None and args.video is not None:
            raise ValueError("Pass only one of --image or --video.")
        if args.video is not None:
            video, preprocessed_image = load_video(args.video, args.image_size, args.frames)
        elif args.image is not None:
            image_tensor, preprocessed_image = load_image(args.image, args.image_size)
            image = image_tensor.unsqueeze(0)
            video = image.unsqueeze(2).repeat(1, 1, args.frames, 1, 1)
        else:
            image = torch.randn(1, 3, args.image_size, args.image_size)
            video = image.unsqueeze(2).repeat(1, 1, args.frames, 1, 1)
            preprocessed_image = None

        video = video.to(runtime.device, non_blocking=runtime.device.type == "cuda")
        with inference_context(runtime):
            embedding = model(video)
        pooled = embedding.mean(dim=1)

    if args.output_embedding is not None:
        import numpy as np

        args.output_embedding.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output_embedding, embedding.detach().cpu().numpy())
    if args.output_preprocessed_image is not None and preprocessed_image is not None:
        args.output_preprocessed_image.parent.mkdir(parents=True, exist_ok=True)
        preprocessed_image.save(args.output_preprocessed_image)

    print(f"checkpoint={args.checkpoint}")
    if args.input_list is not None:
        print(f"input_list={args.input_list} ({args.list_kind})")
    else:
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
