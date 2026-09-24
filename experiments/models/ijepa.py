#!/usr/bin/env python3
"""Extract I-JEPA tokens from one image."""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch


ROOT_DIR = Path(__file__).resolve().parents[2]
IJEPA_DIR = ROOT_DIR / "repos" / "ijepa"
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
        default=ROOT_DIR / "weights" / "ijepa" / "IN1K-vit.h.14-300e.pth.tar",
        help="Path to an I-JEPA checkpoint.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Optional image path. If omitted, use a seeded random image.",
    )
    parser.add_argument(
        "--input-list",
        type=Path,
        default=None,
        help=(
            "Text file of newline-separated image paths for batched extraction. "
            "Mutually exclusive with --image. Loads the model once and runs "
            "sub-batches of --batch-size, writing one (N, T, D) token array."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Sub-batch size for --input-list forward passes.",
    )
    parser.add_argument(
        "--load-workers",
        type=int,
        default=24,
        help=(
            "Thread pool size for decoding each sub-batch's images, which is "
            "otherwise sequential CPU work the GPU sits idle for. Default "
            "targets a fraction of the container's cores, not all of them, "
            "out of courtesy to other hssh users on shared CPU."
        ),
    )
    parser.add_argument("--image-size", type=int, default=224)
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
        help="Optional image path for the square preprocessed model input.",
    )
    parser.add_argument("--seed", type=int, default=0)
    add_runtime_arguments(parser)
    return parser.parse_args()


def clean_backbone_key(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        cleaned[key.replace("module.", "")] = value
    return cleaned


def load_image(path: Path, image_size: int) -> torch.Tensor:
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
    return ((tensor - mean) / std).unsqueeze(0), image


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(IJEPA_DIR))

    from src.models.vision_transformer import vit_huge

    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)

    runtime = resolve_runtime(args.device, args.precision)
    seed_inference(args.seed, runtime)

    model = vit_huge(img_size=[args.image_size], patch_size=14)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint.get("target_encoder") or checkpoint.get("encoder")
    if state_dict is None:
        raise KeyError(f"Checkpoint has no target_encoder or encoder key. Keys: {sorted(checkpoint)}")

    state_dict = clean_backbone_key(state_dict)
    sys.path.insert(0, str(ROOT_DIR))
    from experiments.positional_embeddings import resize_square_patch_pos_embed

    pos_resize = resize_square_patch_pos_embed(
        state_dict,
        key="pos_embed",
        target_tokens=model.pos_embed.shape[1],
    )
    load_msg = model.load_state_dict(state_dict, strict=True)
    del checkpoint, state_dict
    model = model.to(runtime.device).eval()

    if args.input_list is not None:
        if args.image is not None:
            raise ValueError("Pass only one of --image or --input-list.")
        paths = [
            Path(line) for line in args.input_list.read_text().splitlines() if line.strip()
        ]
        if not paths:
            raise ValueError(f"No paths found in {args.input_list}")
        chunks = []
        num_batches = -(-len(paths) // args.batch_size)
        with ThreadPoolExecutor(max_workers=args.load_workers) as load_pool:
            for start in range(0, len(paths), args.batch_size):
                chunk_paths = paths[start : start + args.batch_size]
                loaded = load_pool.map(lambda p: load_image(p, args.image_size)[0], chunk_paths)
                batch = torch.cat(list(loaded), dim=0).to(
                    runtime.device, non_blocking=runtime.device.type == "cuda"
                )
                with inference_context(runtime):
                    batch_embedding = model(batch)
                chunks.append(batch_embedding.detach().half().cpu())
                print(f"batch {start // args.batch_size + 1}/{num_batches}: {len(chunk_paths)} images")
        embedding = torch.cat(chunks, dim=0)
        pooled = embedding.float().mean(dim=1)
        preprocessed_image = None
    else:
        if args.image is None:
            image = torch.randn(1, 3, args.image_size, args.image_size)
            preprocessed_image = None
        else:
            image, preprocessed_image = load_image(args.image, args.image_size)
        image = image.to(runtime.device, non_blocking=runtime.device.type == "cuda")
        with inference_context(runtime):
            embedding = model(image)
        pooled = embedding.mean(dim=1)

    if args.output_embedding is not None:
        import numpy as np

        args.output_embedding.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output_embedding, embedding.detach().cpu().numpy())
    if args.output_preprocessed_image is not None and preprocessed_image is not None:
        args.output_preprocessed_image.parent.mkdir(parents=True, exist_ok=True)
        preprocessed_image.save(args.output_preprocessed_image)

    print(f"checkpoint={args.checkpoint}")
    print(f"image={args.image or ('input-list' if args.input_list is not None else 'random')}")
    print(f"device={runtime.device.type}")
    print(f"precision={runtime.precision}")
    print(f"positional_embedding={pos_resize}")
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
