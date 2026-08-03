#!/usr/bin/env python3
"""Extract patch tokens from the custom ijepa_lite affinity-novelty model."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import torch


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_REPO = ROOT_DIR / "repos" / "ijepa_lite"
sys.path.insert(0, str(ROOT_DIR))

from experiments.shared.catalog import (  # noqa: E402
    IJEPA_LITE_AFFINITY_NOVELTY_CHECKPOINT,
    IJEPA_LITE_AFFINITY_NOVELTY_MANIFEST,
    IJEPA_LITE_AFFINITY_NOVELTY_MANIFEST_PATH,
)
from experiments.shared.device import (  # noqa: E402
    add_runtime_arguments,
    inference_context,
    resolve_runtime,
    seed_inference,
)
from experiments.shared.positional_embeddings import (  # noqa: E402
    resize_square_patch_pos_embed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=IJEPA_LITE_AFFINITY_NOVELTY_CHECKPOINT,
        help=(
            "Path to the ijepa_lite training checkpoint. Defaults to the manifest's "
            "repository-relative path and can be overridden with "
            "IJEPA_LITE_AFFINITY_NOVELTY_CHECKPOINT."
        ),
    )
    parser.add_argument(
        "--ijepa-lite-repo",
        type=Path,
        default=Path(os.environ.get("IJEPA_LITE_REPO", DEFAULT_REPO)),
        help="Path to the ijepa_lite source checkout.",
    )
    parser.add_argument(
        "--encoder",
        choices=("target", "context"),
        default="target",
        help="Select the EMA target encoder or the online context encoder.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Optional image path. If omitted, use a seeded random image.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--output-embedding",
        type=Path,
        default=None,
        help="Optional .npy path for non-pooled patch-token embeddings.",
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


def extract_encoder_state_dict(
    payload: object,
    *,
    encoder: str,
) -> dict[str, torch.Tensor]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"Expected a checkpoint mapping, got {type(payload).__name__}.")

    state: object = payload.get("model", payload)
    if not isinstance(state, Mapping):
        raise TypeError("Checkpoint 'model' entry is not a state-dict mapping.")

    nested = state.get(f"{encoder}_encoder")
    if isinstance(nested, Mapping):
        selected = {
            str(key).removeprefix("module."): value
            for key, value in nested.items()
            if isinstance(value, torch.Tensor)
        }
    else:
        prefix = f"{encoder}_encoder."
        selected = {}
        for raw_key, value in state.items():
            if not isinstance(value, torch.Tensor):
                continue
            key = str(raw_key).removeprefix("module.")
            if key.startswith(prefix):
                selected[key.removeprefix(prefix)] = value

    if not selected:
        raise KeyError(f"Checkpoint has no {encoder}_encoder tensor subtree.")
    return selected


def build_encoder(repo: Path, image_size: int) -> torch.nn.Module:
    source_dir = repo / "src"
    if not source_dir.exists():
        raise FileNotFoundError(
            f"ijepa_lite source package not found at {source_dir}. "
            "Initialize the repos/ijepa_lite submodule or pass --ijepa-lite-repo."
        )
    sys.path.insert(0, str(source_dir))

    from ijepa_lite.models.vit_tokens import build_torchvision_vit_tokens

    model_config = IJEPA_LITE_AFFINITY_NOVELTY_MANIFEST["model"]
    patch_size = int(model_config["patch_size"])
    covered_size = (image_size // patch_size) * patch_size
    if covered_size <= 0:
        raise ValueError(f"image_size must be at least {patch_size}, got {image_size}.")
    config = SimpleNamespace(
        arch=str(model_config["arch"]),
        image_size=covered_size,
        patch_size=patch_size,
        embed_dim=int(model_config["embed_dim"]),
        depth=int(model_config["depth"]),
        num_heads=int(model_config["num_heads"]),
        remove_head=bool(model_config["remove_head"]),
        pos_embed_kind=str(model_config["pos_embed_kind"]),
        use_cls_token=bool(model_config["use_cls_token"]),
    )
    with torch.device("meta"):
        model = build_torchvision_vit_tokens(config)
    # torchvision requires divisible construction geometry. Its patchifier can
    # still consume the requested square and naturally ignores border remainders.
    model.vit.image_size = image_size
    return model


def load_image(path: Path, image_size: int) -> tuple[torch.Tensor, object]:
    import numpy as np
    from PIL import Image

    image = Image.open(path).convert("RGB")
    side = min(image.size)
    left = (image.width - side) // 2
    top = (image.height - side) // 2
    image = image.crop((left, top, left + side, top + side))
    image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
    tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().div(255.0)
    preprocessing = IJEPA_LITE_AFFINITY_NOVELTY_MANIFEST["preprocessing"]
    mean = torch.tensor(preprocessing["normalization_mean"]).view(3, 1, 1)
    std = torch.tensor(preprocessing["normalization_std"]).view(3, 1, 1)
    return ((tensor - mean) / std).unsqueeze(0), image


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    repo = args.ijepa_lite_repo.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    patch_size = int(IJEPA_LITE_AFFINITY_NOVELTY_MANIFEST["model"]["patch_size"])
    if args.image_size < patch_size:
        raise ValueError(
            f"--image-size must be at least the {patch_size}-pixel patch size."
        )

    runtime = resolve_runtime(args.device, args.precision)
    seed_inference(args.seed, runtime)
    model = build_encoder(repo, args.image_size)

    payload = torch.load(
        checkpoint,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    state_dict = extract_encoder_state_dict(payload, encoder=args.encoder)
    pos_resize = resize_square_patch_pos_embed(
        state_dict,
        key="vit.encoder.pos_embedding",
        target_tokens=model.vit.encoder.pos_embedding.shape[1],
    )
    load_msg = model.load_state_dict(state_dict, strict=True, assign=True)
    del payload, state_dict
    model.requires_grad_(False)
    model = model.to(runtime.device).eval()

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
        np.save(args.output_embedding, embedding.detach().float().cpu().numpy())
    if args.output_preprocessed_image is not None and preprocessed_image is not None:
        args.output_preprocessed_image.parent.mkdir(parents=True, exist_ok=True)
        preprocessed_image.save(args.output_preprocessed_image)

    print(f"checkpoint={checkpoint}")
    print(f"model_manifest={IJEPA_LITE_AFFINITY_NOVELTY_MANIFEST_PATH}")
    print(f"ijepa_lite_repo={repo}")
    print(f"encoder={args.encoder}")
    print(f"image={args.image or 'random'}")
    print(f"device={runtime.device.type}")
    print(f"precision={runtime.precision}")
    print(
        "covered_image_size="
        f"{(args.image_size // patch_size) * patch_size}"
    )
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
