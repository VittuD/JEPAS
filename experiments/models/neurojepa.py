#!/usr/bin/env python3
"""Extract Neuro-JEPA tokens from one 3D volume."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from safetensors.torch import load_file


ROOT_DIR = Path(__file__).resolve().parents[2]
NEUROJEPA_DIR = ROOT_DIR / "repos" / "Neuro-JEPA"
DEFAULT_MODEL_DIR = ROOT_DIR / "weights" / "neurojepa" / "Neuro-JEPA"
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
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Local Hugging Face Neuro-JEPA model directory.",
    )
    parser.add_argument(
        "--volume",
        type=Path,
        default=None,
        help="Optional .npy, .nii, or .nii.gz volume. If omitted, use a seeded random volume.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-embedding", type=Path, default=None, help="Optional .npy token output.")
    parser.add_argument(
        "--output-preprocessed-slice",
        type=Path,
        default=None,
        help="Optional image path for the middle slice of the preprocessed input volume.",
    )
    add_runtime_arguments(parser)
    return parser.parse_args()


def to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{key: to_namespace(val) for key, val in value.items()})
    if isinstance(value, list):
        return [to_namespace(val) for val in value]
    return value


def clean_backbone_key(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefixes = (
        "student.vision_encoder.",
        "student.encoder.",
        "vision_encoder.",
        "target_encoder.",
        "encoder.",
        "model.",
        "module.",
        "_orig_mod.",
        "backbone.",
    )
    cleaned = {}
    for key, value in state_dict.items():
        clean_key = key.replace("_checkpoint_wrapped_module.", "")
        stripped = True
        while stripped:
            stripped = False
            for prefix in prefixes:
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix):]
                    stripped = True
        cleaned[clean_key] = value
    return cleaned


def load_volume(path: Path, target_shape: list[int], in_chans: int) -> tuple[torch.Tensor, torch.Tensor]:
    import numpy as np

    if path.suffix == ".npy":
        array = np.load(path)
    elif path.suffix == ".nii" or path.name.endswith(".nii.gz"):
        import nibabel as nib

        array = nib.load(str(path)).get_fdata(dtype=np.float32)
    else:
        raise ValueError(f"Unsupported volume format: {path}")

    volume = torch.from_numpy(np.asarray(array, dtype=np.float32))
    if volume.ndim == 4:
        volume = volume[..., 0]
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {tuple(volume.shape)}")
    volume = volume.unsqueeze(0).unsqueeze(0)
    volume = torch.nn.functional.interpolate(
        volume,
        size=tuple(int(v) for v in target_shape),
        mode="trilinear",
        align_corners=False,
    )
    volume = volume.squeeze(0)
    vmin = volume.amin(dim=(1, 2, 3), keepdim=True)
    vmax = volume.amax(dim=(1, 2, 3), keepdim=True)
    volume = (volume - vmin) / torch.clamp(vmax - vmin, min=1e-6)
    volume = (volume - 0.5) / 0.5
    if in_chans != 1:
        volume = volume.repeat(in_chans, 1, 1, 1)
    return volume.unsqueeze(0), volume.squeeze(0)


def save_middle_slice(path: Path, volume: torch.Tensor) -> None:
    import numpy as np
    from PIL import Image

    if volume.ndim == 4:
        volume = volume[0]
    depth = volume.shape[0]
    image = volume[depth // 2].detach().cpu().float()
    image = (image - image.min()) / torch.clamp(image.max() - image.min(), min=1e-6)
    array = (image.numpy() * 255.0).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="L").save(path)


def main() -> None:
    args = parse_args()
    config_path = args.model_dir / "config.json"
    checkpoint_path = args.model_dir / "model.safetensors"

    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    sys.path.insert(0, str(NEUROJEPA_DIR / "src"))
    from neurojepa.models.vision_transformer import vit_base

    config = json.loads(config_path.read_text())
    model_cfg = config["model"]
    meta_cfg = config["meta"]

    runtime = resolve_runtime(args.device, args.precision)
    seed_inference(args.seed, runtime)

    model = vit_base(
        img_size=model_cfg["img_size"],
        patch_size=model_cfg["patch_size"],
        in_chans=model_cfg["in_chans"],
        uniform_power=model_cfg["uniform_power"],
        use_sdpa=meta_cfg["use_sdpa"],
        use_rope=model_cfg["use_rope"],
        use_silu=model_cfg["use_silu"],
        wide_silu=model_cfg["wide_silu"],
        use_activation_checkpointing=False,
        use_moe=model_cfg["use_moe"],
        moe_params=to_namespace(model_cfg["moe_params"]),
    )
    model.eval()

    state_dict = clean_backbone_key(load_file(str(checkpoint_path), device="cpu"))
    load_msg = model.load_state_dict(state_dict, strict=False)
    del state_dict
    model = model.to(runtime.device).eval()

    volume_shape = model_cfg["img_size"]
    if args.volume is None:
        volume = torch.randn(1, model_cfg["in_chans"], *volume_shape)
        preprocessed_volume = volume[0, 0]
    else:
        volume, preprocessed_volume = load_volume(
            args.volume,
            target_shape=volume_shape,
            in_chans=model_cfg["in_chans"],
        )
    volume = volume.to(runtime.device, non_blocking=runtime.device.type == "cuda")
    with inference_context(runtime):
        tokens, moe_scores = model(volume)
    pooled = tokens.mean(dim=1)

    if args.output_embedding is not None:
        import numpy as np

        args.output_embedding.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output_embedding, tokens.detach().float().cpu().numpy())
    if args.output_preprocessed_slice is not None:
        save_middle_slice(args.output_preprocessed_slice, preprocessed_volume)

    print(f"model_dir={args.model_dir}")
    print(f"checkpoint={checkpoint_path}")
    print(f"volume={args.volume or 'random'}")
    print(f"device={runtime.device.type}")
    print(f"precision={runtime.precision}")
    print(
        f"missing_keys={len(load_msg.missing_keys)} "
        f"unexpected_keys={len(load_msg.unexpected_keys)}"
    )
    print(f"tokens_shape={tuple(tokens.shape)}")
    print(f"pooled_shape={tuple(pooled.shape)}")
    print(f"num_moe_score_entries={len(moe_scores)}")
    pooled_stats = pooled.detach().float()
    print(
        f"pooled_mean={pooled_stats.mean().item():.6f} "
        f"pooled_std={pooled_stats.std().item():.6f}"
    )
    if args.output_embedding is not None:
        print(f"output_embedding={args.output_embedding}")
    if args.output_preprocessed_slice is not None:
        print(f"output_preprocessed_slice={args.output_preprocessed_slice}")


if __name__ == "__main__":
    main()
