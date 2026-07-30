#!/usr/bin/env python3
"""Smoke test Neuro-JEPA source inference on one random 3D volume."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from safetensors.torch import load_file


ROOT_DIR = Path(__file__).resolve().parents[1]
NEUROJEPA_DIR = ROOT_DIR / "repos" / "Neuro-JEPA"
DEFAULT_MODEL_DIR = ROOT_DIR / "weights_py" / "neurojepa" / "Neuro-JEPA"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Local Hugging Face Neuro-JEPA model directory.",
    )
    parser.add_argument("--seed", type=int, default=0)
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

    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)

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

    volume_shape = model_cfg["img_size"]
    volume = torch.randn(1, model_cfg["in_chans"], *volume_shape)
    tokens, moe_scores = model(volume)
    pooled = tokens.mean(dim=1)

    print(f"model_dir={args.model_dir}")
    print(f"checkpoint={checkpoint_path}")
    print(
        f"missing_keys={len(load_msg.missing_keys)} "
        f"unexpected_keys={len(load_msg.unexpected_keys)}"
    )
    print(f"tokens_shape={tuple(tokens.shape)}")
    print(f"pooled_shape={tuple(pooled.shape)}")
    print(f"num_moe_score_entries={len(moe_scores)}")
    print(f"pooled_mean={pooled.mean().item():.6f} pooled_std={pooled.std().item():.6f}")


if __name__ == "__main__":
    main()
