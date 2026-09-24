"""Frozen JEPA encoders wrapped as raw (B, D, H, W) single-scale grid backbones.

Each wrapper loads the same checkpoint/model-construction path as the
matching experiments/models/<name>.py adapter (kept independent rather than
importing those CLI scripts, since they're structured as argparse mains, not
reusable modules) but as a persistent nn.Module suitable for a training loop,
with all parameters frozen (requires_grad=False) and the forward pass run
under torch.no_grad() -- only the downstream neck/head are ever trained.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))


class GridBackbone(nn.Module):
    """A frozen encoder whose forward pass returns a (B, D, H, W) token grid."""

    def __init__(self, encoder: nn.Module, *, embed_dim: int, grid_size: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()

    def train(self, mode: bool = True) -> "GridBackbone":
        # Keep the frozen encoder in eval() regardless of the outer module's
        # train()/eval() calls (e.g. LayerNorm has no running stats to worry
        # about, but this keeps intent explicit and future-proof).
        super().train(mode)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder(x)  # (B, T, D)
        b, t, d = tokens.shape
        side = self.grid_size
        if side * side != t:
            raise ValueError(f"Expected a square {side}x{side} grid ({side * side} tokens), got {t}.")
        return tokens.permute(0, 2, 1).reshape(b, d, side, side).contiguous()


def _clean_backbone_key(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key.replace("module.", ""): value for key, value in state_dict.items()}


def build_ijepa_backbone(
    *, checkpoint: Path, image_size: int = 224, device: torch.device
) -> GridBackbone:
    """I-JEPA ViT-H/14, native 224px -> 16x16 grid, D=1280 (same as experiments/models/ijepa.py)."""
    ijepa_dir = ROOT_DIR / "repos" / "ijepa"
    sys.path.insert(0, str(ijepa_dir))
    from src.models.vision_transformer import vit_huge  # noqa: E402

    from experiments.positional_embeddings import resize_square_patch_pos_embed  # noqa: E402

    model = vit_huge(img_size=[image_size], patch_size=14)
    ckpt = torch.load(checkpoint, map_location="cpu")
    state_dict = ckpt.get("target_encoder") or ckpt.get("encoder")
    if state_dict is None:
        raise KeyError(f"Checkpoint has no target_encoder/encoder key. Keys: {sorted(ckpt)}")
    state_dict = _clean_backbone_key(state_dict)
    resize_square_patch_pos_embed(state_dict, key="pos_embed", target_tokens=model.pos_embed.shape[1])
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()

    grid_size = image_size // 14
    return GridBackbone(model, embed_dim=1280, grid_size=grid_size).to(device)


BACKBONE_BUILDERS = {
    "ijepa": build_ijepa_backbone,
}


def build_backbone(model_name: str, *, checkpoint: Path, image_size: int, device: torch.device) -> GridBackbone:
    if model_name not in BACKBONE_BUILDERS:
        raise ValueError(f"No dense_eval backbone builder for {model_name!r} yet. Have: {sorted(BACKBONE_BUILDERS)}")
    return BACKBONE_BUILDERS[model_name](checkpoint=checkpoint, image_size=image_size, device=device)
