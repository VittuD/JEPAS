"""Positional-embedding geometry helpers used by source-model experiments."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def resize_square_patch_pos_embed(
    state_dict: dict[str, torch.Tensor],
    *,
    key: str,
    target_tokens: int,
) -> str:
    """Bicubically resize a patch-only square positional embedding in place."""
    if key not in state_dict:
        raise KeyError(f"Checkpoint has no {key!r} positional embedding.")

    source = state_dict[key]
    if source.ndim != 3 or source.shape[0] != 1:
        raise ValueError(f"{key} must have shape [1, tokens, dim], got {tuple(source.shape)}.")

    source_tokens = int(source.shape[1])
    source_side = math.isqrt(source_tokens)
    target_side = math.isqrt(int(target_tokens))
    if source_side * source_side != source_tokens:
        raise ValueError(f"{key} has {source_tokens} tokens, which is not a square patch grid.")
    if target_side * target_side != target_tokens:
        raise ValueError(f"Target has {target_tokens} tokens, which is not a square patch grid.")
    if source_tokens == target_tokens:
        return f"native ({source_side}x{source_side})"

    dtype = source.dtype
    resized = F.interpolate(
        source.float().reshape(1, source_side, source_side, source.shape[-1]).permute(0, 3, 1, 2),
        size=(target_side, target_side),
        mode="bicubic",
        align_corners=False,
    )
    state_dict[key] = (
        resized.permute(0, 2, 3, 1)
        .reshape(1, target_tokens, source.shape[-1])
        .to(dtype=dtype)
    )
    return f"bicubic ({source_side}x{source_side} -> {target_side}x{target_side})"
