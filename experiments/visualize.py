#!/usr/bin/env python3
"""Render PCA, PC1, KMeans, and optional temporal views of token embeddings."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from experiments.artifacts import SCHEMA_NAME, read_embedding_metadata


def _pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _parse_grid_shape(raw: str | None) -> tuple[int, ...] | None:
    if raw is None:
        return None
    shape = tuple(int(value) for value in raw.replace(",", "x").split("x") if value)
    if not shape or any(value <= 0 for value in shape):
        raise ValueError(f"Invalid grid shape: {raw!r}")
    return shape


def _load_embedding(path: Path, key: str | None = None) -> np.ndarray:
    import h5py

    selected = key or "tokens"
    with h5py.File(path, "r") as handle:
        if selected not in handle:
            raise ValueError(f"{path} has no {selected!r} dataset.")
        return handle[selected][...]


def _prepare_tokens(
    embedding: np.ndarray,
    *,
    sample_index: int | None,
    grid_shape: tuple[int, ...] | None,
) -> tuple[np.ndarray, tuple[int, ...]]:
    array = np.asarray(embedding)
    if sample_index is not None:
        array = array[sample_index]
    elif array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim < 2:
        raise ValueError(f"Expected token embeddings, got {array.shape}.")

    if grid_shape is not None:
        count = int(np.prod(grid_shape))
        if int(np.prod(array.shape[:-1])) != count:
            raise ValueError(f"Grid {grid_shape} does not match embedding {array.shape}.")
        return array.reshape(count, array.shape[-1]).astype(np.float32), grid_shape
    if array.ndim > 2:
        grid_shape = tuple(int(value) for value in array.shape[:-1])
        return array.reshape(-1, array.shape[-1]).astype(np.float32), grid_shape

    side = math.isqrt(array.shape[0])
    if side * side != array.shape[0]:
        raise ValueError("Cannot infer a square token grid; pass --grid-shape.")
    return array.astype(np.float32), (side, side)


def _select_2d_tokens(
    tokens: np.ndarray,
    grid_shape: tuple[int, ...],
    slice_index: int | None,
) -> tuple[np.ndarray, tuple[int, int], str]:
    if len(grid_shape) == 2:
        return tokens, (grid_shape[0], grid_shape[1]), "all"
    leading = int(np.prod(grid_shape[:-2]))
    selected = leading // 2 if slice_index is None else slice_index
    if selected < 0 or selected >= leading:
        raise ValueError(f"Slice must be in 0..{leading - 1}.")
    height, width = grid_shape[-2:]
    slices = tokens.reshape(leading, height * width, tokens.shape[-1])
    return slices[selected], (height, width), str(selected)


def _standardize(tokens: np.ndarray) -> np.ndarray:
    return (tokens - tokens.mean(0, keepdims=True)) / np.maximum(
        tokens.std(0, keepdims=True), 1e-6
    )


def _projection(tokens: np.ndarray) -> tuple[np.ndarray, list[float]]:
    from sklearn.decomposition import PCA

    components = min(3, tokens.shape[0], tokens.shape[1])
    pca = PCA(n_components=components, random_state=0)
    projected = pca.fit_transform(_standardize(tokens.astype(np.float32)))
    projected = np.pad(projected, ((0, 0), (0, 3 - components)))
    return projected, [float(value) for value in pca.explained_variance_ratio_]


def _normalize_projection(projected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rgb = projected[:, :3]
    low = np.percentile(rgb, 1, axis=0, keepdims=True)
    high = np.percentile(rgb, 99, axis=0, keepdims=True)
    rgb = np.clip((rgb - low) / np.maximum(high - low, 1e-6), 0, 1)
    pc1 = projected[:, 0]
    pc1 = (pc1 - pc1.min()) / max(float(pc1.max() - pc1.min()), 1e-6)
    return rgb, pc1


def _token_pca_maps(
    tokens: np.ndarray, grid: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    projected, variance = _projection(tokens)
    rgb, pc1 = _normalize_projection(projected)
    return (
        rgb.reshape(*grid, 3),
        pc1.reshape(*grid),
        {"pca_explained_variance": variance},
    )


def _cluster_space(tokens: np.ndarray, pca_dim: int = 16) -> np.ndarray:
    from sklearn.decomposition import PCA

    standardized = _standardize(tokens.astype(np.float32))
    components = min(pca_dim, tokens.shape[0] - 1, tokens.shape[1])
    return PCA(n_components=components, random_state=0).fit_transform(standardized)


def _kmeans_map(tokens: np.ndarray, grid: tuple[int, int], clusters: int) -> np.ndarray:
    from sklearn.cluster import KMeans

    labels = KMeans(n_clusters=clusters, n_init=10, random_state=0).fit_predict(
        _cluster_space(tokens)
    )
    return labels.reshape(*grid)


def _load_image(path: Path | None) -> np.ndarray | None:
    return None if path is None else np.asarray(Image.open(path).convert("RGB"))


def _render_static(
    output: Path,
    *,
    reference: np.ndarray | None,
    pca_rgb: np.ndarray,
    pc1: np.ndarray,
    labels: np.ndarray,
    title: str,
) -> None:
    plt = _pyplot()
    figure, axes = plt.subplots(1, 4, figsize=(12, 3), layout="constrained")
    if reference is not None:
        axes[0].imshow(reference)
    axes[0].set_title("input")
    axes[1].imshow(pca_rgb, interpolation="nearest")
    axes[1].set_title("PCA RGB")
    axes[2].imshow(pc1, interpolation="nearest", cmap="magma")
    axes[2].set_title("PC1")
    axes[3].imshow(labels, interpolation="nearest", cmap="tab10", vmin=0, vmax=9)
    axes[3].set_title("KMeans (k=4)")
    for axis in axes:
        axis.set_axis_off()
    figure.suptitle(title)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _video_frames(
    path: Path,
    *,
    temporal_slices: int,
    tubelet_size: int,
    size: int,
) -> list[np.ndarray]:
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
    count = temporal_slices * tubelet_size
    sample_fps = count / max(float(probe.stdout.strip() or "1"), 1.0)
    with tempfile.TemporaryDirectory(prefix="jepas_frames_") as directory:
        pattern = Path(directory) / "frame_%04d.jpg"
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(path),
                "-vf",
                f"fps={sample_fps:.8f},scale={size}:{size}:"
                f"force_original_aspect_ratio=increase,crop={size}:{size}",
                "-frames:v", str(count), str(pattern),
            ],
            check=True,
        )
        paths = sorted(Path(directory).glob("frame_*.jpg"))
        if not paths:
            raise RuntimeError(f"No frames extracted from {path}.")
        while len(paths) < count:
            paths.append(paths[-1])
        return [
            np.asarray(Image.open(paths[index * tubelet_size]).convert("RGB"))
            for index in range(temporal_slices)
        ]


def _render_video(
    output: Path,
    *,
    tokens: np.ndarray,
    grid: tuple[int, ...],
    reference_image: Path | None,
    reference_video: Path | None,
    tubelet_size: int,
    size: int,
    fps: float,
    title: str,
) -> None:
    if len(grid) < 3:
        raise ValueError("Temporal visualization requires a temporal token grid.")
    temporal = int(np.prod(grid[:-2]))
    height, width = grid[-2:]
    projected, _ = _projection(tokens)
    rgb, pc1 = _normalize_projection(projected)
    labels = _kmeans_map(tokens, (temporal, height * width), 4).reshape(
        temporal, height, width
    )
    rgb = rgb.reshape(temporal, height, width, 3)
    pc1 = pc1.reshape(temporal, height, width)
    if reference_video is not None:
        references = _video_frames(
            reference_video,
            temporal_slices=temporal,
            tubelet_size=tubelet_size,
            size=size,
        )
    else:
        references = [_load_image(reference_image)] * temporal

    plt = _pyplot()
    with tempfile.TemporaryDirectory(prefix="jepas_viz_") as directory:
        for index in range(temporal):
            frame = Path(directory) / f"frame_{index:04d}.png"
            _render_static(
                frame,
                reference=references[index],
                pca_rgb=rgb[index],
                pc1=pc1[index],
                labels=labels[index],
                title=f"{title} | slice {index}",
            )
        plt.close("all")
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-framerate", str(fps), "-i", str(Path(directory) / "frame_%04d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(output),
            ],
            check=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("embedding", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--grid-shape")
    parser.add_argument("--sample-index", type=int)
    parser.add_argument("--slice-index", type=int)
    parser.add_argument("--key")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--reference-video", type=Path)
    parser.add_argument("--animate", action="store_true")
    parser.add_argument("--tubelet-size", type=int, default=2)
    parser.add_argument("--reference-size", type=int, default=384)
    parser.add_argument("--animation-fps", type=float, default=2)
    parser.add_argument("--title")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embedding = args.embedding.resolve()
    output_dir = args.out_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    raw = _load_embedding(embedding, args.key)
    tokens, grid = _prepare_tokens(
        raw,
        sample_index=args.sample_index,
        grid_shape=_parse_grid_shape(args.grid_shape),
    )
    tokens = F.layer_norm(torch.from_numpy(tokens), (tokens.shape[-1],)).numpy()
    selected, grid_2d, selected_slice = _select_2d_tokens(tokens, grid, args.slice_index)
    pca_rgb, pc1, pca = _token_pca_maps(selected, grid_2d)
    labels = _kmeans_map(selected, grid_2d, 4)
    title = args.title or f"{embedding.name} | grid={'x'.join(map(str, grid))}"
    figure = output_dir / "visualization.png"
    _render_static(
        figure,
        reference=_load_image(args.image),
        pca_rgb=pca_rgb,
        pc1=pc1,
        labels=labels,
        title=title,
    )
    video = None
    if args.animate:
        video = output_dir / "visualization.mp4"
        _render_video(
            video,
            tokens=tokens,
            grid=grid,
            reference_image=args.image,
            reference_video=args.reference_video,
            tubelet_size=args.tubelet_size,
            size=args.reference_size,
            fps=args.animation_fps,
            title=title,
        )
    metadata = {
        "schema": SCHEMA_NAME,
        "embedding": embedding.name,
        "extraction": read_embedding_metadata(embedding),
        "grid_shape": list(grid),
        "slice_index": selected_slice,
        "pca_explained_variance": pca["pca_explained_variance"],
        "figure": figure.name,
        "video": video.name if video else None,
        "video_expected": bool(args.animate),
        "visualization": {
            "sample_index": args.sample_index,
            "tubelet_size": args.tubelet_size,
            "reference_size": args.reference_size,
            "animation_fps": args.animation_fps,
            "title": title,
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"visualization={figure}")


if __name__ == "__main__":
    main()
