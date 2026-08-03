#!/usr/bin/env python3
"""Visualize token embeddings with PCA maps and clustering summaries.

This script is intentionally model-agnostic. It consumes a saved embedding
array/tensor and expects the last dimension to be the feature dimension. The
remaining dimensions are token/grid dimensions, or can be supplied with
``--grid-shape`` for flattened ``[num_tokens, dim]`` embeddings.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F


ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from experiments.shared.provenance import portable_path  # noqa: E402


@dataclass(frozen=True)
class ClusterResult:
    method: str
    params: str
    labels: np.ndarray
    stats: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("embedding", help="Path to a .npy, .npz, .pt, or .pth embedding file.")
    parser.add_argument("--out-dir", required=True, help="Directory for PNG/CSV/JSON outputs.")
    parser.add_argument(
        "--key",
        default=None,
        help="Optional key for dict checkpoints or .npz files. Defaults to the only array/tensor key.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help="Optional batch index to select before grid interpretation.",
    )
    parser.add_argument(
        "--grid-shape",
        default=None,
        help="Token grid shape, e.g. 16x16, 8x32x32, or 6,6,6. Required for non-square [N,D].",
    )
    parser.add_argument(
        "--slice-index",
        type=int,
        default=None,
        help="For grids with more than two dimensions, select this flattened leading slice.",
    )
    parser.add_argument("--image", default=None, help="Optional reference image shown in the first panel.")
    parser.add_argument(
        "--reference-video",
        default=None,
        help="Optional source video sampled for a temporal embedding visualization.",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Render embedding_visualization.mp4 across all leading token-grid slices.",
    )
    parser.add_argument(
        "--reference-size",
        type=int,
        default=384,
        help="Square size used when sampling --reference-video.",
    )
    parser.add_argument(
        "--tubelet-size",
        type=int,
        default=2,
        help="Input frames represented by one temporal token slice.",
    )
    parser.add_argument(
        "--animation-fps",
        type=float,
        default=2.0,
        help="Output frames per second for the temporal visualization.",
    )
    parser.add_argument(
        "--animation-clusters",
        type=int,
        default=4,
        help="Globally fitted KMeans clusters shown in the temporal visualization.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=("kmeans", "gmm"),
        choices=("kmeans", "gmm", "agglomerative", "spectral", "dbscan"),
        help="Clustering methods to run.",
    )
    parser.add_argument(
        "--cluster-k",
        nargs="+",
        type=int,
        default=(3, 4, 6),
        help="Cluster counts for k-based methods.",
    )
    parser.add_argument(
        "--dbscan-eps",
        nargs="+",
        type=float,
        default=(1.5, 2.5, 4.0),
        help="DBSCAN eps values in the standardized PCA cluster space.",
    )
    parser.add_argument(
        "--cluster-pca-dim",
        type=int,
        default=16,
        help="PCA dimensions used for clustering after per-embedding standardization.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Clustering seed.")
    parser.add_argument("--title", default=None, help="Optional figure title.")
    return parser.parse_args()


def _pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _parse_grid_shape(raw: str | None) -> tuple[int, ...] | None:
    if raw is None:
        return None
    parts = raw.replace(",", "x").split("x")
    shape = tuple(int(part) for part in parts if part)
    if not shape or any(dim <= 0 for dim in shape):
        raise ValueError(f"Invalid --grid-shape {raw!r}.")
    return shape


def _load_embedding(path: Path, key: str | None) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)

    if path.suffix == ".npz":
        data = np.load(path)
        if key is None:
            keys = list(data.keys())
            if len(keys) != 1:
                raise ValueError(f"{path} contains keys {keys}; pass --key.")
            key = keys[0]
        return data[key]

    if path.suffix in {".pt", ".pth", ".tar"} or path.name.endswith(".pth.tar"):
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy()
        if not isinstance(obj, dict):
            raise TypeError(f"Unsupported torch object type: {type(obj)!r}")
        if key is None:
            tensor_keys = [k for k, v in obj.items() if isinstance(v, (torch.Tensor, np.ndarray))]
            if len(tensor_keys) != 1:
                raise ValueError(f"{path} contains tensor keys {tensor_keys}; pass --key.")
            key = str(tensor_keys[0])
        value = obj[key]
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            return value
        raise TypeError(f"Key {key!r} has unsupported type {type(value)!r}.")

    raise ValueError(f"Unsupported embedding file extension: {path}")


def _prepare_tokens(
    embedding: np.ndarray,
    *,
    sample_index: int | None,
    grid_shape: tuple[int, ...] | None,
) -> tuple[np.ndarray, tuple[int, ...]]:
    arr = np.asarray(embedding)
    if sample_index is not None:
        arr = arr[int(sample_index)]
    elif arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim < 2:
        raise ValueError(f"Expected token embeddings with at least 2 dims, got shape {arr.shape}.")

    if grid_shape is not None:
        n_tokens = int(np.prod(grid_shape))
        if arr.ndim == 2:
            if arr.shape[0] != n_tokens:
                raise ValueError(
                    f"--grid-shape product {n_tokens} does not match embedding tokens {arr.shape[0]}."
                )
            tokens = arr
        else:
            if tuple(arr.shape[:-1]) != grid_shape:
                raise ValueError(
                    f"--grid-shape {grid_shape} does not match embedding token dims {arr.shape[:-1]}."
                )
            tokens = arr.reshape(n_tokens, arr.shape[-1])
        return tokens.astype(np.float32), grid_shape

    if arr.ndim > 2:
        inferred_grid = tuple(int(v) for v in arr.shape[:-1])
        return arr.reshape(-1, arr.shape[-1]).astype(np.float32), inferred_grid

    n_tokens = int(arr.shape[0])
    side = int(math.isqrt(n_tokens))
    if side * side != n_tokens:
        raise ValueError(
            f"Cannot infer a square grid from {n_tokens} tokens; pass --grid-shape."
        )
    return arr.astype(np.float32), (side, side)


def _select_2d_tokens(
    tokens: np.ndarray,
    grid_shape: tuple[int, ...],
    slice_index: int | None,
) -> tuple[np.ndarray, tuple[int, int], str]:
    if len(grid_shape) == 2:
        return tokens, (grid_shape[0], grid_shape[1]), "all"

    leading = int(np.prod(grid_shape[:-2]))
    h, w = int(grid_shape[-2]), int(grid_shape[-1])
    if slice_index is None:
        slice_index = leading // 2
    if slice_index < 0 or slice_index >= leading:
        raise ValueError(f"--slice-index must be in 0..{leading - 1}, got {slice_index}.")
    reshaped = tokens.reshape(*grid_shape, tokens.shape[-1]).reshape(leading, h * w, tokens.shape[-1])
    return reshaped[int(slice_index)], (h, w), str(int(slice_index))


def _standardize(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mu = x.mean(axis=0, keepdims=True)
    sigma = x.std(axis=0, keepdims=True)
    return (x - mu) / np.maximum(sigma, eps)


def _pca_projection(x: np.ndarray, n_components: int) -> tuple[np.ndarray, Any]:
    from sklearn.decomposition import PCA

    n = max(1, min(int(n_components), x.shape[0], x.shape[1]))
    pca = PCA(n_components=n, random_state=0)
    return pca.fit_transform(x), pca


def _token_pca_maps(tokens: np.ndarray, grid: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    x = _standardize(tokens.astype(np.float32))
    proj, pca = _pca_projection(x, 3)
    if proj.shape[1] < 3:
        proj = np.pad(proj, ((0, 0), (0, 3 - proj.shape[1])), mode="constant")

    rgb = proj[:, :3]
    lo = np.percentile(rgb, 1, axis=0, keepdims=True)
    hi = np.percentile(rgb, 99, axis=0, keepdims=True)
    rgb = np.clip((rgb - lo) / np.maximum(hi - lo, 1e-6), 0.0, 1.0)

    pc1 = proj[:, 0]
    pc1 = (pc1 - pc1.min()) / max(float(pc1.max() - pc1.min()), 1e-6)

    h, w = grid
    stats = {
        "pca_explained_variance": [
            float(v) for v in getattr(pca, "explained_variance_ratio_", np.asarray([]))
        ],
    }
    return rgb.reshape(h, w, 3), pc1.reshape(h, w), stats


def _token_norm_map(tokens: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    norms = np.linalg.norm(tokens.astype(np.float32), axis=1)
    norms = (norms - norms.min()) / max(float(norms.max() - norms.min()), 1e-6)
    return norms.reshape(*grid)


def _cluster_space(tokens: np.ndarray, pca_dim: int) -> np.ndarray:
    x = _standardize(tokens.astype(np.float32))
    if pca_dim <= 0:
        return x
    proj, _ = _pca_projection(x, min(pca_dim, x.shape[0] - 1, x.shape[1]))
    return _standardize(proj.astype(np.float32))


def _connected_components(labels_2d: np.ndarray, cluster_label: int) -> int:
    mask = labels_2d == cluster_label
    if not mask.any():
        return 0

    seen = np.zeros_like(mask, dtype=bool)
    components = 0
    h, w = mask.shape
    for y in range(h):
        for x in range(w):
            if seen[y, x] or not mask[y, x]:
                continue
            components += 1
            stack = [(y, x)]
            seen[y, x] = True
            while stack:
                cy, cx = stack.pop()
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and mask[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
    return components


def _boundary_fraction(labels_2d: np.ndarray) -> float:
    edges = 0
    total = 0
    for axis in (0, 1):
        a = labels_2d.take(indices=range(labels_2d.shape[axis] - 1), axis=axis)
        b = labels_2d.take(indices=range(1, labels_2d.shape[axis]), axis=axis)
        valid = (a >= 0) & (b >= 0)
        total += int(valid.sum())
        edges += int(((a != b) & valid).sum())
    return float(edges / max(total, 1))


def _adjacent_cosine_stats(tokens: np.ndarray, grid: tuple[int, int]) -> dict[str, float]:
    x = tokens.astype(np.float32)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    x = x.reshape(*grid, -1)

    distances: list[np.ndarray] = []
    for a, b in (
        (x[:-1, :, :], x[1:, :, :]),
        (x[:, :-1, :], x[:, 1:, :]),
    ):
        cosine = np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)
        distances.append((1.0 - cosine).reshape(-1))

    d = np.concatenate(distances)
    return {
        "adjacent_cosine_distance": float(d.mean()),
        "adjacent_cosine_distance_sd": float(d.std(ddof=1)) if d.size > 1 else 0.0,
        "adjacent_cosine_similarity": float(1.0 - d.mean()),
    }


def _cluster_stats(labels: np.ndarray, x: np.ndarray, grid: tuple[int, int]) -> dict[str, Any]:
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

    labels = labels.astype(int)
    valid = labels >= 0
    unique = sorted(int(v) for v in np.unique(labels[valid]))
    counts = {str(k): int((labels == k).sum()) for k in unique}
    noise_count = int((labels < 0).sum())
    proportions = {str(k): float(v / max(int(valid.sum()), 1)) for k, v in counts.items()}
    p = np.asarray(list(proportions.values()), dtype=np.float64)
    entropy = float(-(p * np.log2(np.maximum(p, 1e-12))).sum()) if p.size else 0.0

    scores: dict[str, float | None] = {
        "silhouette": None,
        "calinski_harabasz": None,
        "davies_bouldin": None,
    }
    if len(unique) > 1 and int(valid.sum()) > len(unique):
        xv = x[valid]
        yv = labels[valid]
        scores = {
            "silhouette": float(silhouette_score(xv, yv)),
            "calinski_harabasz": float(calinski_harabasz_score(xv, yv)),
            "davies_bouldin": float(davies_bouldin_score(xv, yv)),
        }

    labels_2d = labels.reshape(*grid)
    components = {str(k): int(_connected_components(labels_2d, k)) for k in unique}
    return {
        "n_clusters": int(len(unique)),
        "noise_count": noise_count,
        "counts": counts,
        "proportions": proportions,
        "cluster_entropy": entropy,
        "boundary_fraction": _boundary_fraction(labels_2d),
        "connected_components": components,
        **scores,
    }


def _run_clustering(
    x: np.ndarray,
    *,
    grid: tuple[int, int],
    methods: Iterable[str],
    cluster_k: Iterable[int],
    dbscan_eps: Iterable[float],
    seed: int,
) -> list[ClusterResult]:
    results: list[ClusterResult] = []

    for method in methods:
        if method == "kmeans":
            from sklearn.cluster import KMeans

            for k in cluster_k:
                if k <= 1 or k >= x.shape[0]:
                    continue
                model = KMeans(n_clusters=int(k), n_init=10, random_state=seed)
                labels = model.fit_predict(x)
                params = f"k={int(k)}"
                results.append(ClusterResult(method, params, labels, _cluster_stats(labels, x, grid)))

        elif method == "gmm":
            from sklearn.mixture import GaussianMixture

            for k in cluster_k:
                if k <= 1 or k >= x.shape[0]:
                    continue
                model = GaussianMixture(
                    n_components=int(k),
                    covariance_type="diag",
                    random_state=seed,
                )
                labels = model.fit_predict(x)
                params = f"k={int(k)},cov=diag"
                results.append(ClusterResult(method, params, labels, _cluster_stats(labels, x, grid)))

        elif method == "agglomerative":
            from sklearn.cluster import AgglomerativeClustering

            for k in cluster_k:
                if k <= 1 or k >= x.shape[0]:
                    continue
                model = AgglomerativeClustering(n_clusters=int(k), linkage="ward")
                labels = model.fit_predict(x)
                params = f"k={int(k)},linkage=ward"
                results.append(ClusterResult(method, params, labels, _cluster_stats(labels, x, grid)))

        elif method == "spectral":
            from sklearn.cluster import SpectralClustering

            for k in cluster_k:
                if k <= 1 or k >= x.shape[0]:
                    continue
                neighbors = max(2, min(10, x.shape[0] - 1))
                model = SpectralClustering(
                    n_clusters=int(k),
                    affinity="nearest_neighbors",
                    n_neighbors=neighbors,
                    assign_labels="kmeans",
                    random_state=seed,
                )
                labels = model.fit_predict(x)
                params = f"k={int(k)},nn={neighbors}"
                results.append(ClusterResult(method, params, labels, _cluster_stats(labels, x, grid)))

        elif method == "dbscan":
            from sklearn.cluster import DBSCAN

            min_samples = max(3, int(round(math.sqrt(x.shape[0]) / 2)))
            for eps in dbscan_eps:
                model = DBSCAN(eps=float(eps), min_samples=min_samples)
                labels = model.fit_predict(x)
                params = f"eps={float(eps):g},min_samples={min_samples}"
                results.append(ClusterResult(method, params, labels, _cluster_stats(labels, x, grid)))

    return results


def _render_label_map(ax: Any, labels: np.ndarray, grid: tuple[int, int], title: str) -> None:
    ax.imshow(labels.reshape(*grid).astype(float), interpolation="nearest", cmap="tab20")
    ax.set_title(title, fontsize=8)
    ax.set_axis_off()


def _load_reference_image(path: str | None) -> np.ndarray | None:
    if path is None:
        return None
    plt = _pyplot()
    return plt.imread(path)


def _load_video_reference_frames(
    path: Path,
    *,
    temporal_slices: int,
    tubelet_size: int,
    reference_size: int,
) -> list[np.ndarray]:
    from PIL import Image

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
    input_frames = temporal_slices * tubelet_size
    sample_fps = input_frames / duration
    with tempfile.TemporaryDirectory(prefix="jepas_visualization_frames_") as tmp:
        pattern = Path(tmp) / "input_%04d.jpg"
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(path),
                "-map",
                "0:v:0",
                "-vf",
                (
                    f"fps={sample_fps:.8f},"
                    f"scale={reference_size}:{reference_size}:"
                    "force_original_aspect_ratio=increase,"
                    f"crop={reference_size}:{reference_size}"
                ),
                "-frames:v",
                str(input_frames),
                str(pattern),
            ],
            check=True,
        )
        frame_paths = sorted(Path(tmp).glob("input_*.jpg"))
        if not frame_paths:
            raise RuntimeError(f"ffmpeg did not extract frames from {path}.")
        while len(frame_paths) < input_frames:
            frame_paths.append(frame_paths[-1])
        return [
            np.asarray(Image.open(frame_paths[index * tubelet_size]).convert("RGB"))
            for index in range(temporal_slices)
        ]


def _animation_reference_frames(
    *,
    image_path: str | None,
    video_path: str | None,
    temporal_slices: int,
    tubelet_size: int,
    reference_size: int,
) -> list[np.ndarray | None]:
    if video_path is not None:
        return _load_video_reference_frames(
            Path(video_path),
            temporal_slices=temporal_slices,
            tubelet_size=tubelet_size,
            reference_size=reference_size,
        )
    image = _load_reference_image(image_path)
    return [image] * temporal_slices


def _temporal_maps(
    tokens: np.ndarray,
    *,
    grid_shape: tuple[int, ...],
    cluster_pca_dim: int,
    cluster_k: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    from sklearn.cluster import KMeans

    if len(grid_shape) < 3:
        raise ValueError("--animate requires a token grid with at least three dimensions.")
    temporal_slices = int(np.prod(grid_shape[:-2]))
    h, w = int(grid_shape[-2]), int(grid_shape[-1])
    expected_tokens = temporal_slices * h * w
    if tokens.shape[0] != expected_tokens:
        raise ValueError(
            f"Temporal grid {grid_shape} describes {expected_tokens} tokens, "
            f"but the embedding has {tokens.shape[0]}."
        )
    if cluster_k <= 1 or cluster_k >= tokens.shape[0]:
        raise ValueError("--animation-clusters must be greater than 1 and less than token count.")

    standardized = _standardize(tokens.astype(np.float32))
    projection, pca = _pca_projection(standardized, 3)
    if projection.shape[1] < 3:
        projection = np.pad(
            projection,
            ((0, 0), (0, 3 - projection.shape[1])),
            mode="constant",
        )

    rgb = projection[:, :3]
    lo = np.percentile(rgb, 1, axis=0, keepdims=True)
    hi = np.percentile(rgb, 99, axis=0, keepdims=True)
    rgb = np.clip((rgb - lo) / np.maximum(hi - lo, 1e-6), 0.0, 1.0)

    pc1 = projection[:, 0]
    pc1 = (pc1 - pc1.min()) / max(float(pc1.max() - pc1.min()), 1e-6)

    cluster_input = _cluster_space(tokens, cluster_pca_dim)
    labels = KMeans(n_clusters=cluster_k, n_init=10, random_state=seed).fit_predict(
        cluster_input
    )
    shape_2d = (temporal_slices, h, w)
    metadata = {
        "temporal_slices": temporal_slices,
        "selected_2d_grid": [h, w],
        "projection_fit": "all temporal and spatial tokens",
        "cluster_fit": "all temporal and spatial tokens",
        "cluster_method": "kmeans",
        "cluster_k": cluster_k,
        "pca_explained_variance": [
            float(value)
            for value in getattr(pca, "explained_variance_ratio_", np.asarray([]))
        ],
    }
    return (
        rgb.reshape(temporal_slices, h, w, 3),
        pc1.reshape(*shape_2d),
        labels.reshape(*shape_2d),
        metadata,
    )


def _save_temporal_video(
    *,
    output: Path,
    reference_frames: list[np.ndarray | None],
    pca_rgb: np.ndarray,
    pc1: np.ndarray,
    clusters: np.ndarray,
    fps: float,
    tubelet_size: int,
    title: str,
    cluster_k: int,
) -> None:
    plt = _pyplot()
    temporal_slices = int(pca_rgb.shape[0])
    if len(reference_frames) != temporal_slices:
        raise ValueError(
            f"Expected {temporal_slices} reference frames, got {len(reference_frames)}."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="jepas_visualization_video_") as tmp:
        for index in range(temporal_slices):
            fig, axes = plt.subplots(2, 2, figsize=(8.0, 8.0), squeeze=False)
            reference = reference_frames[index]
            if reference is None:
                axes[0, 0].text(
                    0.5,
                    0.5,
                    "no reference",
                    ha="center",
                    va="center",
                    transform=axes[0, 0].transAxes,
                )
            else:
                axes[0, 0].imshow(reference)
            axes[0, 0].set_title("input frame", fontsize=10)
            axes[0, 1].imshow(pca_rgb[index], interpolation="nearest")
            axes[0, 1].set_title("token PCA RGB (global basis)", fontsize=10)
            axes[1, 0].imshow(pc1[index], interpolation="nearest", cmap="magma")
            axes[1, 0].set_title("token PC1 (global basis)", fontsize=10)
            axes[1, 1].imshow(
                clusters[index],
                interpolation="nearest",
                cmap="tab10",
                vmin=0,
                vmax=9,
            )
            axes[1, 1].set_title(f"KMeans (global k={cluster_k})", fontsize=10)
            for axis in axes.reshape(-1):
                axis.set_xticks([])
                axis.set_yticks([])
            first_frame = index * tubelet_size
            last_frame = first_frame + tubelet_size - 1
            fig.suptitle(
                f"{title}\ntemporal slice {index + 1}/{temporal_slices} "
                f"| sampled input frames {first_frame}-{last_frame}",
                fontsize=11,
            )
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
            fig.savefig(Path(tmp) / f"frame_{index:04d}.png", dpi=128)
            plt.close(fig)

        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                f"{fps:g}",
                "-i",
                str(Path(tmp) / "frame_%04d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output),
            ],
            check=True,
        )


def _save_sample_figure(
    *,
    out_path: Path,
    reference_image: np.ndarray | None,
    token_norm: np.ndarray,
    pca_rgb: np.ndarray,
    pc1: np.ndarray,
    clusters: list[ClusterResult],
    grid: tuple[int, int],
    title: str,
) -> None:
    plt = _pyplot()

    cluster_rows: list[list[ClusterResult]] = []
    for result in clusters:
        if not cluster_rows or cluster_rows[-1][0].method != result.method:
            cluster_rows.append([])
        if len(cluster_rows[-1]) == 3:
            cluster_rows.append([])
        cluster_rows[-1].append(result)

    cols = 3
    rows = 1 + len(cluster_rows)
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.0 * rows), squeeze=False)

    if reference_image is None:
        axes[0, 0].imshow(token_norm, interpolation="nearest", cmap="viridis")
        axes[0, 0].set_title("token norm", fontsize=8)
    else:
        axes[0, 0].imshow(reference_image)
        axes[0, 0].set_title("image", fontsize=8)
    axes[0, 0].set_axis_off()

    axes[0, 1].imshow(pca_rgb, interpolation="nearest")
    axes[0, 1].set_title("token PCA RGB", fontsize=8)
    axes[0, 1].set_axis_off()

    axes[0, 2].imshow(pc1, interpolation="nearest", cmap="magma")
    axes[0, 2].set_title("token PC1", fontsize=8)
    axes[0, 2].set_axis_off()

    for row_idx, row_clusters in enumerate(cluster_rows, start=1):
        for col_idx, result in enumerate(row_clusters):
            _render_label_map(axes[row_idx, col_idx], result.labels, grid, f"{result.method}\n{result.params}")

    for ax in axes.reshape(-1):
        if not ax.has_data():
            ax.set_axis_off()

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98), h_pad=2.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "embedding",
        "key",
        "sample_index",
        "grid_shape",
        "slice_index",
        "tokens_shape",
        "method",
        "params",
        "n_clusters",
        "noise_count",
        "cluster_entropy",
        "boundary_fraction",
        "silhouette",
        "calinski_harabasz",
        "davies_bouldin",
        "counts",
        "proportions",
        "connected_components",
        "pca_explained_variance",
        "adjacent_cosine_distance",
        "adjacent_cosine_distance_sd",
        "adjacent_cosine_similarity",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    args = parse_args()
    if args.reference_video is not None and not args.animate:
        raise ValueError("--reference-video requires --animate.")
    if args.animate and args.tubelet_size <= 0:
        raise ValueError("--tubelet-size must be positive.")
    if args.animate and args.reference_size <= 0:
        raise ValueError("--reference-size must be positive.")
    if args.animate and args.animation_fps <= 0:
        raise ValueError("--animation-fps must be positive.")

    embedding_path = Path(args.embedding).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".matplotlib"))

    raw = _load_embedding(embedding_path, args.key)
    grid_shape = _parse_grid_shape(args.grid_shape)
    tokens, token_grid_shape = _prepare_tokens(
        raw,
        sample_index=args.sample_index,
        grid_shape=grid_shape,
    )
    tokens = F.layer_norm(torch.from_numpy(tokens), (tokens.shape[-1],)).numpy()
    slice_tokens, grid_2d, selected_slice = _select_2d_tokens(
        tokens,
        token_grid_shape,
        slice_index=args.slice_index,
    )

    pca_rgb, pc1, pca_stats = _token_pca_maps(slice_tokens, grid_2d)
    continuous_stats = _adjacent_cosine_stats(slice_tokens, grid_2d)
    x_cluster = _cluster_space(slice_tokens, int(args.cluster_pca_dim))
    clusters = _run_clustering(
        x_cluster,
        grid=grid_2d,
        methods=args.methods,
        cluster_k=args.cluster_k,
        dbscan_eps=args.dbscan_eps,
        seed=int(args.seed),
    )

    title = args.title or f"{embedding_path.name} | grid={token_grid_shape} | slice={selected_slice}"
    _save_sample_figure(
        out_path=out_dir / "embedding_visualization.png",
        reference_image=_load_reference_image(args.image),
        token_norm=_token_norm_map(slice_tokens, grid_2d),
        pca_rgb=pca_rgb,
        pc1=pc1,
        clusters=clusters,
        grid=grid_2d,
        title=title,
    )

    animation_metadata: dict[str, Any] | None = None
    animation_path = out_dir / "embedding_visualization.mp4"
    if args.animate:
        temporal_pca_rgb, temporal_pc1, temporal_clusters, animation_metadata = (
            _temporal_maps(
                tokens,
                grid_shape=token_grid_shape,
                cluster_pca_dim=int(args.cluster_pca_dim),
                cluster_k=int(args.animation_clusters),
                seed=int(args.seed),
            )
        )
        reference_frames = _animation_reference_frames(
            image_path=args.image,
            video_path=args.reference_video,
            temporal_slices=int(animation_metadata["temporal_slices"]),
            tubelet_size=int(args.tubelet_size),
            reference_size=int(args.reference_size),
        )
        _save_temporal_video(
            output=animation_path,
            reference_frames=reference_frames,
            pca_rgb=temporal_pca_rgb,
            pc1=temporal_pc1,
            clusters=temporal_clusters,
            fps=float(args.animation_fps),
            tubelet_size=int(args.tubelet_size),
            title=title,
            cluster_k=int(args.animation_clusters),
        )
        animation_metadata.update(
            {
                "reference_video": (
                    portable_path(Path(args.reference_video))
                    if args.reference_video is not None
                    else None
                ),
                "reference_image": (
                    portable_path(Path(args.image)) if args.image is not None else None
                ),
                "reference_protocol": (
                    "sample source video and select the first frame of each token tubelet"
                    if args.reference_video is not None
                    else "repeat the static reference image for every temporal slice"
                ),
                "tubelet_size": int(args.tubelet_size),
                "fps": float(args.animation_fps),
            }
        )

    rows = []
    base_row = {
        "embedding": portable_path(embedding_path),
        "key": args.key,
        "sample_index": args.sample_index,
        "grid_shape": json.dumps(token_grid_shape),
        "slice_index": selected_slice,
        "tokens_shape": json.dumps(list(slice_tokens.shape)),
        "pca_explained_variance": json.dumps(pca_stats["pca_explained_variance"]),
        **continuous_stats,
    }
    for result in clusters:
        row = {
            **base_row,
            "method": result.method,
            "params": result.params,
        }
        row.update(
            {
                key: json.dumps(value) if isinstance(value, dict) else value
                for key, value in result.stats.items()
            }
        )
        rows.append(row)

    _write_summary_csv(out_dir / "summary.csv", rows)
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    outputs = {
        "figure": "embedding_visualization.png",
        "summary_csv": "summary.csv",
        "summary_json": "summary.json",
    }
    if animation_metadata is not None:
        outputs["video"] = animation_path.name

    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "embedding": portable_path(embedding_path),
                "key": args.key,
                "sample_index": args.sample_index,
                "input_shape": list(np.asarray(raw).shape),
                "grid_shape": list(token_grid_shape),
                "selected_2d_grid": list(grid_2d),
                "slice_index": selected_slice,
                "outputs": outputs,
                "animation": animation_metadata,
            },
            indent=2,
        )
        + "\n"
    )

    print(f"[visualize_embedding] wrote outputs to {out_dir}", flush=True)
    print(f"[visualize_embedding] tokens={tuple(slice_tokens.shape)} rows={len(rows)}", flush=True)
    if animation_metadata is not None:
        print(
            f"[visualize_embedding] video={animation_path} "
            f"frames={animation_metadata['temporal_slices']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
