#!/usr/bin/env python3
"""Compute per-sample token-embedding statistics from a batched embeddings.h5.

Adapted from `ijepa_lite`'s `src/ijepa_lite/scripts/visualize_token_embeddings.py`
(https://github.com/VittuD/ijepa_lite), but reads tokens that are already on
disk (`experiments/extract_batch.py` output, shape `(N, T, D)`) instead of
re-running the encoder, and drops the per-sample PNG figures — this is meant
to run over thousands of samples, not a handful for visual inspection.

Clustering is restricted to kmeans + gmm (agglomerative/spectral/dbscan were
dropped). Every other metric from the source script is kept: per-sample PCA
explained variance, adjacent-token cosine smoothness, and per-cluster
n_clusters/counts/proportions/entropy/boundary_fraction/connected_components/
silhouette/calinski_harabasz/davies_bouldin.

Unlike the source script (which only ever handled a square 2D patch grid),
this generalizes every spatial metric to an arbitrary-rank grid so it also
covers the video models' 3D (tubelets x H x W) token layout: adjacency,
boundary_fraction, and connected_components all walk however many grid axes
are present, so a video sample's temporal axis is a real neighbor axis too
(6-connectivity), not just flattened away.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from experiments.artifacts import read_embedding_metadata  # noqa: E402
from experiments.catalog import MODEL_ALIASES, canonical_model_name, model_specs  # noqa: E402


# vjepa2.py / echojepa.py both hardcode tubelet_size=2 when building the
# source-model encoder; this mirrors that to turn --frames back into a
# tubelet count when inferring a video sample's 3D grid shape.
TUBELET_SIZE = 2

CLUSTER_METHODS = ("kmeans", "gmm")
DEFAULT_CLUSTER_K = (2, 3, 4, 6, 8)


@dataclass(frozen=True)
class ClusterResult:
    method: str
    params: str
    labels: np.ndarray
    stats: dict[str, Any]


def infer_grid(token_count: int, *, input_kind: str, frames: int | None) -> tuple[int, ...]:
    """Recover the (2D image or 3D video) token grid shape from metadata alone."""
    if input_kind == "image":
        side = math.isqrt(token_count)
        if side * side != token_count:
            raise ValueError(f"Expected a square image token grid, got {token_count} tokens.")
        return (side, side)
    if input_kind == "video":
        if not frames:
            raise ValueError("Video grid inference requires settings.frames in the embeddings metadata.")
        if frames % TUBELET_SIZE:
            raise ValueError(f"frames={frames} is not a multiple of tubelet_size={TUBELET_SIZE}.")
        tubelets = frames // TUBELET_SIZE
        if token_count % tubelets:
            raise ValueError(f"{token_count} tokens is not divisible by {tubelets} tubelets.")
        spatial = token_count // tubelets
        side = math.isqrt(spatial)
        if side * side != spatial:
            raise ValueError(
                f"Expected a square per-tubelet spatial grid, got {spatial} "
                f"from {token_count} tokens / {tubelets} tubelets."
            )
        return (tubelets, side, side)
    raise ValueError(f"Unsupported input_kind {input_kind!r} for grid inference.")


def _standardize(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mu = x.mean(axis=0, keepdims=True)
    sigma = x.std(axis=0, keepdims=True)
    return (x - mu) / np.maximum(sigma, eps)


def _pca_projection(x: np.ndarray, n_components: int) -> tuple[np.ndarray, Any]:
    from sklearn.decomposition import PCA

    n = max(1, min(int(n_components), x.shape[0], x.shape[1]))
    pca = PCA(n_components=n, random_state=0)
    return pca.fit_transform(x), pca


def pca_explained_variance(tokens: np.ndarray, n_components: int = 3) -> list[float]:
    x = _standardize(tokens.astype(np.float32))
    _, pca = _pca_projection(x, n_components)
    return [float(v) for v in getattr(pca, "explained_variance_ratio_", np.asarray([]))]


def adjacent_token_stats(tokens: np.ndarray, grid: tuple[int, ...]) -> dict[str, float]:
    """Cosine distance between every pair of grid-adjacent tokens, any rank."""
    x = tokens.astype(np.float32)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    x = x.reshape(*grid, -1)

    distances: list[np.ndarray] = []
    for axis in range(len(grid)):
        if grid[axis] < 2:
            continue
        a = np.take(x, indices=range(grid[axis] - 1), axis=axis)
        b = np.take(x, indices=range(1, grid[axis]), axis=axis)
        cosine = np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)
        distances.append((1.0 - cosine).reshape(-1))

    if not distances:
        return {"adjacent_cosine_distance": 0.0, "adjacent_cosine_distance_sd": 0.0, "adjacent_cosine_similarity": 1.0}
    d = np.concatenate(distances)
    return {
        "adjacent_cosine_distance": float(d.mean()),
        "adjacent_cosine_distance_sd": float(d.std(ddof=1)) if d.size > 1 else 0.0,
        "adjacent_cosine_similarity": float(1.0 - d.mean()),
    }


def cluster_space(tokens: np.ndarray, pca_dim: int) -> np.ndarray:
    x = _standardize(tokens.astype(np.float32))
    if pca_dim <= 0:
        return x
    proj, _ = _pca_projection(x, min(pca_dim, x.shape[0] - 1, x.shape[1]))
    return _standardize(proj.astype(np.float32))


def _connected_components(labels: np.ndarray, cluster_label: int) -> int:
    """Flood-fill component count at any grid rank (4-conn for 2D, 6-conn for 3D, ...)."""
    mask = labels == cluster_label
    if not mask.any():
        return 0

    shape = mask.shape
    ndim = mask.ndim
    offsets: list[tuple[int, ...]] = []
    for axis in range(ndim):
        for delta in (-1, 1):
            offset = [0] * ndim
            offset[axis] = delta
            offsets.append(tuple(offset))

    seen = np.zeros_like(mask, dtype=bool)
    components = 0
    for start in np.ndindex(shape):
        if seen[start] or not mask[start]:
            continue
        components += 1
        stack = [start]
        seen[start] = True
        while stack:
            cur = stack.pop()
            for off in offsets:
                nxt = tuple(c + o for c, o in zip(cur, off))
                if all(0 <= n < s for n, s in zip(nxt, shape)) and not seen[nxt] and mask[nxt]:
                    seen[nxt] = True
                    stack.append(nxt)
    return components


def _boundary_fraction(labels_grid: np.ndarray) -> float:
    edges = 0
    total = 0
    for axis in range(labels_grid.ndim):
        if labels_grid.shape[axis] < 2:
            continue
        a = labels_grid.take(indices=range(labels_grid.shape[axis] - 1), axis=axis)
        b = labels_grid.take(indices=range(1, labels_grid.shape[axis]), axis=axis)
        valid = (a >= 0) & (b >= 0)
        total += int(valid.sum())
        edges += int(((a != b) & valid).sum())
    return float(edges / max(total, 1))


def cluster_stats(labels: np.ndarray, x: np.ndarray, grid: tuple[int, ...]) -> dict[str, Any]:
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

    labels = labels.astype(int)
    valid = labels >= 0
    unique = sorted(int(v) for v in np.unique(labels[valid]))
    counts = {str(k): int((labels == k).sum()) for k in unique}
    noise_count = int((labels < 0).sum())
    proportions = {str(k): float(v / max(int(valid.sum()), 1)) for k, v in counts.items()}
    p = np.asarray(list(proportions.values()), dtype=np.float64)
    entropy = float(-(p * np.log2(np.maximum(p, 1e-12))).sum()) if p.size else 0.0

    scores: dict[str, float | None] = {"silhouette": None, "calinski_harabasz": None, "davies_bouldin": None}
    if len(unique) > 1 and int(valid.sum()) > len(unique):
        xv, yv = x[valid], labels[valid]
        scores = {
            "silhouette": float(silhouette_score(xv, yv)),
            "calinski_harabasz": float(calinski_harabasz_score(xv, yv)),
            "davies_bouldin": float(davies_bouldin_score(xv, yv)),
        }

    labels_grid = labels.reshape(*grid)
    components = {str(k): int(_connected_components(labels_grid, k)) for k in unique}
    return {
        "n_clusters": int(len(unique)),
        "noise_count": noise_count,
        "counts": counts,
        "proportions": proportions,
        "cluster_entropy": entropy,
        "boundary_fraction": _boundary_fraction(labels_grid),
        "connected_components": components,
        **scores,
    }


def run_clustering(
    x: np.ndarray,
    *,
    grid: tuple[int, ...],
    methods: tuple[str, ...],
    cluster_k: tuple[int, ...],
    seed: int,
) -> list[ClusterResult]:
    results: list[ClusterResult] = []
    for method in methods:
        if method == "kmeans":
            from sklearn.cluster import KMeans

            for k in cluster_k:
                if k <= 1 or k >= x.shape[0]:
                    continue
                labels = KMeans(n_clusters=int(k), n_init=10, random_state=seed).fit_predict(x)
                results.append(ClusterResult(method, f"k={int(k)}", labels, cluster_stats(labels, x, grid)))
        elif method == "gmm":
            from sklearn.mixture import GaussianMixture

            for k in cluster_k:
                if k <= 1 or k >= x.shape[0]:
                    continue
                labels = GaussianMixture(
                    n_components=int(k), covariance_type="diag", random_state=seed
                ).fit_predict(x)
                results.append(
                    ClusterResult(method, f"k={int(k)},cov=diag", labels, cluster_stats(labels, x, grid))
                )
        else:
            raise ValueError(f"Unsupported clustering method: {method!r}")
    return results


# --- multiprocessing worker -------------------------------------------------

_WORKER: dict[str, Any] = {}


def _worker_init(
    embeddings_path: str,
    grid: tuple[int, ...],
    methods: tuple[str, ...],
    cluster_k: tuple[int, ...],
    cluster_pca_dim: int,
    seed: int,
) -> None:
    _WORKER["handle"] = h5py.File(embeddings_path, "r")
    _WORKER["grid"] = grid
    _WORKER["methods"] = methods
    _WORKER["cluster_k"] = cluster_k
    _WORKER["cluster_pca_dim"] = cluster_pca_dim
    _WORKER["seed"] = seed


def _process_index(index: int) -> tuple[int, list[dict[str, Any]]]:
    handle = _WORKER["handle"]
    grid = _WORKER["grid"]
    tokens = handle["tokens"][index].astype(np.float32)

    explained_variance = pca_explained_variance(tokens)
    continuous_stats = adjacent_token_stats(tokens, grid)
    x_cluster = cluster_space(tokens, int(_WORKER["cluster_pca_dim"]))
    clusters = run_clustering(
        x_cluster,
        grid=grid,
        methods=_WORKER["methods"],
        cluster_k=_WORKER["cluster_k"],
        seed=int(_WORKER["seed"]),
    )

    rows: list[dict[str, Any]] = []
    for result in clusters:
        row: dict[str, Any] = {
            "method": result.method,
            "params": result.params,
            "pca_explained_variance": json.dumps(explained_variance),
        }
        row.update(continuous_stats)
        row.update(
            {key: json.dumps(value) if isinstance(value, dict) else value for key, value in result.stats.items()}
        )
        rows.append(row)
    return index, rows


# --- CLI ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    choices = tuple(sorted((*model_specs(), *MODEL_ALIASES)))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=choices)
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=None,
        help="Defaults to experiments/outputs/bulk_2048/<model>/embeddings.h5.",
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="Defaults alongside --embeddings.")
    parser.add_argument("--methods", nargs="+", default=CLUSTER_METHODS, choices=CLUSTER_METHODS)
    parser.add_argument("--cluster-k", nargs="+", type=int, default=DEFAULT_CLUSTER_K)
    parser.add_argument("--cluster-pca-dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=32, help="Process pool size.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N samples (testing).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_name = canonical_model_name(args.model)

    embeddings_path = (
        args.embeddings
        or ROOT_DIR / "experiments" / "outputs" / "bulk_2048" / model_name / "embeddings.h5"
    ).expanduser().resolve()
    if not embeddings_path.is_file():
        raise FileNotFoundError(embeddings_path)
    out_dir = (args.out_dir or embeddings_path.parent / "token_metrics").expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = read_embedding_metadata(embeddings_path)
    sources = meta.get("sources")
    if not sources:
        raise ValueError(f"{embeddings_path} has no per-row sources; was it written by extract_batch.py?")
    token_shape = meta["token_shape"]
    n_total, token_count = int(token_shape[0]), int(token_shape[-2])
    input_kind = meta.get("input_kind")
    frames = (meta.get("settings") or {}).get("frames")
    grid = infer_grid(token_count, input_kind=input_kind, frames=frames)

    n = n_total if args.limit is None else min(int(args.limit), n_total)
    methods = tuple(args.methods)
    cluster_k = tuple(args.cluster_k)

    print(f"[token_metrics] model={model_name} n={n}/{n_total} grid={grid} methods={methods} cluster_k={cluster_k}")

    all_rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_worker_init,
        initargs=(str(embeddings_path), grid, methods, cluster_k, int(args.cluster_pca_dim), int(args.seed)),
    ) as pool:
        done = 0
        for index, rows in pool.map(_process_index, range(n)):
            source = sources[index]
            for row in rows:
                all_rows.append(
                    {
                        "model": model_name,
                        "dataset_index": index,
                        "sample_id": f"{model_name}_{index:05d}",
                        "source": source,
                        **row,
                    }
                )
            done += 1
            if done % 200 == 0 or done == n:
                print(f"  {done}/{n} samples done ({len(all_rows)} rows)")

    fieldnames = [
        "model",
        "dataset_index",
        "sample_id",
        "source",
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
    summary_csv = out_dir / "summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    (out_dir / "summary.json").write_text(json.dumps(all_rows, indent=2) + "\n")
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "model": model_name,
                "embeddings": str(embeddings_path),
                "n_samples": n,
                "grid": list(grid),
                "input_kind": input_kind,
                "methods": list(methods),
                "cluster_k": list(cluster_k),
                "cluster_pca_dim": int(args.cluster_pca_dim),
                "seed": int(args.seed),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[token_metrics] wrote {len(all_rows)} rows to {summary_csv}")


if __name__ == "__main__":
    main()
