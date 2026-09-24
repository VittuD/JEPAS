#!/usr/bin/env python3
"""Null-model baselines for token_metrics.py: same pipeline, random tokens.

Several token_metrics.py metrics are partly artifacts of token-grid geometry
(T tokens, D dims) rather than genuine model structure -- e.g. more tokens
mechanically gives more room for connected components, and PCA explained
variance concentrates more when D is small relative to T. Comparing raw
metric values across models with different (T, D, grid) conflates "model is
more structured" with "model happens to have this shape."

This computes the same metrics on pure Gaussian noise at each model's exact
(T, D, grid), repeated over --draws seeds, giving a null mean/sd per
metric x method x params. token_metrics_aggregate.py can then normalize real
values against this null (z-score or ratio) instead of comparing raw numbers.

Geometry is read from an existing embeddings.h5 for --model (any one works,
since geometry is fixed per model regardless of dataset) rather than
re-deriving it from catalog fields, so this always matches what
token_metrics.py actually saw.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import h5py
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from experiments.artifacts import read_embedding_metadata  # noqa: E402
from experiments.catalog import MODEL_ALIASES, canonical_model_name, model_specs  # noqa: E402
from experiments.token_metrics import (  # noqa: E402
    CLUSTER_METHODS,
    DEFAULT_CLUSTER_K,
    adjacent_token_stats,
    cluster_space,
    infer_grid,
    pca_explained_variance,
    run_clustering,
)


def geometry_from_embeddings(embeddings_path: Path) -> tuple[int, int, tuple[int, ...], str, int | None]:
    meta = read_embedding_metadata(embeddings_path)
    token_shape = meta["token_shape"]
    token_count, dim = int(token_shape[-2]), int(token_shape[-1])
    input_kind = meta.get("input_kind")
    frames = (meta.get("settings") or {}).get("frames")
    grid = infer_grid(token_count, input_kind=input_kind, frames=frames)
    return token_count, dim, grid, input_kind, frames


_WORKER: dict[str, Any] = {}


def _worker_init(
    token_count: int,
    dim: int,
    grid: tuple[int, ...],
    methods: tuple[str, ...],
    cluster_k: tuple[int, ...],
    cluster_pca_dim: int,
    base_seed: int,
) -> None:
    _WORKER.update(
        token_count=token_count,
        dim=dim,
        grid=grid,
        methods=methods,
        cluster_k=cluster_k,
        cluster_pca_dim=cluster_pca_dim,
        base_seed=base_seed,
    )


def _process_draw(draw_index: int) -> tuple[int, list[dict[str, Any]]]:
    rng = np.random.default_rng(_WORKER["base_seed"] + draw_index)
    tokens = rng.normal(size=(_WORKER["token_count"], _WORKER["dim"])).astype(np.float32)
    grid = _WORKER["grid"]

    explained_variance = pca_explained_variance(tokens)
    continuous_stats = adjacent_token_stats(tokens, grid)
    x_cluster = cluster_space(tokens, int(_WORKER["cluster_pca_dim"]))
    clusters = run_clustering(
        x_cluster,
        grid=grid,
        methods=_WORKER["methods"],
        cluster_k=_WORKER["cluster_k"],
        seed=_WORKER["base_seed"] + draw_index,
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
    return draw_index, rows


def parse_args() -> argparse.Namespace:
    choices = tuple(sorted((*model_specs(), *MODEL_ALIASES)))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=choices)
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=None,
        help="Any embeddings.h5 for --model, just to read its (T, D, grid). Defaults to the bulk_2048 native run.",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--methods", nargs="+", default=CLUSTER_METHODS, choices=CLUSTER_METHODS)
    parser.add_argument("--cluster-k", nargs="+", type=int, default=DEFAULT_CLUSTER_K)
    parser.add_argument("--cluster-pca-dim", type=int, default=16)
    parser.add_argument("--draws", type=int, default=30, help="Number of random-token samples to average over.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    model_name = canonical_model_name(args.model)

    embeddings_path = (
        args.embeddings
        or ROOT_DIR / "experiments" / "outputs" / "bulk_2048" / model_name / "embeddings.h5"
    ).expanduser().resolve()
    if not embeddings_path.is_file():
        raise FileNotFoundError(embeddings_path)
    out_dir = (
        args.out_dir or ROOT_DIR / "experiments" / "outputs" / "null_baselines" / model_name
    ).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    token_count, dim, grid, input_kind, frames = geometry_from_embeddings(embeddings_path)
    methods, cluster_k = tuple(args.methods), tuple(args.cluster_k)

    print(
        f"[token_metrics_null] model={model_name} T={token_count} D={dim} grid={grid} "
        f"draws={args.draws} methods={methods} cluster_k={cluster_k}"
    )

    all_rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_worker_init,
        initargs=(token_count, dim, grid, methods, cluster_k, int(args.cluster_pca_dim), int(args.seed)),
    ) as pool:
        for draw_index, rows in pool.map(_process_draw, range(args.draws)):
            for row in rows:
                all_rows.append({"model": model_name, "draw_index": draw_index, **row})

    fieldnames = [
        "model",
        "draw_index",
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
    summary_csv = out_dir / "null_summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    (out_dir / "null_summary.json").write_text(json.dumps(all_rows, indent=2) + "\n")
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "model": model_name,
                "geometry_source": str(embeddings_path),
                "token_count": token_count,
                "dim": dim,
                "grid": list(grid),
                "input_kind": input_kind,
                "frames": frames,
                "draws": args.draws,
                "methods": list(methods),
                "cluster_k": list(cluster_k),
                "seed": args.seed,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[token_metrics_null] wrote {len(all_rows)} rows to {summary_csv}")


if __name__ == "__main__":
    main()
