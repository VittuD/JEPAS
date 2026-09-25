#!/usr/bin/env python3
"""Side-by-side token maps for every model on the same inputs of a bulk run.

Reads a bulk run directory (`<bulk>/<dataset>/<model>/embeddings.h5` plus
`token_metrics/summary.csv`, as produced by scripts/leonardo_bulk.sbatch) and, for
each chosen sample, writes one figure: a row of input frames, then for every model
a PCA-RGB row and a k-means row. Video token grids are shown at the first, middle
and last tubelet.

The k-means maps are computed with the same pipeline and seed as token_metrics.py
(cluster_space + run_clustering), so the map is exactly what was scored; the driver
compares its boundary fraction / component count with summary.csv as a check.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from experiments.token_metrics import cluster_space, run_clustering  # noqa: E402
from experiments.token_metrics_null import geometry_from_embeddings  # noqa: E402
from experiments.visualize import (  # noqa: E402
    _normalize_projection,
    _projection,
    _pyplot,
    _video_frames,
)

PICKS = ("random", "low", "median", "high")
TUBELET_SIZE = 2


def load_sources(embeddings: Path) -> list[str]:
    import h5py

    with h5py.File(embeddings, "r") as handle:
        return [x.decode() if isinstance(x, bytes) else str(x) for x in handle["sources"][...]]


def check_alignment(sources_by_model: dict[str, list[str]]) -> None:
    """Every model must have seen the same inputs in the same order."""
    names = sorted(sources_by_model)
    reference = sources_by_model[names[0]]
    for name in names[1:]:
        other = sources_by_model[name]
        if len(other) != len(reference):
            raise ValueError(f"{name} has {len(other)} samples, {names[0]} has {len(reference)}.")
        for index, (a, b) in enumerate(zip(reference, other)):
            if a != b:
                raise ValueError(f"Sample {index} differs: {names[0]}={a!r} vs {name}={b!r}.")


def load_metric_by_index(summary_csv: Path, *, metric: str, method: str, params: str) -> dict[int, float]:
    values: dict[int, float] = {}
    with summary_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["method"] == method and row["params"] == params:
                values[int(row["dataset_index"])] = float(row[metric])
    return values


def select_indices(
    metric_by_model: dict[str, dict[int, float]],
    *,
    pick: str,
    count: int,
    seed: int,
) -> list[int]:
    """Choose sample indices that are meaningful for *all* models at once.

    Each model's metric is converted to a percentile rank, so models with different
    scales are comparable, then averaged over models. `low`/`high` take the samples
    that are smoothest/roughest on average, `median` those closest to the middle,
    `random` a seeded draw.
    """
    if pick not in PICKS:
        raise ValueError(f"pick must be one of {PICKS}, got {pick!r}.")
    common = sorted(set.intersection(*(set(v) for v in metric_by_model.values())))
    if not common:
        raise ValueError("No sample index is present for every model.")
    count = min(count, len(common))
    if pick == "random":
        rng = np.random.default_rng(seed)
        return sorted(int(i) for i in rng.choice(common, size=count, replace=False))
    ranks = []
    for values in metric_by_model.values():
        column = np.asarray([values[i] for i in common])
        ranks.append(column.argsort().argsort() / max(len(common) - 1, 1))
    mean_rank = np.mean(ranks, axis=0)
    if pick == "low":
        order = np.argsort(mean_rank)
    elif pick == "high":
        order = np.argsort(-mean_rank)
    else:
        order = np.argsort(np.abs(mean_rank - 0.5))
    return [common[int(i)] for i in order[:count]]


def _reference_frames(source: Path, *, kind: str, grid: tuple[int, ...], size: int) -> list[np.ndarray] | None:
    from PIL import Image

    try:
        if kind == "image":
            image = Image.open(source).convert("RGB")
            side = min(image.size)
            left, top = (image.width - side) // 2, (image.height - side) // 2
            image = image.crop((left, top, left + side, top + side)).resize((size, size))
            return [np.asarray(image)]
        return _video_frames(source, temporal_slices=grid[0], tubelet_size=TUBELET_SIZE, size=size)
    except Exception as error:  # a missing/undecodable input must not abort the figure
        print(f"  reference unavailable for {source.name}: {error}", file=sys.stderr)
        return None


def _model_maps(
    tokens: np.ndarray, grid: tuple[int, ...], *, k: int, cluster_pca_dim: int, seed: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """PCA-RGB and k-means label maps over the whole token grid, plus the scored stats."""
    projected, _ = _projection(tokens)
    rgb, _ = _normalize_projection(projected)
    rgb = rgb.reshape(*grid, 3)
    x = cluster_space(tokens, cluster_pca_dim)
    result = run_clustering(x, grid=grid, methods=("kmeans",), cluster_k=(k,), seed=seed)[0]
    return rgb, result.labels.reshape(grid), result.stats


def _slice_columns(grid: tuple[int, ...]) -> list[int | None]:
    if len(grid) == 2:
        return [None]
    t = grid[0]
    return sorted({0, t // 2, t - 1})


def _take(array: np.ndarray, grid: tuple[int, ...], t: int | None) -> np.ndarray:
    return array if t is None else array[t]


def render_sample(
    output: Path,
    *,
    title: str,
    grid_by_model: dict[str, tuple[int, ...]],
    maps: dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]],
    reference: list[np.ndarray] | None,
    reference_grid: tuple[int, ...],
    k: int,
) -> None:
    plt = _pyplot()
    models = list(maps)
    columns = _slice_columns(reference_grid)
    rows = 1 + 2 * len(models)
    figure, axes = plt.subplots(
        rows, len(columns), figsize=(2.6 * len(columns) + 1.2, 2.3 * rows), squeeze=False, layout="constrained"
    )
    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])
    for c, t in enumerate(columns):
        axis = axes[0][c]
        if reference is not None:
            axis.imshow(reference[0 if t is None else min(t, len(reference) - 1)])
        axis.set_title("input" if t is None else f"tubelet {t}", fontsize=9)
    axes[0][0].set_ylabel("input", fontsize=8)
    for m, model in enumerate(models):
        rgb, labels, stats = maps[model]
        grid = grid_by_model[model]
        cols = _slice_columns(grid)
        components = sum(stats["connected_components"].values())
        for c in range(len(columns)):
            t = cols[min(c, len(cols) - 1)]
            axes[1 + 2 * m][c].imshow(_take(rgb, grid, t), interpolation="nearest")
            axes[2 + 2 * m][c].imshow(
                _take(labels, grid, t), interpolation="nearest", cmap="tab10", vmin=0, vmax=9
            )
        axes[1 + 2 * m][0].set_ylabel(f"{model}\nPCA RGB", fontsize=8)
        axes[2 + 2 * m][0].set_ylabel(
            f"k-means k={k}\nbf={stats['boundary_fraction']:.2f} cc={components}", fontsize=8
        )
    figure.suptitle(title, fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=130)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bulk_dir", type=Path)
    p.add_argument("dataset")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--indices", help="Comma-separated sample indices (overrides --pick).")
    p.add_argument("--pick", choices=PICKS, default="random")
    p.add_argument("--count", type=int, default=4)
    p.add_argument("--models", nargs="+", help="Default: every model found for the dataset.")
    p.add_argument("--k", type=int, default=3, help="Must be one of the k used in the bulk run.")
    p.add_argument("--metric", default="boundary_fraction", help="Per-sample metric used by --pick.")
    p.add_argument("--cluster-pca-dim", type=int, help="Default: the value recorded by the bulk token_metrics run.")
    p.add_argument("--cluster-seed", type=int, help="Default: the seed recorded by the bulk token_metrics run.")
    p.add_argument("--seed", type=int, default=0, help="Seed for --pick random.")
    p.add_argument("--reference-size", type=int, default=256)
    p.add_argument("--no-reference", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import h5py

    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "jepas-matplotlib"))
    dataset_dir = args.bulk_dir / args.dataset
    models = args.models or sorted(
        p.name for p in dataset_dir.iterdir() if (p / "embeddings.h5").is_file()
    )
    if not models:
        raise SystemExit(f"No embeddings under {dataset_dir}.")
    sources = {m: load_sources(dataset_dir / m / "embeddings.h5") for m in models}
    check_alignment(sources)
    print(f"alignment OK: {len(next(iter(sources.values())))} samples identical across {models}")

    # Reproduce the scored clustering exactly: reuse the seed / PCA dim the bulk run recorded.
    recorded = json.loads((dataset_dir / models[0] / "token_metrics" / "manifest.json").read_text())
    cluster_seed = recorded["seed"] if args.cluster_seed is None else args.cluster_seed
    cluster_pca_dim = recorded["cluster_pca_dim"] if args.cluster_pca_dim is None else args.cluster_pca_dim
    print(f"clustering: seed={cluster_seed} pca_dim={cluster_pca_dim}")

    params = f"k={args.k}"
    scored = {
        m: load_metric_by_index(
            dataset_dir / m / "token_metrics" / "summary.csv",
            metric=args.metric,
            method="kmeans",
            params=params,
        )
        for m in models
    }
    if args.indices:
        indices = [int(i) for i in args.indices.split(",")]
    else:
        indices = select_indices(scored, pick=args.pick, count=args.count, seed=args.seed)
    print(f"samples ({args.pick if not args.indices else 'explicit'}): {indices}")

    geometry = {m: geometry_from_embeddings(dataset_dir / m / "embeddings.h5") for m in models}
    handles = {m: h5py.File(dataset_dir / m / "embeddings.h5", "r") for m in models}
    reference_model = next(iter(models))
    kind, reference_grid = geometry[reference_model][3], geometry[reference_model][2]
    for index in indices:
        source = Path(sources[reference_model][index])
        maps = {}
        for m in models:
            tokens = handles[m]["tokens"][index].astype(np.float32)
            grid = geometry[m][2]
            maps[m] = _model_maps(tokens, grid, k=args.k, cluster_pca_dim=cluster_pca_dim, seed=cluster_seed)
            recorded = scored[m].get(index)
            if args.metric == "boundary_fraction" and recorded is not None:
                drift = abs(recorded - maps[m][2]["boundary_fraction"])
                if drift > 1e-6:
                    print(f"  WARNING {m}[{index}]: recomputed bf differs from summary.csv by {drift:.2g}", file=sys.stderr)
        reference = None if args.no_reference else _reference_frames(
            source, kind=kind, grid=reference_grid, size=args.reference_size
        )
        output = args.out_dir / f"{args.dataset}_{index:04d}.png"
        render_sample(
            output,
            title=f"{args.dataset}[{index}]  {source.name}",
            grid_by_model={m: geometry[m][2] for m in models},
            maps=maps,
            reference=reference,
            reference_grid=reference_grid,
            k=args.k,
        )
        print(f"wrote {output}")
    for handle in handles.values():
        handle.close()


if __name__ == "__main__":
    main()
