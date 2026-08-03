#!/usr/bin/env python3
"""Render a compact side-by-side view from a common-resolution manifest."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from experiments.visualization.visualize_embedding import (  # noqa: E402
    _cluster_space,
    _prepare_tokens,
    _select_2d_tokens,
    _token_pca_maps,
)
from experiments.shared.provenance import resolve_portable_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def visualization_slice_index(
    record: dict[str, object],
    *,
    temporal_grid: int | None,
) -> int | None:
    if temporal_grid is None:
        return None

    artifacts = record.get("artifacts")
    if isinstance(artifacts, dict):
        manifest_value = artifacts.get("visualization_manifest")
        if manifest_value:
            manifest_path = resolve_portable_path(str(manifest_value))
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                selected = manifest.get("slice_index")
                if selected not in (None, "all"):
                    return int(selected)

    return 0


def load_maps(record: dict[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from PIL import Image
    from sklearn.cluster import KMeans

    embedding = np.load(resolve_portable_path(str(record["embedding"])))
    spatial = tuple(int(value) for value in record["spatial_grid"])
    temporal = record.get("temporal_grid")
    grid_shape = (int(temporal), *spatial) if temporal is not None else spatial

    tokens, token_grid = _prepare_tokens(
        embedding,
        sample_index=None,
        grid_shape=grid_shape,
    )
    tokens = F.layer_norm(torch.from_numpy(tokens), (tokens.shape[-1],)).numpy()
    slice_tokens, grid_2d, _ = _select_2d_tokens(
        tokens,
        token_grid,
        slice_index=visualization_slice_index(
            record,
            temporal_grid=int(temporal) if temporal is not None else None,
        ),
    )
    pca_rgb, pc1, _ = _token_pca_maps(slice_tokens, grid_2d)

    cluster_input = _cluster_space(slice_tokens, pca_dim=16)
    labels = KMeans(n_clusters=4, n_init=10, random_state=0).fit_predict(cluster_input)
    clusters = labels.reshape(*grid_2d)

    artifact_preview = record.get("artifacts", {}).get("preview", {}).get("path")
    if artifact_preview:
        preview_path = resolve_portable_path(str(artifact_preview))
    else:
        figure_path = resolve_portable_path(str(record["figure"]))
        preview_name = "input_frame.jpg" if temporal is not None else "input.jpg"
        preview_path = figure_path.parents[1] / preview_name
    preview = np.asarray(Image.open(preview_path).convert("RGB"))
    return preview, pca_rgb, pc1, clusters


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    records = manifest.get("models", [])
    if not records:
        raise ValueError(f"{args.manifest} contains no model records.")

    output = args.output or args.manifest.with_name("comparison.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output.parent / ".matplotlib"))

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    columns = ("model input", "token PCA RGB", "token PC1", "KMeans (k=4)")
    fig, axes = plt.subplots(
        len(records),
        len(columns),
        figsize=(3.0 * len(columns), 2.8 * len(records)),
        squeeze=False,
        layout="constrained",
    )

    for column, title in enumerate(columns):
        axes[0, column].set_title(title, fontsize=10)

    for row, record in enumerate(records):
        preview, pca_rgb, pc1, clusters = load_maps(record)
        axes[row, 0].imshow(preview)
        axes[row, 1].imshow(pca_rgb, interpolation="nearest")
        axes[row, 2].imshow(pc1, interpolation="nearest", cmap="magma")
        axes[row, 3].imshow(clusters, interpolation="nearest", cmap="tab10", vmin=0, vmax=9)
        axes[row, 0].set_ylabel(str(record["name"]), fontsize=10)
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        f"Common input resolution: {manifest['resolution']}x{manifest['resolution']}",
        fontsize=12,
    )
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"[common_resolution] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
