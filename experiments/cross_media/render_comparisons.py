#!/usr/bin/env python3
"""Render per-input and full-matrix comparisons for cross-media results."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from experiments.common_resolution.render_comparison import load_maps  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def setup_plotting(out_dir: Path):
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def render_input_comparison(
    plt,
    *,
    input_name: str,
    records: list[dict[str, object]],
    resolution_label: str,
    output: Path,
) -> None:
    columns = ("adapted model input", "token PCA RGB", "token PC1", "KMeans (k=4)")
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
        axes[row, 0].set_ylabel(str(record["model"]), fontsize=10)
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(f"{input_name} across models | {resolution_label}", fontsize=12)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def render_matrix(
    plt,
    *,
    records_by_pair: dict[tuple[str, str], dict[str, object]],
    models: list[str],
    inputs: list[str],
    map_kind: str,
    output: Path,
) -> None:
    fig, axes = plt.subplots(
        len(models),
        len(inputs),
        figsize=(3.0 * len(inputs), 2.8 * len(models)),
        squeeze=False,
        layout="constrained",
    )
    for column, input_name in enumerate(inputs):
        axes[0, column].set_title(input_name, fontsize=10)
    for row, model in enumerate(models):
        for column, input_name in enumerate(inputs):
            record = records_by_pair.get((input_name, model))
            if record is None:
                axes[row, column].text(
                    0.5,
                    0.5,
                    "not run",
                    ha="center",
                    va="center",
                    transform=axes[row, column].transAxes,
                )
                axes[row, column].set_facecolor("#eeeeee")
                if column == 0:
                    axes[row, column].set_ylabel(model, fontsize=10)
                axes[row, column].set_xticks([])
                axes[row, column].set_yticks([])
                continue
            _preview, pca_rgb, pc1, _clusters = load_maps(record)
            if map_kind == "pca_rgb":
                axes[row, column].imshow(pca_rgb, interpolation="nearest")
            else:
                axes[row, column].imshow(pc1, interpolation="nearest", cmap="magma")
            if column == 0:
                axes[row, column].set_ylabel(model, fontsize=10)
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    fig.suptitle(
        "Token PCA RGB by model and input"
        if map_kind == "pca_rgb"
        else "Token PC1 by model and input",
        fontsize=12,
    )
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    records = manifest["results"]
    models = list(manifest["models"])
    inputs = list(manifest["inputs"])
    records_by_pair = {
        (str(record["input_name"]), str(record["model"])): record
        for record in records
    }
    resolution_label = (
        "model-native resolutions"
        if manifest.get("resolution_mode") == "native"
        else f"{manifest['resolution']}x{manifest['resolution']}"
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plt = setup_plotting(args.out_dir)

    for input_name in inputs:
        input_records = [
            records_by_pair[(input_name, model)]
            for model in models
            if (input_name, model) in records_by_pair
        ]
        if not input_records:
            continue
        render_input_comparison(
            plt,
            input_name=input_name,
            records=input_records,
            resolution_label=resolution_label,
            output=args.out_dir / f"{input_name}.png",
        )
    render_matrix(
        plt,
        records_by_pair=records_by_pair,
        models=models,
        inputs=inputs,
        map_kind="pca_rgb",
        output=args.out_dir / "matrix_pca_rgb.png",
    )
    render_matrix(
        plt,
        records_by_pair=records_by_pair,
        models=models,
        inputs=inputs,
        map_kind="pc1",
        output=args.out_dir / "matrix_pc1.png",
    )
    print(f"[cross_media] wrote comparisons to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
