#!/usr/bin/env python3
"""Compare JEPA embeddings across matched or all catalogued media."""

from __future__ import annotations

import argparse
import html
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT_DIR / "experiments" / "outputs" / "comparisons"
sys.path.insert(0, str(ROOT_DIR))

from experiments.catalog import (  # noqa: E402
    MATCHED_INPUT,
    MediaSpec,
    ModelSpec,
    default_compare_models,
    media_specs,
    model_specs,
)
from experiments.device import add_runtime_arguments, resolve_runtime  # noqa: E402
from experiments.extract import extract_one, extraction_command  # noqa: E402
from experiments.provenance import (  # noqa: E402
    file_provenance,
    model_code_provenance,
    portable_path,
    run_command,
)
from experiments.visualize import (  # noqa: E402
    _kmeans_map,
    _prepare_tokens,
    _select_2d_tokens,
    _token_pca_maps,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairing", choices=("matched", "all"), default="all")
    parser.add_argument(
        "--resolution",
        default="384",
        help="Square input size or 'native' for each model's native size.",
    )
    parser.add_argument("--models", nargs="+", choices=tuple(model_specs()))
    parser.add_argument("--inputs", nargs="+", choices=tuple(media_specs()))
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    add_runtime_arguments(parser)
    return parser.parse_args()


def resolution_for(spec: ModelSpec, value: str) -> int:
    if value == "native":
        return spec.native_resolution
    resolution = int(value)
    if resolution <= 0:
        raise ValueError("--resolution must be positive or 'native'.")
    return resolution


def selected_pairs(args: argparse.Namespace) -> list[tuple[ModelSpec, MediaSpec]]:
    models = model_specs()
    media = media_specs()
    model_names = list(dict.fromkeys(args.models or default_compare_models()))
    input_names = list(dict.fromkeys(args.inputs or media))
    pairs = []
    for model_name in model_names:
        model = models[model_name]
        if model.modality == "volume":
            continue
        names = [MATCHED_INPUT[model_name]] if args.pairing == "matched" else input_names
        pairs.extend((model, media[name]) for name in names)
    return pairs


def square_image(source: Path, destination: Path, resolution: int) -> None:
    image = Image.open(source).convert("RGB")
    side = min(image.size)
    left = (image.width - side) // 2
    top = (image.height - side) // 2
    image = image.crop((left, top, left + side, top + side))
    image = image.resize((resolution, resolution), Image.Resampling.BICUBIC)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, quality=95)


def first_video_frame(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
            "-map", "0:v:0", "-frames:v", "1", str(destination),
        ],
        check=True,
    )


def prepare_inputs(
    pairs: list[tuple[ModelSpec, MediaSpec]],
    run_dir: Path,
    resolution_value: str,
    *,
    force: bool,
) -> tuple[dict[str, Path], dict[tuple[str, int], Path]]:
    first_frames: dict[str, Path] = {}
    references: dict[tuple[str, int], Path] = {}
    input_dir = run_dir / "inputs"
    for _model, media in pairs:
        if media.modality == "video" and media.name not in first_frames:
            frame = input_dir / f"{media.name}_frame0.jpg"
            if force or not frame.exists():
                first_video_frame(media.path, frame)
            first_frames[media.name] = frame
    for model, media in pairs:
        resolution = resolution_for(model, resolution_value)
        key = (media.name, resolution)
        if key in references:
            continue
        source = first_frames[media.name] if media.modality == "video" else media.path
        reference = input_dir / f"{media.name}_{resolution}.jpg"
        if force or not reference.exists():
            square_image(source, reference, resolution)
        references[key] = reference
    return first_frames, references


def effective_input(
    model: ModelSpec,
    media: MediaSpec,
    first_frames: dict[str, Path],
) -> tuple[Path, str]:
    if model.modality == "image" and media.modality == "video":
        return first_frames[media.name], "video_frame0_to_image"
    if model.modality == "video" and media.modality == "image":
        return media.path, "image_repeated_as_video"
    return media.path, "native"


def visualization_command(
    *,
    tokens: Path,
    output_dir: Path,
    reference: Path,
    reference_video: Path | None,
    model: ModelSpec,
    media: MediaSpec,
    resolution: int,
    frames: int,
) -> list[str]:
    temporal = frames // 2 if model.modality == "video" else None
    spatial = resolution // model.patch_size
    grid = (temporal, spatial, spatial) if temporal is not None else (spatial, spatial)
    command = [
        portable_path(Path(sys.executable)),
        "experiments/visualize.py",
        portable_path(tokens),
        "--out-dir",
        portable_path(output_dir),
        "--grid-shape",
        "x".join(map(str, grid)),
        "--slice-index",
        "0" if temporal is not None else "0",
        "--image",
        portable_path(reference),
        "--title",
        f"{model.name} on {media.name} at {resolution}px",
    ]
    if temporal is not None:
        command.extend(("--animate", "--reference-size", str(resolution)))
        if reference_video is not None:
            command.extend(("--reference-video", portable_path(reference_video)))
    return command


def run_pair(
    model: ModelSpec,
    media: MediaSpec,
    *,
    first_frames: dict[str, Path],
    references: dict[tuple[str, int], Path],
    run_dir: Path,
    resolution_value: str,
    frames: int,
    device: str,
    precision: str,
    force: bool,
) -> dict[str, Any]:
    resolution = resolution_for(model, resolution_value)
    pair_dir = run_dir / media.name / model.name
    source, adaptation = effective_input(model, media, first_frames)
    extraction = extract_one(
        model,
        source,
        pair_dir,
        device=device,
        precision=precision,
        image_size=resolution,
        frames=frames,
        force=force,
    )
    reference = references[(media.name, resolution)]
    visualization_dir = pair_dir / "visualization"
    command = visualization_command(
        tokens=Path(extraction["tokens"]),
        output_dir=visualization_dir,
        reference=reference,
        reference_video=source if model.modality == "video" and media.modality == "video" else None,
        model=model,
        media=media,
        resolution=resolution,
        frames=frames,
    )
    figure = visualization_dir / "visualization.png"
    video = visualization_dir / "visualization.mp4"
    if force or not figure.exists() or (model.modality == "video" and not video.exists()):
        run_command(command, cwd=ROOT_DIR, log_path=pair_dir / "visualize.log")
    return {
        "model": model.name,
        "input": media.name,
        "input_modality": media.modality,
        "model_modality": model.modality,
        "adaptation": adaptation,
        "resolution": resolution,
        "patch_size": model.patch_size,
        "spatial_grid": [resolution // model.patch_size] * 2,
        "temporal_grid": frames // 2 if model.modality == "video" else None,
        "tokens": extraction["tokens"],
        "pooled": extraction["pooled"],
        "token_shape": extraction["token_shape"],
        "reference": portable_path(reference),
        "figure": portable_path(figure),
        "video": portable_path(video) if video.exists() else None,
        "extract_command": extraction["command"],
        "visualize_command": shlex.join(command),
    }


def load_maps(record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    embedding = np.load(ROOT_DIR / record["tokens"])
    spatial = tuple(record["spatial_grid"])
    temporal = record["temporal_grid"]
    grid = (temporal, *spatial) if temporal is not None else spatial
    tokens, grid = _prepare_tokens(embedding, sample_index=None, grid_shape=grid)
    tokens = F.layer_norm(torch.from_numpy(tokens), (tokens.shape[-1],)).numpy()
    selected, grid_2d, _ = _select_2d_tokens(tokens, grid, 0 if temporal else None)
    pca_rgb, pc1, _ = _token_pca_maps(selected, grid_2d)
    labels = _kmeans_map(selected, grid_2d, 4)
    reference = np.asarray(Image.open(ROOT_DIR / record["reference"]).convert("RGB"))
    return reference, pca_rgb, pc1, labels


def render_comparisons(records: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in sorted(records, key=lambda item: (item["input"], item["model"])):
        grouped.setdefault(record["input"], []).append(record)
    for input_name, rows in grouped.items():
        figure, axes = plt.subplots(
            len(rows), 4, figsize=(12, 2.8 * len(rows)), squeeze=False, layout="constrained"
        )
        for column, title in enumerate(("input", "PCA RGB", "PC1", "KMeans (k=4)")):
            axes[0, column].set_title(title)
        for row, record in enumerate(rows):
            reference, pca_rgb, pc1, labels = load_maps(record)
            axes[row, 0].imshow(reference)
            axes[row, 1].imshow(pca_rgb, interpolation="nearest")
            axes[row, 2].imshow(pc1, interpolation="nearest", cmap="magma")
            axes[row, 3].imshow(labels, interpolation="nearest", cmap="tab10", vmin=0, vmax=9)
            axes[row, 0].set_ylabel(record["model"])
            for axis in axes[row]:
                axis.set_xticks([])
                axis.set_yticks([])
        figure.savefig(output_dir / f"{input_name}.png", dpi=160)
        plt.close(figure)


def write_index(records: list[dict[str, Any]], run_dir: Path) -> None:
    rows = []
    for record in sorted(records, key=lambda item: (item["input"], item["model"])):
        figure = os.path.relpath(ROOT_DIR / record["figure"], run_dir)
        video_path = (
            os.path.relpath(ROOT_DIR / record["video"], run_dir)
            if record["video"]
            else None
        )
        video = (
            f'<a href="{html.escape(video_path)}">video</a>'
            if video_path is not None
            else ""
        )
        rows.append(
            "<tr>"
            f"<th>{html.escape(record['input'])}</th>"
            f"<th>{html.escape(record['model'])}</th>"
            f'<td><a href="{html.escape(figure)}"><img src="{html.escape(figure)}"></a></td>'
            f"<td>{video}</td>"
            "</tr>"
        )
    document = """<!doctype html><meta charset="utf-8"><title>JEPA comparison</title>
<style>
body{font:14px sans-serif;margin:24px} table{border-collapse:collapse}
th,td{border:1px solid #ccc;padding:8px;text-align:left}
img{width:720px;max-width:75vw;height:auto}
</style>
<h1>JEPA comparison</h1>
<table><tr><th>Input</th><th>Model</th><th>Visualization</th><th>Temporal</th></tr>
""" + "\n".join(rows) + "\n</table>\n"
    (run_dir / "index.html").write_text(document)


def dry_run(args: argparse.Namespace, pairs: list[tuple[ModelSpec, MediaSpec]]) -> None:
    for model, media in pairs:
        resolution = resolution_for(model, args.resolution)
        output = (
            args.output_root
            / f"{args.pairing}_{args.resolution}"
            / media.name
            / model.name
        )
        source = media.path
        if model.modality == "image" and media.modality == "video":
            source = (
                args.output_root
                / f"{args.pairing}_{args.resolution}"
                / "inputs"
                / f"{media.name}_frame0.jpg"
            )
        command, _, _ = extraction_command(
            model,
            weights=model.weights_path,
            input_path=source,
            output_dir=output,
            device=args.device,
            precision=args.precision,
            image_size=resolution,
            frames=args.frames,
            seed=0,
        )
        print(shlex.join(command))


def main() -> None:
    args = parse_args()
    if args.frames <= 0 or args.frames % 2:
        raise ValueError("--frames must be a positive multiple of 2.")
    pairs = selected_pairs(args)
    if args.dry_run:
        dry_run(args, pairs)
        return

    runtime = resolve_runtime(args.device, args.precision)
    device, precision = runtime.device.type, runtime.precision
    run_dir = args.output_root / f"{args.pairing}_{args.resolution}"
    first_frames, references = prepare_inputs(
        pairs, run_dir, args.resolution, force=args.force
    )
    records = [
        run_pair(
            model,
            media,
            first_frames=first_frames,
            references=references,
            run_dir=run_dir,
            resolution_value=args.resolution,
            frames=args.frames,
            device=device,
            precision=precision,
            force=args.force,
        )
        for model, media in pairs
    ]
    render_comparisons(records, run_dir / "comparisons")
    write_index(records, run_dir)

    models = model_specs()
    used_models = list(dict.fromkeys(record["model"] for record in records))
    manifest = {
        "pairing": args.pairing,
        "resolution": args.resolution,
        "frames": args.frames,
        "device": device,
        "precision": precision,
        "models": {
            name: {
                "checkpoint": file_provenance(models[name].checkpoint_path, include_hash=False),
                "source": model_code_provenance(models[name].model_code_path),
                "positional_geometry": models[name].positional_geometry,
            }
            for name in used_models
        },
        "inputs": {
            name: {
                "file": portable_path(spec.path),
                "source": spec.source,
            }
            for name, spec in media_specs().items()
            if name in {record["input"] for record in records}
        },
        "results": records,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"results={run_dir}")


if __name__ == "__main__":
    main()
