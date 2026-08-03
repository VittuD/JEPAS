#!/usr/bin/env python3
"""Compare JEPA embeddings across matched or all catalogued media."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT_DIR / "experiments" / "outputs"
sys.path.insert(0, str(ROOT_DIR))

from experiments.artifacts import (  # noqa: E402
    EMBEDDINGS_NAME,
    FIGURE_NAME,
    SCHEMA_NAME,
    VIDEO_NAME,
    cleanup_embeddings,
    load_visualization_metadata,
    valid_visualization,
)
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
    parser.add_argument("--keep-embeddings", action="store_true")
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
    temporary_dir: Path,
) -> dict[str, Path]:
    first_frames: dict[str, Path] = {}
    for model, media in pairs:
        if model.modality == "image" and media.modality == "video":
            first_frames.setdefault(media.name, temporary_dir / f"{media.name}_frame0.jpg")
    return first_frames


def effective_input(
    model: ModelSpec,
    media: MediaSpec,
    first_frames: dict[str, Path],
) -> tuple[Path, str]:
    if model.modality == "image" and media.modality == "video":
        frame = first_frames[media.name]
        if not frame.exists():
            first_video_frame(media.path, frame)
        return frame, "video_frame0_to_image"
    if model.modality == "video" and media.modality == "image":
        return media.path, "image_repeated_as_video"
    return media.path, "native"


def visualization_command(
    *,
    embeddings: Path,
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
        portable_path(embeddings),
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
    run_dir: Path,
    resolution_value: str,
    frames: int,
    device: str,
    precision: str,
    force: bool,
    keep_embeddings: bool,
) -> dict[str, Any]:
    resolution = resolution_for(model, resolution_value)
    pair_dir = run_dir / media.name / model.name
    if model.modality == "image" and media.modality == "video":
        adaptation = "video_frame0_to_image"
    elif model.modality == "video" and media.modality == "image":
        adaptation = "image_repeated_as_video"
    else:
        adaptation = "native"
    requires_video = model.modality == "video"
    if not force and valid_visualization(pair_dir, require_video=requires_video):
        if not keep_embeddings:
            cleanup_embeddings(pair_dir, require_video=requires_video)
        metadata = load_visualization_metadata(pair_dir)
        assert metadata is not None
        extraction = metadata["extraction"]
        return {
            "model": model.name,
            "input": media.name,
            "input_modality": media.modality,
            "model_modality": model.modality,
            "adaptation": adaptation,
            "resolution": resolution,
            "patch_size": model.patch_size,
            "spatial_grid": [resolution // model.patch_size] * 2,
            "temporal_grid": frames // 2 if requires_video else None,
            "embeddings": portable_path(pair_dir / EMBEDDINGS_NAME)
            if (pair_dir / EMBEDDINGS_NAME).exists()
            else None,
            "token_shape": extraction["token_shape"],
            "token_dtype": extraction["token_dtype"],
            "pooled_shape": extraction["pooled_shape"],
            "pooled_dtype": extraction["pooled_dtype"],
            "preview": portable_path(pair_dir / "input.jpg"),
            "figure": portable_path(pair_dir / FIGURE_NAME),
            "video": portable_path(pair_dir / VIDEO_NAME) if requires_video else None,
            "status": "reused",
        }
    source, adaptation = effective_input(model, media, first_frames)
    extraction = extract_one(
        model,
        source,
        pair_dir,
        device=device,
        precision=precision,
        image_size=resolution,
        frames=frames,
        metadata_input=media.path,
        force=force,
    )
    command = visualization_command(
        embeddings=Path(extraction["embeddings"]),
        output_dir=pair_dir,
        reference=pair_dir / "input.jpg",
        reference_video=source if model.modality == "video" and media.modality == "video" else None,
        model=model,
        media=media,
        resolution=resolution,
        frames=frames,
    )
    figure = pair_dir / FIGURE_NAME
    video = pair_dir / VIDEO_NAME
    if force or not valid_visualization(pair_dir, require_video=requires_video):
        run_command(
            command,
            cwd=ROOT_DIR,
            log_path=pair_dir / "run.log",
            log_mode="a",
        )
    if not valid_visualization(pair_dir, require_video=requires_video):
        raise RuntimeError(f"Visualization did not produce valid artifacts in {pair_dir}.")
    if not keep_embeddings:
        cleanup_embeddings(pair_dir, require_video=requires_video)
    return {
        "model": model.name,
        "input": media.name,
        "input_modality": media.modality,
        "model_modality": model.modality,
        "adaptation": adaptation,
        "resolution": resolution,
        "patch_size": model.patch_size,
        "spatial_grid": [resolution // model.patch_size] * 2,
        "temporal_grid": frames // 2 if requires_video else None,
        "embeddings": extraction["embeddings"] if keep_embeddings else None,
        "token_shape": extraction["token_shape"],
        "token_dtype": extraction["token_dtype"],
        "pooled_shape": extraction["pooled_shape"],
        "pooled_dtype": extraction["pooled_dtype"],
        "preview": extraction["preview"],
        "figure": portable_path(figure),
        "video": portable_path(video) if video.exists() else None,
        "extract_command": extraction["command"],
        "visualize_command": shlex.join(command),
        "status": "computed",
    }


def dry_run(args: argparse.Namespace, pairs: list[tuple[ModelSpec, MediaSpec]]) -> None:
    for model, media in pairs:
        resolution = resolution_for(model, args.resolution)
        output = (
            args.output_root
            / f"comparison_{args.pairing}_{args.resolution}"
            / media.name
            / model.name
        )
        source = media.path
        if model.modality == "image" and media.modality == "video":
            source = (
                args.output_root
                / f"comparison_{args.pairing}_{args.resolution}"
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
    run_dir = args.output_root / f"comparison_{args.pairing}_{args.resolution}"
    with tempfile.TemporaryDirectory(prefix="jepas_compare_") as temporary:
        first_frames = prepare_inputs(pairs, Path(temporary))
        records = [
            run_pair(
                model,
                media,
                first_frames=first_frames,
                run_dir=run_dir,
                resolution_value=args.resolution,
                frames=args.frames,
                device=device,
                precision=precision,
                force=args.force,
                keep_embeddings=args.keep_embeddings,
            )
            for model, media in pairs
        ]

    models = model_specs()
    used_models = list(dict.fromkeys(record["model"] for record in records))
    manifest = {
        "schema": SCHEMA_NAME,
        "schema_version": 1,
        "experiment": run_dir.name,
        "pairing": args.pairing,
        "resolution": args.resolution,
        "frames": args.frames,
        "device": device,
        "precision": precision,
        "keep_embeddings": args.keep_embeddings,
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
