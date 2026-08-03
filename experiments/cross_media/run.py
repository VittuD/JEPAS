#!/usr/bin/env python3
"""Run every 2D/video JEPA model against every catalogued image and video."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shlex
import sys
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[2]
SMOKE_DIR = ROOT_DIR / "experiments" / "smoke"
VISUALIZER = ROOT_DIR / "experiments" / "visualization" / "visualize_embedding.py"
COMPARISON_RENDERER = ROOT_DIR / "experiments" / "cross_media" / "render_comparisons.py"
REFERENCE_PREPARER = ROOT_DIR / "experiments" / "cross_media" / "prepare_reference.py"
VALIDATOR = ROOT_DIR / "experiments" / "cross_media" / "validate_results.py"
sys.path.insert(0, str(ROOT_DIR))

from experiments.shared.catalog import (  # noqa: E402
    MediaSpec,
    ModelSpec,
    default_model_names,
    media_specs,
    model_specs,
)
from experiments.shared.device import add_runtime_arguments, resolve_runtime  # noqa: E402
from experiments.shared.provenance import (  # noqa: E402
    file_provenance,
    model_code_provenance,
    portable_path,
    run_command,
)


def parse_args(
    *,
    resolution_mode_default: str = "fixed",
    output_root_default: Path | None = None,
) -> argparse.Namespace:
    models = tuple(model_specs())
    inputs = tuple(media_specs())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument(
        "--resolution-mode",
        choices=("fixed", "native"),
        default=resolution_mode_default,
        help="Use --resolution for every model or each model's catalogued native resolution.",
    )
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument(
        "--models",
        nargs="+",
        default=default_model_names(),
        choices=models,
    )
    parser.add_argument("--inputs", nargs="+", default=inputs, choices=inputs)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            output_root_default
            or ROOT_DIR / "experiments" / "outputs" / "cross_media"
        ),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-visualization", action="store_true")
    parser.add_argument("--hash-checkpoints", action="store_true")
    add_runtime_arguments(parser)
    return parser.parse_args()


def common_checkpoint_cache(resolution: int) -> dict[str, dict[str, object]]:
    path = (
        ROOT_DIR
        / "experiments"
        / "outputs"
        / "common_resolution"
        / str(resolution)
        / "manifest.json"
    )
    if not path.exists():
        return {}
    manifest = json.loads(path.read_text())
    return {
        str(record["name"]): record["checkpoint"]
        for record in manifest.get("models", [])
        if isinstance(record.get("checkpoint"), dict)
    }


def prepare_video_frames(
    selected_media: list[MediaSpec],
    *,
    run_dir: Path,
    force: bool,
) -> tuple[dict[str, Path], list[dict[str, object]]]:
    converted: dict[str, Path] = {}
    records: list[dict[str, object]] = []
    conversion_dir = run_dir / "converted_inputs"
    for media in selected_media:
        if media.modality != "video":
            continue
        frame = conversion_dir / f"{media.name}_first_frame.jpg"
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            portable_path(media.path),
            "-map",
            "0:v:0",
            "-frames:v",
            "1",
            portable_path(frame),
        ]
        if force or not frame.exists():
            frame.parent.mkdir(parents=True, exist_ok=True)
            run_command(
                command,
                cwd=ROOT_DIR,
                log_path=conversion_dir / f"{media.name}.log",
            )
            status = "computed"
        else:
            status = "reused"
        converted[media.name] = frame
        records.append(
            {
                "source_input": media.name,
                "operation": "video first frame",
                "command": shlex.join(command),
                "status": status,
                "output": file_provenance(frame, include_hash=True),
            }
        )
    return converted, records


def prepare_reference_images(
    selected_media: list[MediaSpec],
    *,
    first_frames: dict[str, Path],
    resolution: int,
    run_dir: Path,
    force: bool,
) -> tuple[dict[str, Path], list[dict[str, object]]]:
    references: dict[str, Path] = {}
    records: list[dict[str, object]] = []
    conversion_dir = run_dir / "converted_inputs"
    for media in selected_media:
        source = first_frames[media.name] if media.modality == "video" else media.path
        reference = conversion_dir / f"{media.name}_reference_{resolution}.jpg"
        command = [
            portable_path(Path(sys.executable)),
            portable_path(REFERENCE_PREPARER),
            portable_path(source),
            portable_path(reference),
            "--resolution",
            str(resolution),
        ]
        if force or not reference.exists():
            run_command(
                command,
                cwd=ROOT_DIR,
                log_path=conversion_dir / f"{media.name}_reference_{resolution}.log",
            )
            status = "computed"
        else:
            status = "reused"
        references[media.name] = reference
        records.append(
            {
                "source_input": media.name,
                "operation": "canonical square frame-0 reference",
                "command": shlex.join(command),
                "status": status,
                "output": file_provenance(reference, include_hash=True),
            }
        )
    return references, records


def adapted_input(
    model: ModelSpec,
    media: MediaSpec,
    first_frames: dict[str, Path],
    frames: int,
) -> tuple[Path, str, str]:
    if model.modality == "image" and media.modality == "video":
        return first_frames[media.name], "image", "video_to_image_first_frame"
    if model.modality == "video" and media.modality == "image":
        return media.path, "image", f"image_to_video_repeat_{frames}_frames"
    return media.path, media.modality, "native_modality"


def run_pair(
    model: ModelSpec,
    media: MediaSpec,
    *,
    effective_input: Path,
    reference_image: Path,
    argument_kind: str,
    adaptation: str,
    resolution: int,
    frames: int,
    run_dir: Path,
    force: bool,
    force_visualization: bool,
    hash_checkpoints: bool,
    checkpoint_cache: dict[str, dict[str, object]],
    device: str,
    precision: str,
) -> dict[str, object]:
    pair_dir = run_dir / media.name / model.name
    embedding = pair_dir / "tokens.npy"
    preview = pair_dir / ("input.jpg" if model.modality == "image" else "input_frame.jpg")
    visualization = pair_dir / "visualization"
    figure = visualization / "embedding_visualization.png"
    animation = visualization / "embedding_visualization.mp4"
    result_path = pair_dir / "result.json"
    previous_result = json.loads(result_path.read_text()) if result_path.exists() else {}
    previous_runtime = previous_result.get("runtime")
    requested_runtime = {"device": device, "precision": precision}
    if (
        embedding.exists()
        and not force
        and previous_runtime != requested_runtime
    ):
        raise ValueError(
            f"{embedding} was not produced with runtime {requested_runtime}. "
            "Pass --force or use a different --output-root."
        )

    spatial_grid = resolution // model.patch_size
    covered_pixels = spatial_grid * model.patch_size
    if resolution % model.patch_size:
        print(
            f"[{media.name}/{model.name}] {resolution}px with patch {model.patch_size} "
            f"covers {covered_pixels}px on a {spatial_grid}x{spatial_grid} grid.",
            flush=True,
        )

    extraction_command = [
        portable_path(Path(sys.executable)),
        portable_path(SMOKE_DIR / model.extractor[0]),
        *model.extractor[1:],
        "--image-size",
        str(resolution),
        f"--{argument_kind}",
        portable_path(effective_input),
        "--output-embedding",
        portable_path(embedding),
        "--output-preprocessed-image",
        portable_path(preview),
        "--device",
        device,
        "--precision",
        precision,
    ]
    if model.modality == "video":
        extraction_command.extend(("--frames", str(frames)))

    if force or not embedding.exists():
        run_command(extraction_command, cwd=ROOT_DIR, log_path=pair_dir / "extract.log")
        extraction_status = "computed"
    else:
        print(f"[{media.name}/{model.name}] embedding exists, skipping", flush=True)
        extraction_status = "reused"

    if model.modality == "video":
        grid_shape = (frames // 2, spatial_grid, spatial_grid)
        slice_index = 0
    else:
        grid_shape = (spatial_grid, spatial_grid)
        slice_index = None

    visualization_command = [
        portable_path(Path(sys.executable)),
        portable_path(VISUALIZER),
        portable_path(embedding),
        "--out-dir",
        portable_path(visualization),
        "--grid-shape",
        "x".join(str(value) for value in grid_shape),
        "--image",
        portable_path(reference_image),
        "--title",
        (
            f"{model.name} on {media.name} | input={resolution}x{resolution} "
            f"| grid={'x'.join(map(str, grid_shape))}"
        ),
    ]
    if slice_index is not None:
        visualization_command.extend(
            (
                "--slice-index",
                str(slice_index),
                "--animate",
                "--reference-size",
                str(resolution),
                "--tubelet-size",
                "2",
            )
        )
        if argument_kind == "video":
            visualization_command.extend(
                ("--reference-video", portable_path(effective_input))
            )

    visualization_complete = figure.exists() and (
        model.modality != "video" or animation.exists()
    )
    if force or force_visualization or not visualization_complete:
        run_command(
            visualization_command,
            cwd=ROOT_DIR,
            log_path=pair_dir / "visualize.log",
        )
        visualization_status = "computed"
    else:
        print(f"[{media.name}/{model.name}] visualization exists, skipping", flush=True)
        visualization_status = "reused"

    previous_checkpoint = previous_result.get("checkpoint")
    if not isinstance(previous_checkpoint, dict):
        previous_checkpoint = checkpoint_cache.get(model.name)
    record = {
        "name": f"{media.name}/{model.name}",
        "model": model.name,
        "model_modality": model.modality,
        "input_name": media.name,
        "input_modality": media.modality,
        "native_resolution": model.native_resolution,
        "input_resolution": resolution,
        "patch_size": model.patch_size,
        "spatial_grid": [spatial_grid, spatial_grid],
        "covered_pixels": [covered_pixels, covered_pixels],
        "temporal_grid": frames // 2 if model.modality == "video" else None,
        "adaptation": adaptation,
        "source_input": file_provenance(media.path, include_hash=True),
        "effective_input": file_provenance(effective_input, include_hash=True),
        "comparison_reference": file_provenance(reference_image, include_hash=True),
        "input_source": media.source,
        "checkpoint": file_provenance(
            model.checkpoint_path,
            include_hash=hash_checkpoints,
            previous=previous_checkpoint,
        ),
        "checkpoint_source": model.checkpoint_source,
        "model_code": model_code_provenance(model.model_code_path),
        "model_configuration": (
            file_provenance(model.configuration_path, include_hash=True)
            if model.configuration_path is not None
            else None
        ),
        "experiment_code": {
            "extractor": file_provenance(SMOKE_DIR / model.extractor[0], include_hash=True),
            "visualizer": file_provenance(VISUALIZER, include_hash=True),
        },
        "positional_geometry": model.positional_geometry,
        "embedding": portable_path(embedding),
        "embedding_shape": list(np.load(embedding, mmap_mode="r").shape),
        "figure": portable_path(figure),
        "animation": portable_path(animation) if animation.exists() else None,
        "commands": {
            "working_directory": ".",
            "extract": shlex.join(extraction_command),
            "visualize": shlex.join(visualization_command),
        },
        "execution": {
            "extraction": extraction_status,
            "visualization": visualization_status,
        },
        "runtime": {
            **requested_runtime,
        },
        "artifacts": {
            "embedding": file_provenance(embedding, include_hash=True),
            "preview": file_provenance(reference_image, include_hash=True),
            "model_preprocessed_preview": file_provenance(preview, include_hash=True),
            "figure": file_provenance(figure, include_hash=True),
            "animation": (
                file_provenance(animation, include_hash=True)
                if animation.exists()
                else None
            ),
            "visualization_manifest": portable_path(visualization / "manifest.json"),
            "visualization_summary": portable_path(visualization / "summary.json"),
        },
    }
    pair_dir.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(record, indent=2) + "\n")
    return record


def main(
    *,
    resolution_mode_default: str = "fixed",
    output_root_default: Path | None = None,
    runner_path: Path | None = None,
) -> None:
    args = parse_args(
        resolution_mode_default=resolution_mode_default,
        output_root_default=output_root_default,
    )
    if args.resolution <= 0:
        raise ValueError("--resolution must be positive.")
    if args.frames <= 0 or args.frames % 2:
        raise ValueError("--frames must be a positive multiple of 2.")
    runtime = resolve_runtime(args.device, args.precision)
    device = runtime.device.type
    precision = runtime.precision

    models = model_specs()
    inputs = media_specs()
    selected_models = [models[name] for name in dict.fromkeys(args.models)]
    selected_media = [inputs[name] for name in dict.fromkeys(args.inputs)]
    resolution_by_model = {
        model.name: (
            model.native_resolution
            if args.resolution_mode == "native"
            else args.resolution
        )
        for model in selected_models
    }
    for model in selected_models:
        if not model.checkpoint_path.exists():
            raise FileNotFoundError(model.checkpoint_path)
    for media in selected_media:
        if not media.path.exists():
            raise FileNotFoundError(media.path)

    run_key = "native" if args.resolution_mode == "native" else str(args.resolution)
    run_dir = args.output_root / run_key
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    existing_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    existing_mode = str(
        existing_manifest.get(
            "resolution_mode",
            "native" if existing_manifest.get("resolution") == "native" else "fixed",
        )
    )
    if existing_manifest and existing_mode != args.resolution_mode and not args.force:
        raise ValueError(
            f"{manifest_path} uses resolution mode {existing_mode}, not "
            f"{args.resolution_mode}. Use a different --output-root or pass --force."
        )
    existing_frames = int(existing_manifest.get("frames", args.frames))
    if existing_frames != args.frames and not args.force:
        raise ValueError(
            f"{manifest_path} was generated with {existing_frames} frames, not {args.frames}. "
            "Use a different --output-root or pass --force."
        )

    first_frames, new_conversions = prepare_video_frames(
        selected_media,
        run_dir=run_dir,
        force=args.force,
    )
    references_by_resolution: dict[int, dict[str, Path]] = {}
    for resolution in sorted(set(resolution_by_model.values())):
        reference_images, reference_conversions = prepare_reference_images(
            selected_media,
            first_frames=first_frames,
            resolution=resolution,
            run_dir=run_dir,
            force=args.force,
        )
        references_by_resolution[resolution] = reference_images
        new_conversions.extend(reference_conversions)

    checkpoint_cache: dict[str, dict[str, object]] = {}
    for model in selected_models:
        cached = common_checkpoint_cache(resolution_by_model[model.name]).get(model.name)
        if cached is not None:
            checkpoint_cache[model.name] = cached

    new_records: list[dict[str, object]] = []
    for model in selected_models:
        model_resolution = resolution_by_model[model.name]
        for media in selected_media:
            effective_input, argument_kind, adaptation = adapted_input(
                model,
                media,
                first_frames,
                args.frames,
            )
            new_records.append(
                run_pair(
                    model,
                    media,
                    effective_input=effective_input,
                    reference_image=references_by_resolution[model_resolution][media.name],
                    argument_kind=argument_kind,
                    adaptation=adaptation,
                    resolution=model_resolution,
                    frames=args.frames,
                    run_dir=run_dir,
                    force=args.force,
                    force_visualization=args.force_visualization,
                    hash_checkpoints=args.hash_checkpoints,
                    checkpoint_cache=checkpoint_cache,
                    device=device,
                    precision=precision,
                )
            )

    records_by_pair = {
        (str(record["input_name"]), str(record["model"])): record
        for record in existing_manifest.get("results", [])
    }
    records_by_pair.update(
        {
            (str(record["input_name"]), str(record["model"])): record
            for record in new_records
        }
    )
    selected_model_names = {pair[1] for pair in records_by_pair}
    selected_input_names = {pair[0] for pair in records_by_pair}
    model_names = [name for name in models if name in selected_model_names]
    input_names = [name for name in inputs if name in selected_input_names]
    input_order = {name: index for index, name in enumerate(input_names)}
    model_order = {name: index for index, name in enumerate(model_names)}
    records = list(records_by_pair.values())
    records.sort(
        key=lambda record: (
            input_order[str(record["input_name"])],
            model_order[str(record["model"])],
        )
    )

    conversions_by_key = {
        (
            str(record["source_input"]),
            str(record["operation"]),
            str(record.get("output", {}).get("path", "")),
        ): record
        for record in existing_manifest.get("conversions", [])
    }
    conversions_by_key.update(
        {
            (
                str(record["source_input"]),
                str(record["operation"]),
                str(record.get("output", {}).get("path", "")),
            ): record
            for record in new_conversions
        }
    )
    conversion_order = {name: index for index, name in enumerate(inputs)}
    conversions = sorted(
        conversions_by_key.values(),
        key=lambda record: (
            conversion_order[str(record["source_input"])],
            str(record["operation"]),
        ),
    )
    comparison_command = [
        portable_path(Path(sys.executable)),
        portable_path(COMPARISON_RENDERER),
        portable_path(manifest_path),
        "--out-dir",
        portable_path(run_dir / "comparisons"),
    ]
    validation_command = [
        portable_path(Path(sys.executable)),
        portable_path(VALIDATOR),
        portable_path(manifest_path),
    ]
    package_names = (
        "torch",
        "torchvision",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "timm",
        "transformers",
    )
    manifest_resolution: int | str = (
        "native" if args.resolution_mode == "native" else args.resolution
    )
    if args.resolution_mode == "native":
        comparison_note = (
            "Each input is square-cropped and resized to the selected model's native "
            "spatial resolution. Image models receive frame 0 from videos; video models "
            "repeat still images across the configured frame count. Token-grid sizes, "
            "temporal training, and domains remain model-specific."
        )
    else:
        comparison_note = (
            f"All inputs are resized to {args.resolution} square pixels. Image models "
            "receive frame 0 from videos; video models repeat still images across the "
            "configured frame count. Patch sizes, temporal training, and domains remain "
            "model-specific."
        )
    manifest = {
        "resolution": manifest_resolution,
        "resolution_mode": args.resolution_mode,
        "model_resolutions": {
            name: (
                models[name].native_resolution
                if args.resolution_mode == "native"
                else args.resolution
            )
            for name in model_names
        },
        "frames": args.frames,
        "invocation": shlex.join([portable_path(Path(sys.executable)), *sys.argv]),
        "working_directory": ".",
        "runner": file_provenance(
            (runner_path or Path(__file__)).resolve(),
            include_hash=True,
        ),
        "orchestration": file_provenance(Path(__file__).resolve(), include_hash=True),
        "models": model_names,
        "inputs": input_names,
        "conversions": conversions,
        "results": records,
        "environment": {
            "python_executable": portable_path(Path(sys.executable)),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in package_names
            },
            "thread_settings": {
                name: os.environ.get(name)
                for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")
            },
            "inference": {
                "device": device,
                "precision": precision,
            },
            "venv_blueprint": portable_path(
                ROOT_DIR / "scripts" / "create_inference_venv.sh"
            ),
            "requirements": portable_path(
                ROOT_DIR
                / "requirements"
                / (
                    "vjepa2_inference_cuda.txt"
                    if device == "cuda"
                    else "vjepa2_inference.txt"
                )
            ),
        },
        "comparison": {
            "command": shlex.join(comparison_command),
            "renderer": file_provenance(COMPARISON_RENDERER, include_hash=True),
            "output_directory": portable_path(run_dir / "comparisons"),
        },
        "validation": {
            "command": shlex.join(validation_command),
            "validator": file_provenance(VALIDATOR, include_hash=True),
            "log": portable_path(run_dir / "validation.log"),
        },
        "comparison_note": comparison_note,
        "excluded": {
            "neurojepa": "3D volume model; image/video adaptation is not semantically comparable.",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    run_command(
        comparison_command,
        cwd=ROOT_DIR,
        log_path=run_dir / "comparisons" / "render.log",
    )
    run_command(
        validation_command,
        cwd=ROOT_DIR,
        log_path=run_dir / "validation.log",
    )

    command_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'cd "$(git rev-parse --show-toplevel)"',
    ]
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        value = os.environ.get(name)
        if value is not None:
            command_lines.append(f"export {name}={shlex.quote(value)}")
    command_lines.append("")
    for conversion in conversions:
        command_lines.extend(
            (
                f"# {conversion['source_input']}: {conversion['operation']}",
                str(conversion["command"]),
                "",
            )
        )
    for record in records:
        command_lines.extend(
            (
                f"# {record['input_name']} -> {record['model']}",
                str(record["commands"]["extract"]),
                str(record["commands"]["visualize"]),
                "",
            )
        )
    command_lines.extend(
        (
            "# Render comparison matrices",
            shlex.join(comparison_command),
            "",
            "# Validate shared references and temporal slices",
            shlex.join(validation_command),
            "",
        )
    )
    commands_path = run_dir / "commands.sh"
    commands_path.write_text("\n".join(command_lines))
    commands_path.chmod(0o755)
    print(f"[cross_media] wrote {manifest_path}", flush=True)
    print(f"[cross_media] wrote {commands_path}", flush=True)


if __name__ == "__main__":
    main()
