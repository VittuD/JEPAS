#!/usr/bin/env python3
"""Extract and visualize model tokens at one shared square input resolution."""

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
COMPARISON_RENDERER = ROOT_DIR / "experiments" / "common_resolution" / "render_comparison.py"
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

def parse_args() -> argparse.Namespace:
    models = tuple(model_specs())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument(
        "--models",
        nargs="+",
        default=default_model_names(),
        choices=models,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT_DIR / "experiments" / "outputs" / "common_resolution",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute existing embeddings and figures.",
    )
    parser.add_argument(
        "--force-visualization",
        action="store_true",
        help="Recompute figures without recomputing embeddings.",
    )
    parser.add_argument(
        "--hash-checkpoints",
        action="store_true",
        help="Record checkpoint SHA-256 hashes. This reads every selected checkpoint in full.",
    )
    add_runtime_arguments(parser)
    return parser.parse_args()


def run_model(
    spec: ModelSpec,
    media: MediaSpec,
    *,
    resolution: int,
    frames: int,
    output_root: Path,
    force: bool,
    force_visualization: bool,
    hash_checkpoints: bool,
    device: str,
    precision: str,
) -> dict[str, object]:
    if not media.path.exists():
        raise FileNotFoundError(media.path)
    if not spec.checkpoint_path.exists():
        raise FileNotFoundError(spec.checkpoint_path)

    model_dir = output_root / str(resolution) / spec.name
    embedding = model_dir / "tokens.npy"
    preview = model_dir / ("input.jpg" if spec.modality == "image" else "input_frame.jpg")
    visualization = model_dir / "visualization"
    figure = visualization / "embedding_visualization.png"
    animation = visualization / "embedding_visualization.mp4"
    result_path = model_dir / "result.json"
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

    spatial_grid = resolution // spec.patch_size
    covered_pixels = spatial_grid * spec.patch_size
    if resolution % spec.patch_size:
        print(
            f"[{spec.name}] {resolution}px is not divisible by patch size {spec.patch_size}; "
            f"the {spatial_grid}x{spatial_grid} patch grid covers {covered_pixels}px.",
            flush=True,
        )

    extraction_command = [
        portable_path(Path(sys.executable)),
        portable_path(SMOKE_DIR / spec.extractor[0]),
        *spec.extractor[1:],
        "--image-size",
        str(resolution),
        f"--{spec.modality}",
        portable_path(media.path),
        "--output-embedding",
        portable_path(embedding),
        "--output-preprocessed-image",
        portable_path(preview),
        "--device",
        device,
        "--precision",
        precision,
    ]
    if spec.modality == "video":
        extraction_command.extend(("--frames", str(frames)))

    if force or not embedding.exists():
        run_command(extraction_command, cwd=ROOT_DIR, log_path=model_dir / "extract.log")
        extraction_status = "computed"
    else:
        print(f"[{spec.name}] embedding exists, skipping: {embedding}", flush=True)
        extraction_status = "reused"

    if spec.modality == "video":
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
        portable_path(preview),
        "--title",
        f"{spec.name} | input={resolution}x{resolution} | grid={'x'.join(map(str, grid_shape))}",
    ]
    if slice_index is not None:
        visualization_command.extend(
            (
                "--slice-index",
                str(slice_index),
                "--animate",
                "--reference-video",
                portable_path(media.path),
                "--reference-size",
                str(resolution),
                "--tubelet-size",
                "2",
            )
        )

    visualization_complete = figure.exists() and (
        spec.modality != "video" or animation.exists()
    )
    if force or force_visualization or not visualization_complete:
        run_command(visualization_command, cwd=ROOT_DIR, log_path=model_dir / "visualize.log")
        visualization_status = "computed"
    else:
        print(f"[{spec.name}] visualization exists, skipping: {figure}", flush=True)
        visualization_status = "reused"

    previous_checkpoint = previous_result.get("checkpoint")
    record = {
        "name": spec.name,
        "native_resolution": spec.native_resolution,
        "input_resolution": resolution,
        "patch_size": spec.patch_size,
        "spatial_grid": [spatial_grid, spatial_grid],
        "covered_pixels": [covered_pixels, covered_pixels],
        "temporal_grid": frames // 2 if spec.modality == "video" else None,
        "input": file_provenance(media.path, include_hash=True),
        "input_name": media.name,
        "input_source": media.source,
        "checkpoint": file_provenance(
            spec.checkpoint_path,
            include_hash=hash_checkpoints,
            previous=previous_checkpoint if isinstance(previous_checkpoint, dict) else None,
        ),
        "checkpoint_source": spec.checkpoint_source,
        "model_code": model_code_provenance(spec.model_code_path),
        "model_configuration": (
            file_provenance(spec.configuration_path, include_hash=True)
            if spec.configuration_path is not None
            else None
        ),
        "experiment_code": {
            "extractor": file_provenance(
                SMOKE_DIR / spec.extractor[0],
                include_hash=True,
            ),
            "visualizer": file_provenance(VISUALIZER, include_hash=True),
        },
        "positional_geometry": spec.positional_geometry,
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
            "preview": file_provenance(preview, include_hash=True),
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
    model_dir.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> None:
    args = parse_args()
    if args.resolution <= 0:
        raise ValueError("--resolution must be positive.")
    if args.frames <= 0 or args.frames % 2:
        raise ValueError("--frames must be a positive multiple of the tubelet size (2).")
    runtime = resolve_runtime(args.device, args.precision)
    device = runtime.device.type
    precision = runtime.precision

    run_dir = args.output_root / str(args.resolution)
    manifest_path = run_dir / "manifest.json"
    existing_records: dict[str, dict[str, object]] = {}
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text())
        existing_frames = int(existing_manifest.get("frames", args.frames))
        if existing_frames != args.frames and not args.force:
            raise ValueError(
                f"{manifest_path} was generated with {existing_frames} frames, not {args.frames}. "
                "Use a different --output-root or pass --force."
            )
        existing_records = {
            str(record["name"]): record
            for record in existing_manifest.get("models", [])
        }

    specs = model_specs()
    media = media_specs()
    matched_inputs = {
        "ijepa": "dog",
        "radjepa": "chest_xray",
        "vjepa2": "diving48",
        "vjepa2-1": "diving48",
        "echojepa": "echonet",
        "ijepa-lite-affinity-novelty": "chest_xray",
    }
    selected = list(dict.fromkeys(args.models))
    new_records = [
        run_model(
            specs[name],
            media[matched_inputs[name]],
            resolution=args.resolution,
            frames=args.frames,
            output_root=args.output_root,
            force=args.force,
            force_visualization=args.force_visualization,
            hash_checkpoints=args.hash_checkpoints,
            device=device,
            precision=precision,
        )
        for name in selected
    ]
    existing_records.update({str(record["name"]): record for record in new_records})
    records = [
        existing_records[name]
        for name in specs
        if name in existing_records
    ]

    run_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = run_dir / "comparison.png"
    comparison_command = [
        portable_path(Path(sys.executable)),
        portable_path(COMPARISON_RENDERER),
        portable_path(manifest_path),
        "--output",
        portable_path(comparison_path),
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
    manifest = {
        "resolution": args.resolution,
        "frames": args.frames,
        "invocation": shlex.join([portable_path(Path(sys.executable)), *sys.argv]),
        "working_directory": ".",
        "runner": file_provenance(Path(__file__).resolve(), include_hash=True),
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
        "models": records,
        "comparison": {
            "command": shlex.join(comparison_command),
            "renderer": file_provenance(COMPARISON_RENDERER, include_hash=True),
            "output": portable_path(comparison_path),
        },
        "excluded": {
            "neurojepa": "3D volume model; no directly comparable 2D input resolution.",
        },
        "comparison_note": (
            "Input pixels are matched. Patch sizes and training domains remain model-specific, "
            "so token counts and semantics are not controlled."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[common_resolution] wrote {manifest_path}", flush=True)
    run_command(comparison_command, cwd=ROOT_DIR, log_path=run_dir / "comparison.log")

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
    for record in records:
        command_lines.extend(
            (
                f"# {record['name']}",
                str(record["commands"]["extract"]),
                str(record["commands"]["visualize"]),
                "",
            )
        )
    command_lines.extend(("# Combined comparison", shlex.join(comparison_command), ""))
    commands_path = run_dir / "commands.sh"
    commands_path.write_text("\n".join(command_lines))
    commands_path.chmod(0o755)
    print(f"[common_resolution] wrote {commands_path}", flush=True)


if __name__ == "__main__":
    main()
