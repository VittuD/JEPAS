#!/usr/bin/env python3
"""Extract non-pooled and mean-pooled embeddings with a source-backed model."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from experiments.catalog import (  # noqa: E402
    MODEL_ALIASES,
    ModelSpec,
    canonical_model_name,
    model_specs,
)
from experiments.device import add_runtime_arguments, resolve_runtime  # noqa: E402
from experiments.provenance import (  # noqa: E402
    file_provenance,
    model_code_provenance,
    portable_path,
    run_command,
)


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
VOLUME_SUFFIXES = {".npy", ".nii", ".nii.gz"}


def path_kind(path: Path) -> str | None:
    name = path.name.lower()
    if name.endswith(".nii.gz") or path.suffix.lower() in VOLUME_SUFFIXES:
        return "volume"
    if path.suffix.lower() in IMAGE_SUFFIXES:
        return "image"
    if path.suffix.lower() in VIDEO_SUFFIXES:
        return "video"
    return None


def input_stem(path: Path) -> str:
    name = path.name[:-7] if path.name.lower().endswith(".nii.gz") else path.stem
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-") or "input"


def expand_inputs(paths: list[Path], *, recursive: bool) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = raw.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_file():
            files.append(path)
            continue
        iterator = path.rglob("*") if recursive else path.iterdir()
        files.extend(candidate.resolve() for candidate in iterator if candidate.is_file())
    return sorted(dict.fromkeys(path for path in files if path_kind(path) is not None))


def checkpoint_for(spec: ModelSpec, weights: Path) -> Path:
    if weights.is_file():
        return weights
    if spec.name == "radjepa":
        return weights / "jepa_encoder.pth.tar"
    if spec.name == "neurojepa":
        return weights / "model.safetensors"
    return weights


def model_code_for(spec: ModelSpec, weights: Path) -> Path:
    if spec.name == "radjepa" and weights.is_dir():
        return weights / "modeling_radjepa.py"
    return spec.model_code_path


def extraction_command(
    spec: ModelSpec,
    *,
    weights: Path,
    input_path: Path,
    output_dir: Path,
    device: str,
    precision: str,
    image_size: int | None,
    frames: int,
    seed: int,
) -> tuple[list[str], Path, Path]:
    kind = path_kind(input_path)
    if kind not in spec.input_kinds:
        raise ValueError(
            f"{spec.name} accepts {', '.join(spec.input_kinds)}, not {kind or 'unknown'}."
        )

    tokens = output_dir / "tokens.npy"
    preview = output_dir / ("input.png" if kind == "volume" else "input.jpg")
    command = [
        portable_path(Path(sys.executable)),
        portable_path(spec.adapter),
        spec.weights_flag,
        portable_path(weights),
        *spec.adapter_args,
        "--device",
        device,
        "--precision",
        precision,
        "--seed",
        str(seed),
        "--output-embedding",
        portable_path(tokens),
    ]
    if kind == "volume":
        command.extend(
            (
                "--volume",
                portable_path(input_path),
                "--output-preprocessed-slice",
                portable_path(preview),
            )
        )
    else:
        command.extend(
            (
                f"--{kind}",
                portable_path(input_path),
                "--output-preprocessed-image",
                portable_path(preview),
            )
        )
        if image_size is not None:
            command.extend(("--image-size", str(image_size)))
        if "video" in spec.input_kinds:
            command.extend(("--frames", str(frames)))
    return command, tokens, preview


def extract_one(
    spec: ModelSpec,
    input_path: Path,
    output_dir: Path,
    *,
    weights: Path | None = None,
    device: str,
    precision: str,
    image_size: int | None,
    frames: int,
    seed: int = 0,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    weights = (weights or spec.weights_path).expanduser().resolve()
    input_path = input_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    command, tokens_path, preview_path = extraction_command(
        spec,
        weights=weights,
        input_path=input_path,
        output_dir=output_dir,
        device=device,
        precision=precision,
        image_size=image_size,
        frames=frames,
        seed=seed,
    )
    pooled_path = output_dir / "pooled.npy"

    if dry_run:
        return {"command": shlex.join(command), "status": "planned"}

    checkpoint = checkpoint_for(spec, weights)
    model_code = model_code_for(spec, weights)
    for required in (input_path, checkpoint, model_code):
        if not required.exists():
            raise FileNotFoundError(required)

    if force or not (tokens_path.exists() and pooled_path.exists() and preview_path.exists()):
        output_dir.mkdir(parents=True, exist_ok=True)
        run_command(command, cwd=ROOT_DIR, log_path=output_dir / "run.log")
        tokens = np.load(tokens_path)
        if tokens.ndim < 2:
            raise ValueError(f"Expected token embeddings, got {tokens.shape}.")
        np.save(pooled_path, tokens.astype(np.float32).mean(axis=-2, dtype=np.float32))
        status = "computed"
    else:
        status = "reused"

    tokens = np.load(tokens_path, mmap_mode="r")
    pooled = np.load(pooled_path, mmap_mode="r")
    return {
        "model": spec.name,
        "input_kind": path_kind(input_path),
        "input": portable_path(input_path),
        "checkpoint": portable_path(checkpoint),
        "output_dir": portable_path(output_dir),
        "tokens": portable_path(tokens_path),
        "pooled": portable_path(pooled_path),
        "preview": portable_path(preview_path),
        "token_shape": list(tokens.shape),
        "pooled_shape": list(pooled.shape),
        "command": shlex.join(command),
        "status": status,
    }


def parse_args() -> argparse.Namespace:
    choices = tuple(sorted((*model_specs(), *MODEL_ALIASES)))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=choices)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    add_runtime_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = model_specs()[canonical_model_name(args.model)]
    inputs = expand_inputs(args.input, recursive=args.recursive)
    if not inputs:
        raise ValueError("No supported inputs found.")
    if args.frames <= 0 or args.frames % 2:
        raise ValueError("--frames must be a positive multiple of 2.")

    if args.dry_run:
        device, precision = args.device, args.precision
    else:
        runtime = resolve_runtime(args.device, args.precision)
        device, precision = runtime.device.type, runtime.precision

    records = []
    for input_path in inputs:
        record = extract_one(
            spec,
            input_path,
            args.output_dir / input_stem(input_path),
            weights=args.weights,
            device=device,
            precision=precision,
            image_size=args.image_size,
            frames=args.frames,
            seed=args.seed,
            force=args.force,
            dry_run=args.dry_run,
        )
        records.append(record)
        print(record["command"] if args.dry_run else f"[{record['status']}] {record['output_dir']}")

    if args.dry_run:
        return

    weights = (args.weights or spec.weights_path).expanduser().resolve()
    checkpoint = checkpoint_for(spec, weights)
    manifest = {
        "model": spec.name,
        "device": device,
        "precision": precision,
        "checkpoint": file_provenance(checkpoint, include_hash=False),
        "model_code": model_code_provenance(model_code_for(spec, weights)),
        "results": records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
