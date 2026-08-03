#!/usr/bin/env python3
"""Extract token and pooled JEPA embeddings from files or directories."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[2]
SMOKE_DIR = ROOT_DIR / "experiments" / "smoke"
sys.path.insert(0, str(ROOT_DIR))

from experiments.shared.catalog import (  # noqa: E402
    IJEPA_LITE_AFFINITY_NOVELTY_CHECKPOINT,
    IJEPA_LITE_AFFINITY_NOVELTY_MANIFEST_PATH,
)
from experiments.shared.device import add_runtime_arguments, resolve_runtime  # noqa: E402
from experiments.shared.provenance import (  # noqa: E402
    file_provenance,
    model_code_provenance,
    portable_path,
    run_command,
)


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
VOLUME_SUFFIXES = {".npy", ".nii", ".nii.gz"}


@dataclass(frozen=True)
class BackendSpec:
    name: str
    script: Path
    input_kinds: tuple[str, ...]
    weights_flag: str
    default_weights: Path
    model_code: Path
    variant: str | None = None
    configuration: Path | None = None


def backend_specs() -> dict[str, BackendSpec]:
    vjepa_script = SMOKE_DIR / "vjepa2_random_image_smoke.py"
    vjepa_code = ROOT_DIR / "repos" / "vjepa2"
    return {
        "ijepa": BackendSpec(
            name="ijepa",
            script=SMOKE_DIR / "ijepa_random_image_smoke.py",
            input_kinds=("image",),
            weights_flag="--checkpoint",
            default_weights=ROOT_DIR / "weights" / "ijepa" / "IN1K-vit.h.14-300e.pth.tar",
            model_code=ROOT_DIR / "repos" / "ijepa",
        ),
        "radjepa": BackendSpec(
            name="radjepa",
            script=SMOKE_DIR / "radjepa_random_image_smoke.py",
            input_kinds=("image",),
            weights_flag="--model-dir",
            default_weights=ROOT_DIR / "weights" / "radjepa" / "hf-RadJEPA",
            model_code=ROOT_DIR / "weights" / "radjepa" / "hf-RadJEPA" / "modeling_radjepa.py",
        ),
        "vjepa2-vitl": BackendSpec(
            name="vjepa2-vitl",
            script=vjepa_script,
            input_kinds=("image", "video"),
            weights_flag="--checkpoint",
            default_weights=ROOT_DIR / "weights" / "vjepa2" / "vitl.pt",
            model_code=vjepa_code,
            variant="vjepa2-vitl",
        ),
        "vjepa2-vith": BackendSpec(
            name="vjepa2-vith",
            script=vjepa_script,
            input_kinds=("image", "video"),
            weights_flag="--checkpoint",
            default_weights=ROOT_DIR / "weights" / "vjepa2" / "vith.pt",
            model_code=vjepa_code,
            variant="vjepa2-vith",
        ),
        "vjepa2-1-vitb": BackendSpec(
            name="vjepa2-1-vitb",
            script=vjepa_script,
            input_kinds=("image", "video"),
            weights_flag="--checkpoint",
            default_weights=(
                ROOT_DIR / "weights" / "vjepa2" / "vjepa2_1_vitb_dist_vitG_384.pt"
            ),
            model_code=vjepa_code,
            variant="vjepa2-1-vitb",
        ),
        "vjepa2-1-vitg": BackendSpec(
            name="vjepa2-1-vitg",
            script=vjepa_script,
            input_kinds=("image", "video"),
            weights_flag="--checkpoint",
            default_weights=ROOT_DIR / "weights" / "vjepa2" / "vjepa2_1_vitg_384.pt",
            model_code=vjepa_code,
            variant="vjepa2-1-vitg",
        ),
        "echojepa": BackendSpec(
            name="echojepa",
            script=SMOKE_DIR / "echojepa_vjepa2_random_image_smoke.py",
            input_kinds=("image", "video"),
            weights_flag="--checkpoint",
            default_weights=(
                ROOT_DIR / "weights" / "echojepa" / "vitl-vmix22m-pt220-c55.pt"
            ),
            model_code=ROOT_DIR / "repos" / "EchoJEPA",
        ),
        "neurojepa": BackendSpec(
            name="neurojepa",
            script=SMOKE_DIR / "neurojepa_random_volume_smoke.py",
            input_kinds=("volume",),
            weights_flag="--model-dir",
            default_weights=ROOT_DIR / "weights" / "neurojepa" / "Neuro-JEPA",
            model_code=ROOT_DIR / "repos" / "Neuro-JEPA",
        ),
        "ijepa-lite-affinity-novelty": BackendSpec(
            name="ijepa-lite-affinity-novelty",
            script=SMOKE_DIR / "ijepa_lite_affinity_novelty_smoke.py",
            input_kinds=("image",),
            weights_flag="--checkpoint",
            default_weights=IJEPA_LITE_AFFINITY_NOVELTY_CHECKPOINT,
            model_code=ROOT_DIR / "repos" / "ijepa_lite",
            configuration=IJEPA_LITE_AFFINITY_NOVELTY_MANIFEST_PATH,
        ),
    }


MODEL_ALIASES = {
    "echojepa-vjepa2": "echojepa",
    "vjepa2": "vjepa2-vitl",
    "vjepa2-1": "vjepa2-1-vitb",
}


def parse_args() -> argparse.Namespace:
    models = tuple(sorted((*backend_specs(), *MODEL_ALIASES)))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=models)
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Optional checkpoint or local model directory override.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="One or more input files or directories.",
    )
    parser.add_argument(
        "--output-dir",
        "--output",
        dest="output_dir",
        type=Path,
        required=True,
        help="Directory containing one result folder per input.",
    )
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--recursive", action="store_true", help="Recurse into input directories.")
    parser.add_argument("--force", action="store_true", help="Replace existing mismatched results.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print extraction commands without checking device availability or running inference.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record a failed input and continue with the remaining inputs.",
    )
    parser.add_argument(
        "--hash-inputs",
        action="store_true",
        help="Include SHA-256 input hashes in result provenance.",
    )
    parser.add_argument(
        "--hash-weights",
        action="store_true",
        help="Include a SHA-256 checkpoint hash. This can be slow for large weights.",
    )
    parser.add_argument(
        "--save-preview",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save the square model input or representative volume slice.",
    )
    add_runtime_arguments(parser)
    return parser.parse_args()


def path_kind(path: Path) -> str | None:
    name = path.name.lower()
    if name.endswith(".nii.gz"):
        return "volume"
    if path.suffix.lower() in IMAGE_SUFFIXES:
        return "image"
    if path.suffix.lower() in VIDEO_SUFFIXES:
        return "video"
    if path.suffix.lower() in VOLUME_SUFFIXES:
        return "volume"
    return None


def expand_inputs(
    paths: list[Path],
    *,
    allowed_kinds: tuple[str, ...],
    recursive: bool,
) -> list[Path]:
    expanded: list[Path] = []
    for raw_path in paths:
        path = raw_path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_file():
            kind = path_kind(path)
            if kind not in allowed_kinds:
                raise ValueError(
                    f"{path} is a {kind or 'unsupported'} file; "
                    f"model accepts {', '.join(allowed_kinds)} inputs."
                )
            expanded.append(path)
            continue

        iterator = path.rglob("*") if recursive else path.iterdir()
        expanded.extend(
            candidate.resolve()
            for candidate in iterator
            if candidate.is_file() and path_kind(candidate) in allowed_kinds
        )

    unique = list(dict.fromkeys(expanded))
    if not unique:
        raise ValueError("No supported input files were found.")
    return sorted(unique)


def input_stem(path: Path) -> str:
    name = path.name[:-7] if path.name.lower().endswith(".nii.gz") else path.stem
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    return slug or "input"


def result_names(inputs: list[Path]) -> dict[Path, str]:
    stems: dict[str, list[Path]] = {}
    for path in inputs:
        stems.setdefault(input_stem(path), []).append(path)

    names: dict[Path, str] = {}
    for stem, paths in stems.items():
        for path in paths:
            if len(paths) == 1:
                names[path] = stem
            else:
                suffix = hashlib.sha256(portable_path(path).encode()).hexdigest()[:8]
                names[path] = f"{stem}-{suffix}"
    return names


def checkpoint_file(spec: BackendSpec, weights: Path) -> Path:
    if weights.is_file():
        return weights
    if spec.name == "radjepa":
        return weights / "jepa_encoder.pth.tar"
    if spec.name == "neurojepa":
        return weights / "model.safetensors"
    return weights


def model_code_path(spec: BackendSpec, weights: Path) -> Path:
    if spec.name == "radjepa" and weights.is_dir():
        return weights / "modeling_radjepa.py"
    return spec.model_code


def extraction_command(
    spec: BackendSpec,
    *,
    weights: Path,
    input_path: Path,
    output_dir: Path,
    device: str,
    precision: str,
    image_size: int | None,
    frames: int,
    seed: int,
    save_preview: bool,
) -> tuple[list[str], Path, Path | None]:
    tokens = output_dir / "tokens.npy"
    input_kind = path_kind(input_path)
    command = [
        portable_path(Path(sys.executable)),
        portable_path(spec.script),
        spec.weights_flag,
        portable_path(weights),
        "--device",
        device,
        "--precision",
        precision,
        "--seed",
        str(seed),
        "--output-embedding",
        portable_path(tokens),
    ]
    if spec.variant is not None:
        command.extend(("--variant", spec.variant))
    if input_kind == "volume":
        command.extend(("--volume", portable_path(input_path)))
        preview = output_dir / "preview.png" if save_preview else None
        if preview is not None:
            command.extend(("--output-preprocessed-slice", portable_path(preview)))
    else:
        command.extend((f"--{input_kind}", portable_path(input_path)))
        preview = output_dir / "preview.jpg" if save_preview else None
        if preview is not None:
            command.extend(("--output-preprocessed-image", portable_path(preview)))
        if image_size is not None:
            command.extend(("--image-size", str(image_size)))
        if "video" in spec.input_kinds:
            command.extend(("--frames", str(frames)))
    return command, tokens, preview


def run_signature(
    spec: BackendSpec,
    *,
    weights_file: Path,
    input_path: Path,
    input_kind: str,
    device: str,
    precision: str,
    image_size: int | None,
    frames: int,
    seed: int,
    save_preview: bool,
    orchestrator: dict[str, Any],
    extractor: dict[str, Any],
    model_code: dict[str, Any],
    model_configuration: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    input_stat = input_path.stat()
    weight_stat = weights_file.stat()
    payload = {
        "model": spec.name,
        "variant": spec.variant,
        "input": portable_path(input_path),
        "input_kind": input_kind,
        "input_size_bytes": input_stat.st_size,
        "input_mtime_ns": input_stat.st_mtime_ns,
        "weights": portable_path(weights_file),
        "weights_size_bytes": weight_stat.st_size,
        "weights_mtime_ns": weight_stat.st_mtime_ns,
        "device": device,
        "precision": precision,
        "image_size": image_size,
        "frames": frames if "video" in spec.input_kinds else None,
        "seed": seed,
        "save_preview": save_preview,
        "orchestrator": orchestrator,
        "extractor": extractor,
        "model_code": model_code,
        "model_configuration": model_configuration,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return digest, payload


def build_manifest(
    *,
    args: argparse.Namespace,
    spec: BackendSpec,
    weights: Path,
    device: str,
    precision: str,
    records: list[dict[str, Any]],
    errors: list[dict[str, str]],
) -> dict[str, Any]:
    package_names = (
        "torch",
        "torchvision",
        "numpy",
        "pillow",
        "safetensors",
        "timm",
        "transformers",
    )
    return {
        "schema_version": 1,
        "model": spec.name,
        "weights": portable_path(weights),
        "device": device,
        "precision": precision,
        "invocation": shlex.join([portable_path(Path(sys.executable)), *sys.argv]),
        "environment": {
            "python_executable": portable_path(Path(sys.executable)),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in package_names
            },
        },
        "records": records,
        "errors": errors,
        "configuration": {
            "image_size": args.image_size,
            "frames": args.frames,
            "seed": args.seed,
            "recursive": args.recursive,
            "save_preview": args.save_preview,
        },
    }


def main() -> None:
    args = parse_args()
    canonical_model = MODEL_ALIASES.get(args.model, args.model)
    spec = backend_specs()[canonical_model]
    weights = (args.weights or spec.default_weights).expanduser().resolve()
    weights_file = checkpoint_file(spec, weights)
    resolved_model_code = model_code_path(spec, weights)
    if not resolved_model_code.exists():
        raise FileNotFoundError(resolved_model_code)
    if not args.dry_run:
        if not weights.exists():
            raise FileNotFoundError(weights)
        if not weights_file.exists():
            raise FileNotFoundError(weights_file)
    if args.image_size is not None and args.image_size <= 0:
        raise ValueError("--image-size must be positive.")
    if args.frames <= 0 or args.frames % 2:
        raise ValueError("--frames must be a positive multiple of 2.")

    inputs = expand_inputs(
        args.input,
        allowed_kinds=spec.input_kinds,
        recursive=args.recursive,
    )
    names = result_names(inputs)

    if args.dry_run:
        device = args.device
        precision = args.precision
    else:
        runtime = resolve_runtime(args.device, args.precision)
        device = runtime.device.type
        precision = runtime.precision

    planned: list[tuple[Path, Path, list[str], Path, Path | None]] = []
    for input_path in inputs:
        output_dir = args.output_dir.expanduser().resolve() / names[input_path]
        command, tokens, preview = extraction_command(
            spec,
            weights=weights,
            input_path=input_path,
            output_dir=output_dir,
            device=device,
            precision=precision,
            image_size=args.image_size,
            frames=args.frames,
            seed=args.seed,
            save_preview=args.save_preview,
        )
        planned.append((input_path, output_dir, command, tokens, preview))

    if args.dry_run:
        for _input_path, _output_dir, command, _tokens, _preview in planned:
            print(shlex.join(command))
        print(f"planned_inputs={len(planned)}")
        return

    output_root = args.output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    orchestrator_provenance = file_provenance(Path(__file__).resolve(), include_hash=True)
    extractor_provenance = file_provenance(spec.script, include_hash=True)
    model_code_provenance_record = model_code_provenance(resolved_model_code)
    model_configuration_provenance = (
        file_provenance(spec.configuration, include_hash=True)
        if spec.configuration is not None
        else None
    )
    manifest_path = output_root / "manifest.json"
    existing_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    existing_model = existing_manifest.get("model")
    if existing_model not in (None, spec.name):
        raise ValueError(
            f"{manifest_path} belongs to model {existing_model!r}, not {spec.name!r}. "
            "Use a separate output directory."
        )
    records_by_input = {
        str(record["input"]["path"]): record
        for record in existing_manifest.get("records", [])
    }
    errors: list[dict[str, str]] = []

    for input_path, output_dir, command, tokens_path, preview_path in planned:
        input_kind = path_kind(input_path)
        assert input_kind is not None
        signature, signature_payload = run_signature(
            spec,
            weights_file=weights_file,
            input_path=input_path,
            input_kind=input_kind,
            device=device,
            precision=precision,
            image_size=args.image_size,
            frames=args.frames,
            seed=args.seed,
            save_preview=args.save_preview,
            orchestrator=orchestrator_provenance,
            extractor=extractor_provenance,
            model_code=model_code_provenance_record,
            model_configuration=model_configuration_provenance,
        )
        result_path = output_dir / "result.json"
        pooled_path = output_dir / "pooled.npy"
        existing_result = json.loads(result_path.read_text()) if result_path.exists() else None
        complete = tokens_path.exists() and pooled_path.exists() and existing_result is not None
        if complete and existing_result.get("signature") == signature and not args.force:
            print(f"[extract] result exists, skipping: {output_dir}")
            records_by_input[portable_path(input_path)] = existing_result
            continue
        if complete and not args.force:
            raise ValueError(
                f"{output_dir} contains a result from different inputs or settings. "
                "Pass --force or choose another --output-dir."
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            run_command(command, cwd=ROOT_DIR, log_path=output_dir / "extract.log")
            tokens = np.load(tokens_path, mmap_mode="r")
            if tokens.ndim < 2:
                raise ValueError(f"Expected token embeddings, got shape {tokens.shape}.")
            pooled = np.asarray(tokens, dtype=np.float32).mean(axis=-2, dtype=np.float32)
            np.save(pooled_path, pooled)

            record = {
                "schema_version": 1,
                "model": spec.name,
                "variant": spec.variant,
                "input_kind": input_kind,
                "input": file_provenance(
                    input_path,
                    include_hash=args.hash_inputs,
                ),
                "weights": file_provenance(
                    weights_file,
                    include_hash=args.hash_weights,
                ),
                "weights_root": portable_path(weights),
                "orchestrator": orchestrator_provenance,
                "model_code": model_code_provenance_record,
                "model_configuration": model_configuration_provenance,
                "extractor": extractor_provenance,
                "runtime": {
                    "device": device,
                    "precision": precision,
                },
                "configuration": {
                    "image_size": args.image_size,
                    "frames": args.frames if "video" in spec.input_kinds else None,
                    "seed": args.seed,
                },
                "signature": signature,
                "signature_payload": signature_payload,
                "command": shlex.join(command),
                "token_shape": list(tokens.shape),
                "pooled_shape": list(pooled.shape),
                "artifacts": {
                    "tokens": file_provenance(tokens_path, include_hash=True),
                    "pooled": file_provenance(pooled_path, include_hash=True),
                    "preview": (
                        file_provenance(preview_path, include_hash=True)
                        if preview_path is not None and preview_path.exists()
                        else None
                    ),
                    "log": portable_path(output_dir / "extract.log"),
                },
            }
            result_path.write_text(json.dumps(record, indent=2) + "\n")
            records_by_input[portable_path(input_path)] = record
            print(
                f"[extract] {input_path.name}: "
                f"tokens={tuple(tokens.shape)} pooled={tuple(pooled.shape)}"
            )
        except Exception as exc:
            errors.append(
                {
                    "input": portable_path(input_path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            if not args.continue_on_error:
                raise

        records = [
            records_by_input[key]
            for key in sorted(records_by_input)
        ]
        manifest = build_manifest(
            args=args,
            spec=spec,
            weights=weights,
            device=device,
            precision=precision,
            records=records,
            errors=errors,
        )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"manifest={manifest_path}")
    print(f"completed={len(records_by_input)} failed={len(errors)}")


if __name__ == "__main__":
    main()
