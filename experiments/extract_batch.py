#!/usr/bin/env python3
"""Extract token embeddings for many inputs into one batched HDF5 per model.

Unlike `extract.py` (one subprocess, one forward pass, and one
`<input>/<model>/` folder per example — fine for a handful of examples, not
for thousands), this loads the model once, runs GPU forward passes in
sub-batches, and writes a single `embeddings.h5` per model with tokens shaped
`(N, T, D)` and a `sources` dataset recording which input each row came from.
Random sampling (without replacement, seeded) is used when more candidates
are available than requested.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from experiments.catalog import (  # noqa: E402
    MODEL_ALIASES,
    ModelSpec,
    canonical_model_name,
    model_specs,
)
from experiments.artifacts import (  # noqa: E402
    EMBEDDINGS_NAME,
    SCHEMA_NAME,
    valid_embeddings,
    write_embeddings,
)
from experiments.device import add_runtime_arguments, resolve_runtime  # noqa: E402
from experiments.extract import (  # noqa: E402
    checkpoint_for,
    expand_inputs,
    model_code_for,
    path_kind,
)
from experiments.provenance import (  # noqa: E402
    file_provenance,
    model_code_provenance,
    portable_path,
    run_command,
)


def resolve_kind(spec: ModelSpec, requested: str | None) -> str:
    if requested is not None:
        if requested not in spec.input_kinds:
            raise ValueError(f"{spec.name} accepts {spec.input_kinds}, not {requested!r}.")
        return requested
    if len(spec.input_kinds) == 1:
        return spec.input_kinds[0]
    raise ValueError(
        f"{spec.name} accepts {spec.input_kinds}; pass --kind to pick one "
        "for a homogeneous batch."
    )


def batch_command(
    spec: ModelSpec,
    *,
    weights: Path,
    input_list: Path,
    kind: str,
    adapter_output: Path,
    device: str,
    precision: str,
    image_size: int | None,
    frames: int,
    batch_size: int | None,
    seed: int,
) -> list[str]:
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
        portable_path(adapter_output),
        "--input-list",
        portable_path(input_list),
    ]
    if image_size is not None:
        command.extend(("--image-size", str(image_size)))
    if "video" in spec.input_kinds:
        command.extend(("--frames", str(frames)))
        command.extend(("--list-kind", kind))
    if batch_size is not None:
        command.extend(("--batch-size", str(batch_size)))
    return command


def parse_args() -> argparse.Namespace:
    choices = tuple(sorted((*model_specs(), *MODEL_ALIASES)))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=choices)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--kind", choices=("image", "video"), default=None)
    parser.add_argument("--n", type=int, required=True, help="Number of examples to sample.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=None, help="Defaults to the adapter's own default.")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    add_runtime_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frames <= 0 or args.frames % 2:
        raise ValueError("--frames must be a positive multiple of 2.")

    spec = model_specs()[canonical_model_name(args.model)]
    kind = resolve_kind(spec, args.kind)

    candidates = sorted(
        {p for p in expand_inputs(args.input, recursive=args.recursive) if path_kind(p) == kind}
    )
    if len(candidates) < args.n:
        raise ValueError(
            f"Only {len(candidates)} {kind} inputs found under {args.input}; "
            f"need {args.n}. Lower --n, add more --input paths, or pass --recursive."
        )
    rng = np.random.default_rng(args.seed)
    sample_idx = np.sort(rng.choice(len(candidates), size=args.n, replace=False))
    sampled = [candidates[i] for i in sample_idx]

    weights = (args.weights or spec.weights_path).expanduser().resolve()
    checkpoint = checkpoint_for(spec, weights)
    model_code = model_code_for(spec, weights)

    output_dir = (args.output_dir / spec.name).expanduser().resolve()
    embeddings_path = output_dir / EMBEDDINGS_NAME
    adapter_output = output_dir / ".tokens.npy"
    input_list_path = output_dir / "input_list.txt"

    if args.dry_run:
        command = batch_command(
            spec,
            weights=weights,
            input_list=input_list_path,
            kind=kind,
            adapter_output=adapter_output,
            device=args.device,
            precision=args.precision,
            image_size=args.image_size,
            frames=args.frames,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        print(f"n={len(sampled)} sampled from {len(candidates)} {kind} candidates")
        print(shlex.join(command))
        return

    if not args.force and valid_embeddings(embeddings_path):
        print(f"[reused] {output_dir}")
        return

    for required in (checkpoint, model_code):
        if not required.exists():
            raise FileNotFoundError(required)

    runtime = resolve_runtime(args.device, args.precision)
    device, precision = runtime.device.type, runtime.precision

    output_dir.mkdir(parents=True, exist_ok=True)
    input_list_path.write_text("\n".join(str(p) for p in sampled) + "\n")

    command = batch_command(
        spec,
        weights=weights,
        input_list=input_list_path,
        kind=kind,
        adapter_output=adapter_output,
        device=device,
        precision=precision,
        image_size=args.image_size,
        frames=args.frames,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    run_command(command, cwd=ROOT_DIR, log_path=output_dir / "run.log")
    if not adapter_output.is_file():
        raise FileNotFoundError(adapter_output)

    tokens = np.load(adapter_output, mmap_mode="r")
    if tokens.shape[0] != len(sampled):
        raise RuntimeError(
            f"Adapter wrote {tokens.shape[0]} rows for {len(sampled)} sampled inputs."
        )

    metadata = {
        "model": spec.name,
        "input_kind": kind,
        "n": len(sampled),
        "candidates_pool_size": len(candidates),
        "seed": args.seed,
        "checkpoint": portable_path(checkpoint),
        "command": shlex.join(command),
        "settings": {
            "device": device,
            "precision": precision,
            "image_size": args.image_size,
            "frames": args.frames,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
    }
    write_embeddings(
        embeddings_path,
        tokens,
        metadata=metadata,
        sources=[portable_path(p) for p in sampled],
    )
    del tokens
    adapter_output.unlink()

    manifest = {
        "schema": SCHEMA_NAME,
        "schema_version": 1,
        "experiment": args.output_dir.name,
        "model": spec.name,
        "n": len(sampled),
        "seed": args.seed,
        "device": device,
        "precision": precision,
        "checkpoint": file_provenance(checkpoint, include_hash=False),
        "model_code": model_code_provenance(model_code),
        "embeddings": portable_path(embeddings_path),
        "input_list": portable_path(input_list_path),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[computed] {output_dir}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
