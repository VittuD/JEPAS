#!/usr/bin/env python3
"""Validate frame-reference and temporal-slice invariants for cross-media results."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from experiments.shared.provenance import resolve_portable_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    hashes_by_input_resolution: dict[tuple[str, int], set[str]] = defaultdict(set)

    for result in manifest["results"]:
        input_name = str(result["input_name"])
        input_resolution = int(result["input_resolution"])
        preview_hash = str(result["artifacts"]["preview"]["sha256"])
        hashes_by_input_resolution[(input_name, input_resolution)].add(preview_hash)

        if manifest.get("resolution_mode") == "native":
            native_resolution = int(result["native_resolution"])
            if input_resolution != native_resolution:
                raise AssertionError(
                    f"{result['name']}: expected native resolution "
                    f"{native_resolution}, got {input_resolution}"
                )

        visualization_manifest = resolve_portable_path(
            str(result["artifacts"]["visualization_manifest"])
        )
        visualization = json.loads(visualization_manifest.read_text())
        expected_slice = "0" if result["model_modality"] == "video" else "all"
        actual_slice = str(visualization["slice_index"])
        if actual_slice != expected_slice:
            raise AssertionError(
                f"{result['name']}: expected slice {expected_slice}, got {actual_slice}"
            )

    mismatched = {
        f"{input_name}@{resolution}": sorted(hashes)
        for (input_name, resolution), hashes in hashes_by_input_resolution.items()
        if len(hashes) != 1
    }
    if mismatched:
        raise AssertionError(f"Models use different displayed references: {mismatched}")

    print(
        f"[cross_media] validated {len(manifest['results'])} results: "
        "one reference per input/resolution, video slice=0",
        flush=True,
    )


if __name__ == "__main__":
    main()
