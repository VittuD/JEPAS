#!/usr/bin/env python3
"""Safely remove HDF5 embeddings after visualization artifacts validate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from experiments.artifacts import (  # noqa: E402
    EMBEDDINGS_NAME,
    METADATA_NAME,
    cleanup_embeddings,
)


def result_directories(path: Path) -> list[Path]:
    path = path.expanduser().resolve()
    if (path / EMBEDDINGS_NAME).exists() or (path / METADATA_NAME).exists():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    return sorted(
        candidate.parent
        for candidate in path.glob("*/*/metadata.json")
        if candidate.is_file()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    directories = sorted(
        dict.fromkeys(
            result
            for directory in args.directories
            for result in result_directories(directory)
        )
    )
    for directory in directories:
        removed = cleanup_embeddings(directory)
        state = "removed" if removed else "already absent"
        print(f"[{state}] {directory / EMBEDDINGS_NAME}")


if __name__ == "__main__":
    main()
