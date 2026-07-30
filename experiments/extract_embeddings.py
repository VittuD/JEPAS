#!/usr/bin/env python3
"""Minimal CLI placeholder for JEPA embedding extraction experiments."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract image embeddings using model code from repos/."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=("echojepa-vjepa2", "ijepa", "neurojepa", "radjepa", "vjepa2"),
        help="Model family to use.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to a local checkpoint or HF-downloaded model directory.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input image file or directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output embedding file or directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(
        "Embedding extraction backends are not wired yet. "
        f"Requested model={args.model}, weights={args.weights}, input={args.input}, output={args.output}."
    )


if __name__ == "__main__":
    main()
