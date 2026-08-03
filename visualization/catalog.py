"""Discover visualization-ready experiments for the marimo browser."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUTS = ROOT_DIR / "experiments" / "outputs"
SCHEMA_NAME = "jepas-experiment-v1"


def discover_experiments(outputs: Path = DEFAULT_OUTPUTS) -> list[dict[str, Any]]:
    """Return directory-derived rows, columns, media, and metadata."""
    experiments = []
    outputs = outputs.expanduser().resolve()
    if not outputs.is_dir():
        return experiments

    for experiment_dir in sorted(path for path in outputs.iterdir() if path.is_dir()):
        try:
            manifest = json.loads((experiment_dir / "manifest.json").read_text())
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if manifest.get("schema") != SCHEMA_NAME:
            continue

        results = []
        for input_dir in sorted(path for path in experiment_dir.iterdir() if path.is_dir()):
            for model_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
                image = model_dir / "visualization.png"
                try:
                    metadata = json.loads((model_dir / "metadata.json").read_text())
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    continue
                if not image.is_file() or metadata.get("schema") != SCHEMA_NAME:
                    continue
                frame_names = metadata.get("frames", [])
                if not isinstance(frame_names, list):
                    continue
                frames = [model_dir / name for name in frame_names]
                if any(not frame.is_file() for frame in frames):
                    continue
                results.append(
                    {
                        "input": input_dir.name,
                        "model": model_dir.name,
                        "image": image,
                        "frames": frames,
                        "metadata": metadata,
                    }
                )

        if results:
            experiments.append(
                {
                    "id": experiment_dir.name,
                    "name": manifest.get("experiment", experiment_dir.name),
                    "rows": sorted({result["input"] for result in results}),
                    "columns": sorted({result["model"] for result in results}),
                    "results": results,
                }
            )
    return experiments
