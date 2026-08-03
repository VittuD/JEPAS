#!/usr/bin/env python3
"""Run the cross-media matrix at each model's native spatial resolution."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from experiments.cross_media.run import main  # noqa: E402


if __name__ == "__main__":
    main(
        resolution_mode_default="native",
        output_root_default=(
            ROOT_DIR
            / "experiments"
            / "outputs"
            / "cross_media_native_resolution"
        ),
        runner_path=Path(__file__),
    )
