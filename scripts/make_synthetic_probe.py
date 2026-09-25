#!/usr/bin/env python3
"""Generate the synthetic probe set. See experiments/SYNTHETIC_PROBE.md."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.synthetic_probe import main  # noqa: E402

if __name__ == "__main__":
    main()
