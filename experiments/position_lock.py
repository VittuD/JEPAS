#!/usr/bin/env python3
"""How much of a model's token map is fixed to grid position, whatever the input? (SYNTHETIC_PROBE.md, "Position-locked patterns")

For one group of samples (a synthetic family, or all samples of a real dataset) and one
model, on tokens shaped (samples, T*H*W, D) with grid (T, H, W):

- pos_variance_fraction: share of token variance explained by the per-position mean over
  samples. The chance level for pure noise is about 1/samples (reported as `pos_chance`).
- split_half_cosine: cosine similarity of the position-mean maps (grand mean removed) of two
  disjoint halves of the samples. Near 1 = the same map whatever the input; near 0 = none.
- nyquist_ratio: fraction of the per-sample spatial spectral power (DC removed) that sits on
  the Nyquist lines (fx = W/2 or fy = H/2, i.e. period-2 lattices), divided by the fraction
  a white spectrum would put there. 1 = no lattice; above 1 = period-2 structure (the ratio is
  capped at 1/expected, about 4 on a 8x8 grid and 12 on 24x24; `nyquist_share` is the raw share).
- dominant_period: spatial period (in tokens) of the strongest non-DC component of the
  same spectrum (inf if none).

Run under srun/sbatch, not on a login node: it reads whole embeddings files.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

_FAMILY_RE = re.compile(r"synthetic_v\d+[^/]*/(?:image|video)/([^/]+)/")


def family_of(source: str) -> str:
    match = _FAMILY_RE.search(source)
    return match.group(1) if match else "all"


def position_stats(tokens: np.ndarray) -> dict[str, float]:
    """tokens: (S, N, D). Returns the position-locked variance fraction and split-half cosine."""
    x = tokens.astype(np.float64)
    s = x.shape[0]
    grand = x.mean(axis=(0, 1), keepdims=True)
    centred = x - grand
    total = float((centred**2).sum(axis=2).mean())
    mean_map = centred.mean(axis=0)
    pos = float((mean_map**2).sum(axis=1).mean())
    out = {"pos_variance_fraction": pos / total if total > 0 else 0.0, "pos_chance": 1.0 / s}
    if s >= 2:
        half = s // 2
        a = centred[:half].mean(axis=0).ravel()
        b = centred[half : 2 * half].mean(axis=0).ravel()
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        out["split_half_cosine"] = float(a @ b / denom) if denom > 0 else 0.0
    else:
        out["split_half_cosine"] = float("nan")
    return out


def spectrum_stats(tokens: np.ndarray, grid: tuple[int, ...]) -> dict[str, float]:
    """Mean per-sample spatial power spectrum over (H, W); Nyquist-line share and dominant period."""
    h, w = grid[-2], grid[-1]
    lead = int(np.prod(grid[:-2])) if len(grid) > 2 else 1
    x = tokens.astype(np.float32).reshape(tokens.shape[0], lead, h, w, -1)
    x = x - x.mean(axis=(2, 3), keepdims=True)
    power = (np.abs(np.fft.fft2(x, axes=(2, 3))) ** 2).mean(axis=(0, 1, 4))  # (H, W)
    power[0, 0] = 0.0
    total = power.sum()
    if total <= 0:
        return {"nyquist_share": float("nan"), "nyquist_ratio": float("nan"), "dominant_period": float("inf")}
    mask = np.zeros((h, w), dtype=bool)
    if h % 2 == 0:
        mask[h // 2, :] = True
    if w % 2 == 0:
        mask[:, w // 2] = True
    mask[0, 0] = False
    expected = (mask.sum()) / (h * w - 1)
    share = power[mask].sum() / total
    fy, fx = np.unravel_index(int(np.argmax(power)), power.shape)
    fy, fx = min(fy, h - fy), min(fx, w - fx)
    freq = np.hypot(fy / h, fx / w)
    return {
        "nyquist_share": float(share),
        "nyquist_ratio": float(share / expected) if expected > 0 else float("nan"),
        "dominant_period": float(1.0 / freq) if freq > 0 else float("inf"),
    }


def analyse(tokens: np.ndarray, grid: tuple[int, ...]) -> dict[str, float]:
    return {**position_stats(tokens), **spectrum_stats(tokens, grid)}


def main() -> None:
    import h5py

    from experiments.token_metrics_null import geometry_from_embeddings

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, action="append", help="bulk output dir (repeatable)")
    parser.add_argument("--dataset", required=True, help="e.g. synthetic-video or kinetics")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--families", nargs="*", default=None, help="synthetic families to keep (default: all groups)")
    parser.add_argument("--max-samples", type=int, default=128, help="per group, first N (bounds memory)")
    parser.add_argument("--out-csv", type=Path, default=None)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for run in args.run:
        dataset_dir = run / args.dataset
        models = args.models or sorted(p.name for p in dataset_dir.iterdir() if (p / "embeddings.h5").is_file())
        for model in models:
            path = dataset_dir / model / "embeddings.h5"
            if not path.is_file():
                continue
            grid = geometry_from_embeddings(path)[2]
            with h5py.File(path, "r") as handle:
                sources = [x.decode() if isinstance(x, bytes) else str(x) for x in handle["sources"][...]]
                groups: dict[str, list[int]] = {}
                for i, source in enumerate(sources):
                    groups.setdefault(family_of(source), []).append(i)
                for family, idx in sorted(groups.items()):
                    if args.families and family not in args.families:
                        continue
                    idx = idx[: args.max_samples]
                    tokens = handle["tokens"][idx]
                    rows.append({"model": model, "dataset": args.dataset, "family": family,
                                 "samples": len(idx), "grid": "x".join(map(str, grid)), **analyse(tokens, grid)})
    if not rows:
        raise SystemExit("No rows produced.")
    cols = list(rows[0])
    print("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
