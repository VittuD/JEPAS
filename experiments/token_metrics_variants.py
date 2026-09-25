#!/usr/bin/env python3
"""Line up labelled bulk runs (e.g. the same model at two input resolutions).

Unlike token_metrics_seeds.py, which averages runs that are replicates of each
other, this keeps every run separate: each `--run label=DIR` contributes rows
tagged with its label, so a resolution control reads as

    dataset  model          label   grid  boundary_fraction_ratio ...

`grid` is recovered from the run's per-model token_metrics manifest when present.
Only configurations of the chosen method/params are shown.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from experiments.token_metrics_seeds import find_aggregates, to_markdown  # noqa: E402

# Ratio to the null, except where the null mean is ~0 (see token_metrics_seeds).
METRICS = {
    "boundary_fraction": "ratio",
    "total_connected_components": "ratio",
    "adjacent_cosine_similarity": "mean",
    "pca_top3_sum": "ratio",
}


def parse_run(spec: str) -> tuple[str, Path]:
    label, sep, path = spec.partition("=")
    if not sep or not label or not path:
        raise ValueError(f"--run expects LABEL=DIR, got {spec!r}.")
    return label, Path(path)


def grid_of(run_dir: Path, dataset: str, model: str) -> str:
    manifest = run_dir / dataset / model / "token_metrics" / "manifest.json"
    try:
        return "x".join(str(v) for v in json.loads(manifest.read_text())["grid"])
    except (OSError, KeyError, ValueError):
        return ""


def collect(runs: list[tuple[str, Path]], *, method: str, params: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label, directory in runs:
        base = find_aggregates(directory)
        run_dir = base.parent.parent if base.parent.name == "aggregate" else base.parent
        with base.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row["method"] != method or row["params"] != params:
                    continue
                out: dict[str, object] = {
                    "dataset": row["dataset"],
                    "model": row["model"],
                    "label": label,
                    "grid": grid_of(run_dir, row["dataset"], row["model"]),
                }
                for metric, suffix in METRICS.items():
                    value = row.get(f"{metric}_{suffix}", "")
                    out[f"{metric}_{suffix}"] = float(value) if value else ""
                rows.append(out)
    return sorted(rows, key=lambda r: (r["dataset"], r["model"], r["label"]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True, help="LABEL=DIR (bulk dir or aggregates.csv); repeatable.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--method", default="kmeans")
    p.add_argument("--params", default="k=3")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect([parse_run(spec) for spec in args.run], method=args.method, params=args.params)
    if not rows:
        raise SystemExit("No matching rows.")
    columns = ["dataset", "model", "label", "grid"] + [f"{m}_{s}" for m, s in METRICS.items()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "variants.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    text = f"# Variants ({args.method} {args.params})\n\n" + to_markdown(rows, columns) + "\n"
    (args.out_dir / "variants.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
