#!/usr/bin/env python3
"""Metric versus generation parameter for the synthetic probe (SYNTHETIC_PROBE.md, Q2/Q4).

Joins the per-sample rows of a bulk run's `token_metrics/summary.csv` files with the
probe's `manifest.jsonl` (by the file's path below `synthetic_v*/`), groups by
(family, parameter value, model), and reports the mean of each metric plus its ratio
to the Gaussian null (the same null join as token_metrics_aggregate.py).

Only rows of one clustering configuration are shown (default: k-means k=3).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from experiments.token_metrics_aggregate import (  # noqa: E402
    add_null_normalization,
    aggregate_rows,
    iter_null_csvs,
    iter_summary_csvs,
    read_rows,
)

# The manifest field that is swept in each family ("-" = the family has no parameter).
PARAM_FIELD = {
    "noise_blocks": "cells", "stripes": "stripes", "checker": "cells",
    "regions": "k", "shapes": "count", "static_pattern": "k", "cut": "k",
    "moving_shape": "speed", "flicker": "mode",
}
METRICS = ("boundary_fraction", "total_connected_components", "adjacent_cosine_similarity")
_REL_RE = re.compile(r"synthetic_v\d+[^/]*/(.+)$")


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    with path.open() as handle:
        return {rec["file"]: rec for rec in map(json.loads, handle)}


def attach_params(rows: list[dict[str, Any]], manifest: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Add `param` (value of the family's swept field) to each row; drop rows not in the manifest."""
    out = []
    for row in rows:
        match = _REL_RE.search(str(row.get("source", "")))
        record = manifest.get(match.group(1)) if match else None
        if record is None:
            continue
        field = PARAM_FIELD.get(record["family"])
        out.append({**row, "family": record["family"], "param": str(record[field]) if field else "-"})
    return out


def _sort_key(value: str):
    try:
        return (0, float(value))
    except ValueError:
        return (1, value)


def curve_rows(
    rows: list[dict[str, Any]], null_rows: list[dict[str, Any]], *, method: str, params: str
) -> list[dict[str, Any]]:
    rows = [r for r in rows if r["method"] == method and r["params"] == params]
    aggregates = aggregate_rows(rows, group_by=("family", "param", "model", "method", "params"), metrics=METRICS)
    if null_rows:
        null_aggregates = aggregate_rows(null_rows, group_by=("model", "method", "params"), metrics=METRICS)
        aggregates = add_null_normalization(aggregates, null_aggregates, METRICS)
    return sorted(aggregates, key=lambda r: (r["family"], _sort_key(r["param"]), r["model"]))


def to_markdown(aggregates: list[dict[str, Any]]) -> str:
    """One table per family: rows = parameter value, columns = model, cell = ratio (or mean)."""
    lines: list[str] = []
    for family in sorted({r["family"] for r in aggregates}):
        subset = [r for r in aggregates if r["family"] == family]
        models = sorted({r["model"] for r in subset})
        params = sorted({r["param"] for r in subset}, key=_sort_key)
        for metric, suffix in (("boundary_fraction", "ratio"), ("total_connected_components", "ratio"), ("adjacent_cosine_similarity", "mean")):
            lines += [f"### {family}: {metric} ({suffix})", "", "| param | " + " | ".join(models) + " |", "| --- | " + " | ".join("---" for _ in models) + " |"]
            for param in params:
                cells = []
                for model in models:
                    match = [r for r in subset if r["param"] == param and r["model"] == model]
                    value = match[0].get(f"{metric}_{suffix}") if match else None
                    cells.append(f"{value:.3f}" if isinstance(value, float) else "")
                lines.append(f"| {param} | " + " | ".join(cells) + " |")
            lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path, help="Bulk output dir containing synthetic-image/ and/or synthetic-video/.")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--method", default="kmeans")
    p.add_argument("--params", default="k=3")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    summaries = [p for d in ("synthetic-image", "synthetic-video") if (args.run_dir / d).is_dir()
                 for p in iter_summary_csvs([str(args.run_dir / d)])]
    if not summaries:
        raise SystemExit(f"No synthetic summary.csv under {args.run_dir}.")
    rows = attach_params(read_rows(summaries), load_manifest(args.manifest))
    null_paths = iter_null_csvs([str(args.run_dir / "null_baselines")])
    aggregates = curve_rows(rows, read_rows(null_paths) if null_paths else [], method=args.method, params=args.params)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    columns = sorted({c for r in aggregates for c in r})
    with (args.out_dir / "curves.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(aggregates)
    text = f"# Synthetic probe curves ({args.method} {args.params})\n\n" + to_markdown(aggregates)
    (args.out_dir / "curves.md").write_text(text)
    print(f"wrote {args.out_dir / 'curves.md'} ({len(aggregates)} groups from {len(rows)} sample rows)")


if __name__ == "__main__":
    main()
