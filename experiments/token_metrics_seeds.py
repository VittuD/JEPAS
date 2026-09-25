#!/usr/bin/env python3
"""Combine several bulk runs (one per seed) into mean / sd / range per configuration.

Each input is a bulk output directory (or its `aggregate/` folder) that contains
`aggregates.csv` from token_metrics_aggregate.py. A "seed" here is a different
seeded draw of the samples, so the spread measures sampling variability of the
per-configuration means; with only a few seeds the sd is a rough guide, so min/max
are reported too. It also reports, per dataset, whether the model ordering by a
chosen metric is the same in every seed.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

KEY = ("dataset", "model", "method", "params")
DEFAULT_METRICS = (
    "boundary_fraction",
    "total_connected_components",
    "adjacent_cosine_similarity",
    "pca_top3_sum",
    "silhouette",
)
# The ratio is meaningless where the null mean is ~0 (adjacent cosine similarity of
# random tokens is ~0, so real/null explodes); those metrics are tabulated as raw means.
TABLE_SUFFIX = {"adjacent_cosine_similarity": "mean"}
# Columns taken per metric: the raw mean and the value relative to the null baseline.
SUFFIXES = ("mean", "ratio")


def find_aggregates(path: Path) -> Path:
    for candidate in (path, path / "aggregates.csv", path / "aggregate" / "aggregates.csv"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No aggregates.csv under {path}.")


def load_runs(paths: list[Path]) -> list[dict[tuple, dict[str, str]]]:
    runs = []
    for path in paths:
        with find_aggregates(path).open(newline="") as handle:
            runs.append({tuple(row[k] for k in KEY): row for row in csv.DictReader(handle)})
    return runs


def _float(row: dict[str, str], column: str) -> float | None:
    value = row.get(column, "")
    return float(value) if value not in ("", None) else None


def combine(runs: list[dict[tuple, dict[str, str]]], metrics: tuple[str, ...]) -> list[dict[str, object]]:
    """One row per configuration present in every run."""
    keys = sorted(set.intersection(*(set(run) for run in runs)))
    rows = []
    for key in keys:
        out: dict[str, object] = dict(zip(KEY, key))
        out["n_seeds"] = len(runs)
        for metric in metrics:
            for suffix in SUFFIXES:
                column = f"{metric}_{suffix}"
                values = [v for v in (_float(run[key], column) for run in runs) if v is not None]
                if not values:
                    continue
                out[f"{column}_mean"] = statistics.fmean(values)
                out[f"{column}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
                out[f"{column}_min"] = min(values)
                out[f"{column}_max"] = max(values)
        rows.append(out)
    return rows


def ordering_consistency(
    runs: list[dict[tuple, dict[str, str]]], *, metric: str, suffix: str, method: str, params: str
) -> dict[str, tuple[list[list[str]], bool]]:
    """Per dataset: the model ordering (low to high) in each run, and whether they all agree."""
    column = f"{metric}_{suffix}"
    result: dict[str, tuple[list[list[str]], bool]] = {}
    datasets = sorted({key[0] for key in runs[0]})
    for dataset in datasets:
        orders = []
        for run in runs:
            scored = [
                (float(row[column]), key[1])
                for key, row in run.items()
                if key[0] == dataset and key[2] == method and key[3] == params and row.get(column)
            ]
            orders.append([model for _, model in sorted(scored)])
        result[dataset] = (orders, all(order == orders[0] for order in orders))
    return result


def to_markdown(rows: list[dict[str, object]], columns: list[str], *, digits: int = 3) -> str:
    def cell(value: object) -> str:
        return f"{value:.{digits}g}" if isinstance(value, float) else str(value)

    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines += ["| " + " | ".join(cell(row.get(c, "")) for c in columns) + " |" for row in rows]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", type=Path, help="Bulk output dirs (one per seed) or aggregates.csv files.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    p.add_argument("--order-metric", default="boundary_fraction")
    p.add_argument("--order-method", default="kmeans")
    p.add_argument("--order-params", default="k=3")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.runs) < 2:
        raise SystemExit("Give at least two runs.")
    runs = load_runs(args.runs)
    metrics = tuple(args.metrics)
    rows = combine(runs, metrics)
    if not rows:
        raise SystemExit("No configuration is present in every run.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    columns = sorted({c for row in rows for c in row}, key=lambda c: (c not in KEY, c))
    with (args.out_dir / "seed_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    shown = [r for r in rows if r["method"] == args.order_method and r["params"] == args.order_params]
    table_cols = ["dataset", "model"]
    for metric in metrics:
        suffix = TABLE_SUFFIX.get(metric, "ratio")
        table_cols += [f"{metric}_{suffix}_mean", f"{metric}_{suffix}_sd"]
    consistency = ordering_consistency(
        runs, metric=args.order_metric, suffix="ratio", method=args.order_method, params=args.order_params
    )
    lines = [
        f"# Seed summary ({len(runs)} runs)",
        "",
        f"Runs: {', '.join(str(p) for p in args.runs)}",
        "",
        f"## {args.order_method} {args.order_params}: mean and sd across seeds (null ratio; raw mean for adjacent_cosine_similarity)",
        "",
        to_markdown(shown, [c for c in table_cols if any(c in r for r in shown)]),
        "",
        f"## Model ordering by {args.order_metric} ratio (low = smoother), per seed",
        "",
    ]
    for dataset, (orders, same) in consistency.items():
        lines.append(f"- {dataset}: {'same in every seed' if same else 'DIFFERS between seeds'}: "
                     + " | ".join(" < ".join(order) for order in orders))
    (args.out_dir / "seed_summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out_dir / 'seed_summary.csv'} and seed_summary.md ({len(rows)} configurations)")


if __name__ == "__main__":
    main()
