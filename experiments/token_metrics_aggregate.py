#!/usr/bin/env python3
"""Aggregate experiments/token_metrics.py summary.csv files across models.

Adapted from `ijepa_lite`'s `scripts/analyze_token_embedding_viz_stats.py`
(https://github.com/VittuD/ijepa_lite), generalized from that repo's
checkpoint-family (chest_*/in1k_*) comparison scheme to plain model-vs-model
comparison, since here each row's "model" is one of this repo's five JEPA
variants rather than a training-variant checkpoint of the same architecture.

Two models only get a real *matched* (index/source-joined) comparison when
they were run against the exact same sampled inputs in the same order --
e.g. vjepa2-vitl and vjepa2-1-vitb both sampled the same 2048 Diving48 clips
with --seed 0. Pass such pairs explicitly with --compare; nothing is
inferred automatically since most model pairs here use different datasets.

Example:

    python3 experiments/token_metrics_aggregate.py \
        experiments/outputs/bulk_2048/*/token_metrics/summary.csv \
        --compare vjepa2-vitl vjepa2-1-vitb \
        --out-dir experiments/outputs/bulk_2048/token_metrics_aggregate
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from experiments.catalog import canonical_model_name, model_specs  # noqa: E402


# Substring -> dataset label, matched against each row's "source" path.
# Order matters only in that longer/more-specific fragments should come
# first if two could ever both match; none currently overlap.
_DATASET_FRAGMENTS = (
    ("imagenet", "imagenet"),
    ("rsna-pneumonia", "rsna"),
    ("detection/coco", "coco"),
    ("diving48", "diving48"),
    ("kinetics", "kinetics"),
    ("epic-kitchens", "epic-kitchens"),
    ("stanforddataset", "echonet"),
    ("echonet", "echonet"),
)


# Synthetic probe set: <...>/synthetic_v1_seed42/<image|video>/<family>/<file>. One label
# per family ("syn-<family>") so a single extraction job yields per-family rows.
_SYNTHETIC_RE = re.compile(r"synthetic_v\d+[^/]*/(?:image|video)/([^/]+)/")


def infer_dataset(source: str) -> str:
    match = _SYNTHETIC_RE.search(source)
    if match:
        return f"syn-{match.group(1)}"
    lowered = source.lower()
    for fragment, label in _DATASET_FRAGMENTS:
        if fragment in lowered:
            return label
    return "unknown"


def infer_modality(model: str) -> str:
    try:
        return model_specs()[canonical_model_name(model)].modality
    except KeyError:
        return "unknown"


PRIMARY_METRICS = (
    "boundary_fraction",
    "total_connected_components",
    "extra_connected_components",
    "max_connected_components",
    "cluster_entropy",
    "silhouette",
    "calinski_harabasz",
    "davies_bouldin",
    "adjacent_cosine_distance",
    "adjacent_cosine_distance_sd",
    "adjacent_cosine_similarity",
    "pca_pc1",
    "pca_pc2",
    "pca_pc3",
    "pca_top3_sum",
)


@dataclass(frozen=True)
class Stat:
    n: int
    mean: float
    sd: float
    sem: float
    min: float
    max: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "inputs",
        nargs="+",
        help="summary.csv files or directories (searched recursively for summary.csv).",
    )
    p.add_argument("--out-dir", default=None, help="Optional directory for aggregate CSV/JSON/Markdown outputs.")
    p.add_argument(
        "--group-by",
        nargs="+",
        default=("model", "dataset", "method", "params"),
        choices=("model", "dataset", "modality", "dataset_index", "method", "params"),
        help="Columns used for aggregate rows.",
    )
    p.add_argument("--metrics", nargs="+", default=PRIMARY_METRICS, help="Numeric metrics to aggregate.")
    p.add_argument(
        "--compare",
        nargs=2,
        action="append",
        metavar=("BASELINE_MODEL", "OTHER_MODEL"),
        help=(
            "Compare two models on matched source/method/params rows (only "
            "meaningful when both models were run against the same sampled "
            "inputs in the same order). Repeat for multiple pairs."
        ),
    )
    p.add_argument(
        "--null",
        nargs="+",
        default=None,
        help=(
            "null_summary.csv files or directories from token_metrics_null.py "
            "(searched recursively for null_summary.csv). When given, every "
            "aggregate row gets {metric}_null_mean/{metric}_z/{metric}_ratio "
            "columns, normalizing against random-token baselines at that "
            "model's exact token geometry -- see token_metrics_null.py's "
            "docstring for why raw values across differently-shaped models "
            "aren't directly comparable."
        ),
    )
    return p.parse_args()


def iter_summary_csvs(inputs: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_file():
            paths.append(path)
        elif path.is_dir():
            paths.extend(sorted(path.rglob("summary.csv")))
        else:
            print(f"warning: missing input skipped: {path}", file=sys.stderr)
    return sorted(set(p.resolve() for p in paths))


def iter_null_csvs(inputs: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_file():
            paths.append(path)
        elif path.is_dir():
            paths.extend(sorted(path.rglob("null_summary.csv")))
        else:
            print(f"warning: missing null input skipped: {path}", file=sys.stderr)
    return sorted(set(p.resolve() for p in paths))


def add_null_normalization(
    aggregates: list[dict[str, Any]], null_aggregates: list[dict[str, Any]], metrics: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Join {metric}_null_mean/_z/_ratio onto aggregates by (model, method, params).

    Null aggregates are geometry-only (no dataset), so every real aggregate
    row for that model -- regardless of which dataset it's grouped by --
    joins to the same null baseline.
    """
    null_by_key = {(row.get("model", ""), row.get("method", ""), row.get("params", "")): row for row in null_aggregates}
    for row in aggregates:
        key = (row.get("model", ""), row.get("method", ""), row.get("params", ""))
        null_row = null_by_key.get(key)
        if null_row is None:
            continue
        for metric in metrics:
            null_mean, null_sd = null_row.get(f"{metric}_mean"), null_row.get(f"{metric}_sd")
            real_mean = row.get(f"{metric}_mean")
            if null_mean is None or real_mean is None:
                continue
            row[f"{metric}_null_mean"] = null_mean
            if null_sd is not None and null_sd > 1e-12:
                row[f"{metric}_z"] = (real_mean - null_mean) / null_sd
            if abs(null_mean) > 1e-12:
                row[f"{metric}_ratio"] = real_mean / null_mean
    return aggregates


def parse_json(value: str, default: Any) -> Any:
    if value is None or value == "":
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def parse_float(value: Any) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def enrich_row(row: dict[str, str], source_csv: Path) -> dict[str, Any]:
    out: dict[str, Any] = dict(row)
    out["source_csv"] = str(source_csv)
    out["dataset"] = infer_dataset(str(row.get("source", "")))
    out["modality"] = infer_modality(str(row.get("model", "")))

    components = parse_json(str(row.get("connected_components", "")), {})
    component_values = [int(v) for v in components.values()] if isinstance(components, dict) else []
    out["total_connected_components"] = float(sum(component_values))
    out["extra_connected_components"] = float(sum(max(v - 1, 0) for v in component_values))
    out["max_connected_components"] = float(max(component_values) if component_values else 0)

    pca = parse_json(str(row.get("pca_explained_variance", "")), [])
    if not isinstance(pca, list):
        pca = []
    for idx in range(3):
        out[f"pca_pc{idx + 1}"] = float(pca[idx]) if idx < len(pca) else math.nan
    out["pca_top3_sum"] = float(sum(float(v) for v in pca[:3]))

    for key in (
        "n_clusters",
        "noise_count",
        "cluster_entropy",
        "boundary_fraction",
        "silhouette",
        "calinski_harabasz",
        "davies_bouldin",
        "adjacent_cosine_distance",
        "adjacent_cosine_distance_sd",
        "adjacent_cosine_similarity",
    ):
        out[key] = parse_float(row.get(key))
    return out


def read_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(enrich_row(row, path))
    return rows


def finite_values(rows: list[dict[str, Any]], metric: str) -> list[float]:
    values = [parse_float(row.get(metric)) for row in rows]
    return [v for v in values if math.isfinite(v)]


def summarize(values: list[float]) -> Stat | None:
    if not values:
        return None
    sd = stdev(values) if len(values) >= 2 else 0.0
    return Stat(n=len(values), mean=mean(values), sd=sd, sem=sd / math.sqrt(len(values)) if values else math.nan, min=min(values), max=max(values))


def grouped(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(key, "")) for key in keys)].append(row)
    return groups


def aggregate_rows(rows: list[dict[str, Any]], *, group_by: tuple[str, ...], metrics: tuple[str, ...]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key, group_rows in sorted(grouped(rows, group_by).items()):
        row = {name: value for name, value in zip(group_by, key)}
        row["n_rows"] = len(group_rows)
        for metric in metrics:
            stat = summarize(finite_values(group_rows, metric))
            if stat is None:
                continue
            row[f"{metric}_n"] = stat.n
            row[f"{metric}_mean"] = stat.mean
            row[f"{metric}_sd"] = stat.sd
            row[f"{metric}_sem"] = stat.sem
            row[f"{metric}_min"] = stat.min
            row[f"{metric}_max"] = stat.max
        out.append(row)
    return out


def match_key(row: dict[str, Any]) -> tuple[str, ...]:
    # "source" (the original input file path) is the true join key across
    # models -- unlike ijepa_lite's single-dataset viz runs, here different
    # models draw from different datasets, so sample_id/dataset_index alone
    # would silently mismatch rows that just happen to share an index.
    return (str(row.get("source", "")), str(row.get("method", "")), str(row.get("params", "")))


def compare_pair(rows: list[dict[str, Any]], *, baseline: str, other: str, metrics: tuple[str, ...]) -> list[dict[str, Any]]:
    baseline_rows = [row for row in rows if row.get("model") == baseline]
    other_rows = [row for row in rows if row.get("model") == other]
    baseline_by_key = {match_key(row): row for row in baseline_rows}
    other_by_key = {match_key(row): row for row in other_rows}
    common_keys = sorted(set(baseline_by_key) & set(other_by_key))

    if not common_keys:
        print(
            f"warning: no matched rows between {baseline!r} and {other!r} "
            "(they must share source inputs, e.g. same dataset + seed).",
            file=sys.stderr,
        )
        return []

    by_method_params: dict[tuple[str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for key in common_keys:
        b, o = baseline_by_key[key], other_by_key[key]
        by_method_params[(str(b["method"]), str(b["params"]))].append((b, o))

    pair_rows: list[dict[str, Any]] = []
    for (method, params), pairs in sorted(by_method_params.items()):
        row: dict[str, Any] = {
            "baseline": baseline,
            "other": other,
            "method": method,
            "params": params,
            "n_matched": len(pairs),
        }
        for metric in metrics:
            deltas, baseline_values, other_values = [], [], []
            for b, o in pairs:
                bv, ov = parse_float(b.get(metric)), parse_float(o.get(metric))
                if not (math.isfinite(bv) and math.isfinite(ov)):
                    continue
                baseline_values.append(bv)
                other_values.append(ov)
                deltas.append(ov - bv)
            bstat, ostat, dstat = summarize(baseline_values), summarize(other_values), summarize(deltas)
            if bstat is None or ostat is None or dstat is None:
                continue
            row[f"{metric}_baseline_mean"] = bstat.mean
            row[f"{metric}_other_mean"] = ostat.mean
            row[f"{metric}_delta_mean"] = dstat.mean
            row[f"{metric}_delta_sd"] = dstat.sd
            row[f"{metric}_delta_sem"] = dstat.sem
            if abs(bstat.mean) > 1e-12:
                row[f"{metric}_relative_delta_pct"] = 100.0 * dstat.mean / abs(bstat.mean)
        pair_rows.append(row)
    return pair_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return "" if math.isnan(value) else f"{value:.4g}"
    return str(value)


def markdown_table(rows: list[dict[str, Any]], columns: list[str], limit: int | None = None) -> str:
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        return "_No rows._\n"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines) + "\n"


def write_report(*, rows: list[dict[str, Any]], aggregates: list[dict[str, Any]], comparisons: list[dict[str, Any]], metrics: tuple[str, ...]) -> str:
    report: list[str] = ["# Token Embedding Statistics\n", f"Input rows: {len(rows)}\n", f"Metrics: {', '.join(metrics)}\n"]

    compact_cols = [
        "model",
        "dataset",
        "modality",
        "method",
        "params",
        "n_rows",
        "boundary_fraction_mean",
        "boundary_fraction_sd",
        "total_connected_components_mean",
        "extra_connected_components_mean",
        "adjacent_cosine_distance_mean",
        "adjacent_cosine_similarity_mean",
        "silhouette_mean",
        "pca_top3_sum_mean",
    ]
    existing_cols = [col for col in compact_cols if any(col in row for row in aggregates)]
    report.append("## Aggregates\n")
    report.append(markdown_table(aggregates, existing_cols, limit=200))

    null_normalized_metrics = [
        metric
        for metric in ("boundary_fraction", "total_connected_components", "adjacent_cosine_distance", "pca_top3_sum")
        if any(f"{metric}_z" in row for row in aggregates)
    ]
    if null_normalized_metrics:
        null_cols = ["model", "dataset", "modality", "method", "params"]
        for metric in null_normalized_metrics:
            null_cols.extend([f"{metric}_mean", f"{metric}_null_mean", f"{metric}_z", f"{metric}_ratio"])
        existing_null_cols = [col for col in null_cols if any(col in row for row in aggregates)]
        report.append("## Null-Normalized (vs. random tokens at the same T/D/grid)\n")
        report.append(
            "`_null_mean` is the metric's value on pure Gaussian noise at that model's exact token "
            "geometry (see token_metrics_null.py); `_z` = (real - null_mean) / null_sd; `_ratio` = "
            "real / null_mean. Use these, not the raw `_mean` columns, when comparing across models "
            "with different token counts or embedding dimensions.\n"
        )
        report.append(markdown_table(aggregates, existing_null_cols, limit=200))

    if comparisons:
        compare_cols = [
            "baseline",
            "other",
            "method",
            "params",
            "n_matched",
            "boundary_fraction_baseline_mean",
            "boundary_fraction_other_mean",
            "boundary_fraction_delta_mean",
            "boundary_fraction_relative_delta_pct",
            "total_connected_components_baseline_mean",
            "total_connected_components_other_mean",
            "total_connected_components_delta_mean",
            "adjacent_cosine_distance_baseline_mean",
            "adjacent_cosine_distance_other_mean",
            "adjacent_cosine_distance_delta_mean",
            "adjacent_cosine_distance_relative_delta_pct",
            "pca_top3_sum_delta_mean",
        ]
        existing_compare_cols = [col for col in compare_cols if any(col in row for row in comparisons)]
        report.append("## Matched Comparisons\n")
        report.append(markdown_table(comparisons, existing_compare_cols, limit=120))

    report.append("## Reading Guide\n")
    report.append(
        "- Lower `boundary_fraction` means fewer neighboring token-label changes (spatial, and for video also temporal).\n"
        "- Lower `total_connected_components` / `extra_connected_components` means fewer disjoint islands per cluster.\n"
        "- Lower `adjacent_cosine_distance` means neighboring raw tokens are more similar before clustering.\n"
        "- Similar silhouette/Calinski/Davies-Bouldin with lower fragmentation suggests smoother spatial(-temporal) organization, not merely easier feature-space clustering.\n"
        "- Similar `pca_top3_sum` means smoothness differences aren't explained by a large change in top-3 PCA variance alone.\n"
        "- Raw metric values are NOT comparable across models with different token counts (T) or embedding "
        "dims (D) -- more tokens mechanically allows more connected components, and PCA variance concentrates "
        "differently by D. Use the Null-Normalized table (or filter to a single `modality`/matched `dataset`) "
        "for cross-model comparisons; raw `_mean` columns are only safe to compare within the same model on "
        "different datasets, or between models that share identical (T, D, grid).\n"
    )
    return "\n".join(report)


def main() -> None:
    args = parse_args()
    paths = iter_summary_csvs(args.inputs)
    if not paths:
        raise SystemExit("No summary.csv files found.")

    rows = read_rows(paths)
    metrics = tuple(args.metrics)
    aggregates = aggregate_rows(rows, group_by=tuple(args.group_by), metrics=metrics)

    if args.null:
        null_paths = iter_null_csvs(args.null)
        if not null_paths:
            print("warning: --null given but no null_summary.csv files found", file=sys.stderr)
        else:
            null_rows = read_rows(null_paths)
            null_aggregates = aggregate_rows(null_rows, group_by=("model", "method", "params"), metrics=metrics)
            aggregates = add_null_normalization(aggregates, null_aggregates, metrics)

    comparisons: list[dict[str, Any]] = []
    for baseline, other in args.compare or []:
        comparisons.extend(compare_pair(rows, baseline=baseline, other=other, metrics=metrics))

    report = write_report(rows=rows, aggregates=aggregates, comparisons=comparisons, metrics=metrics)
    print(report)

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_csv(out_dir / "aggregates.csv", aggregates)
        write_csv(out_dir / "comparisons.csv", comparisons)
        (out_dir / "report.md").write_text(report)
        (out_dir / "aggregates.json").write_text(json.dumps(aggregates, indent=2) + "\n")
        (out_dir / "comparisons.json").write_text(json.dumps(comparisons, indent=2) + "\n")


if __name__ == "__main__":
    main()
