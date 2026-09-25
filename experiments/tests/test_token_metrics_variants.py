from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from experiments.token_metrics_variants import collect, parse_run


def _run_dir(root: Path, name: str, value: float, grid: list[int]) -> Path:
    directory = root / name
    (directory / "aggregate").mkdir(parents=True)
    (directory / "d" / "m" / "token_metrics").mkdir(parents=True)
    (directory / "d" / "m" / "token_metrics" / "manifest.json").write_text(json.dumps({"grid": grid}))
    fields = ["dataset", "model", "method", "params", "boundary_fraction_ratio",
              "total_connected_components_ratio", "adjacent_cosine_similarity_mean", "pca_top3_sum_ratio"]
    with (directory / "aggregate" / "aggregates.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for params in ("k=2", "k=3"):
            writer.writerow({"dataset": "d", "model": "m", "method": "kmeans", "params": params,
                             "boundary_fraction_ratio": value, "total_connected_components_ratio": 0.1,
                             "adjacent_cosine_similarity_mean": 0.5, "pca_top3_sum_ratio": 9})
    return directory


class VariantsTest(unittest.TestCase):
    def test_parse_run_requires_label_and_path(self) -> None:
        self.assertEqual(parse_run("a=/x"), ("a", Path("/x")))
        with self.assertRaises(ValueError):
            parse_run("nolabel")

    def test_rows_keep_labels_grid_and_filter_params(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = _run_dir(root, "a", 0.3, [8, 24, 24])
            b = _run_dir(root, "b", 0.5, [8, 16, 16])
            rows = collect([("res384", a), ("res256", b)], method="kmeans", params="k=3")
            self.assertEqual([(r["label"], r["grid"], r["boundary_fraction_ratio"]) for r in rows],
                             [("res256", "8x16x16", 0.5), ("res384", "8x24x24", 0.3)])

    def test_cli(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = _run_dir(root, "a", 0.3, [8, 24, 24])
            b = _run_dir(root, "b", 0.5, [8, 16, 16])
            subprocess.run(
                [sys.executable, "experiments/token_metrics_variants.py", "--run", f"res384={a}",
                 "--run", f"res256={b}", "--out-dir", str(root / "out")],
                check=True, capture_output=True, text=True,
            )
            self.assertIn("8x16x16", (root / "out" / "variants.md").read_text())


if __name__ == "__main__":
    unittest.main()
