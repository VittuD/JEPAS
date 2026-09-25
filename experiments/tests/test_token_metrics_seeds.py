from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from experiments.token_metrics_seeds import combine, load_runs, ordering_consistency


def _write(path: Path, values: dict[tuple[str, str], float]) -> Path:
    path.mkdir(parents=True)
    with (path / "aggregates.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["dataset", "model", "method", "params", "boundary_fraction_mean", "boundary_fraction_ratio"]
        )
        writer.writeheader()
        for (dataset, model), value in values.items():
            writer.writerow(
                {"dataset": dataset, "model": model, "method": "kmeans", "params": "k=3",
                 "boundary_fraction_mean": value, "boundary_fraction_ratio": value * 2}
            )
    return path


class SeedSummaryTest(unittest.TestCase):
    def test_combine_reports_mean_sd_and_range(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = _write(root / "a", {("d", "m1"): 0.2})
            b = _write(root / "b", {("d", "m1"): 0.4})
            rows = combine(load_runs([a, b]), ("boundary_fraction",))
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertAlmostEqual(row["boundary_fraction_ratio_mean"], 0.6)
            self.assertAlmostEqual(row["boundary_fraction_ratio_min"], 0.4)
            self.assertAlmostEqual(row["boundary_fraction_ratio_max"], 0.8)
            self.assertAlmostEqual(row["boundary_fraction_ratio_sd"], 0.2 * 2 / 2**0.5)

    def test_only_configurations_in_every_run_are_kept(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = _write(root / "a", {("d", "m1"): 0.2, ("d", "m2"): 0.3})
            b = _write(root / "b", {("d", "m1"): 0.4})
            self.assertEqual(len(combine(load_runs([a, b]), ("boundary_fraction",))), 1)

    def test_ordering_consistency_flags_a_swap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = _write(root / "a", {("d", "m1"): 0.2, ("d", "m2"): 0.3})
            b = _write(root / "b", {("d", "m1"): 0.4, ("d", "m2"): 0.3})
            result = ordering_consistency(
                load_runs([a, b]), metric="boundary_fraction", suffix="ratio", method="kmeans", params="k=3"
            )
            orders, same = result["d"]
            self.assertEqual(orders, [["m1", "m2"], ["m2", "m1"]])
            self.assertFalse(same)

    def test_cli_writes_csv_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = _write(root / "a", {("d", "m1"): 0.2, ("d", "m2"): 0.3})
            b = _write(root / "b", {("d", "m1"): 0.25, ("d", "m2"): 0.35})
            out = root / "out"
            subprocess.run(
                [sys.executable, "experiments/token_metrics_seeds.py", str(a), str(b),
                 "--out-dir", str(out), "--metrics", "boundary_fraction"],
                check=True, capture_output=True, text=True,
            )
            self.assertTrue((out / "seed_summary.csv").is_file())
            self.assertIn("same in every seed", (out / "seed_summary.md").read_text())


if __name__ == "__main__":
    unittest.main()
