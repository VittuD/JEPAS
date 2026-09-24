from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from experiments.artifacts import write_embeddings
from experiments.token_metrics_null import geometry_from_embeddings


class GeometryFromEmbeddingsTest(unittest.TestCase):
    def test_reads_token_count_dim_and_grid_from_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embeddings.h5"
            write_embeddings(
                path,
                __import__("numpy").zeros((2, 16, 4), dtype="float16"),
                metadata={"model": "ijepa", "input_kind": "image", "settings": {"frames": None}},
            )
            token_count, dim, grid, input_kind, frames = geometry_from_embeddings(path)
            self.assertEqual((token_count, dim, grid, input_kind, frames), (16, 4, (4, 4), "image", None))


class EndToEndCliTest(unittest.TestCase):
    def test_writes_null_summary_matching_draws_times_configs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            embeddings = root / "embeddings.h5"
            write_embeddings(
                embeddings,
                __import__("numpy").zeros((1, 16, 4), dtype="float16"),
                metadata={"model": "ijepa", "input_kind": "image", "settings": {"frames": None}},
            )
            out_dir = root / "null"
            result = subprocess.run(
                [
                    sys.executable,
                    "experiments/token_metrics_null.py",
                    "--model",
                    "ijepa",
                    "--embeddings",
                    str(embeddings),
                    "--out-dir",
                    str(out_dir),
                    "--draws",
                    "3",
                    "--cluster-k",
                    "2",
                    "3",
                    "--workers",
                    "2",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("wrote", result.stdout)
            with (out_dir / "null_summary.csv").open() as f:
                rows = list(csv.DictReader(f))
            # 3 draws * 2 methods (kmeans, gmm) * cluster_k in (2, 3) = 12 rows
            self.assertEqual(len(rows), 12)
            self.assertEqual({row["draw_index"] for row in rows}, {"0", "1", "2"})


if __name__ == "__main__":
    unittest.main()
