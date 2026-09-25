from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.artifacts import write_embeddings
from experiments.compare_bulk_maps import check_alignment, select_indices


class CheckAlignmentTest(unittest.TestCase):
    def test_identical_sources_pass(self) -> None:
        check_alignment({"a": ["x", "y"], "b": ["x", "y"]})

    def test_reordered_sources_fail_and_name_the_index(self) -> None:
        with self.assertRaisesRegex(ValueError, "Sample 0 differs"):
            check_alignment({"a": ["x", "y"], "b": ["y", "x"]})

    def test_length_mismatch_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "samples"):
            check_alignment({"a": ["x", "y"], "b": ["x"]})


class SelectIndicesTest(unittest.TestCase):
    METRICS = {
        # Different scales per model, same ordering of samples 0..4 (0 smoothest).
        "a": {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4, 4: 0.5},
        "b": {0: 10.0, 1: 20.0, 2: 30.0, 3: 40.0, 4: 50.0},
    }

    def test_low_and_high_use_rank_across_models(self) -> None:
        self.assertEqual(select_indices(self.METRICS, pick="low", count=2, seed=0), [0, 1])
        self.assertEqual(select_indices(self.METRICS, pick="high", count=2, seed=0), [4, 3])

    def test_median_is_the_middle_sample(self) -> None:
        self.assertEqual(select_indices(self.METRICS, pick="median", count=1, seed=0), [2])

    def test_first_takes_the_lowest_common_indices(self) -> None:
        self.assertEqual(select_indices(self.METRICS, pick="first", count=3, seed=0), [0, 1, 2])

    def test_random_is_seeded_and_capped(self) -> None:
        first = select_indices(self.METRICS, pick="random", count=3, seed=7)
        self.assertEqual(first, select_indices(self.METRICS, pick="random", count=3, seed=7))
        self.assertEqual(len(select_indices(self.METRICS, pick="random", count=99, seed=7)), 5)

    def test_only_indices_present_for_every_model_are_used(self) -> None:
        metrics = {"a": {0: 0.1, 1: 0.2, 2: 0.3}, "b": {1: 1.0, 2: 2.0}}
        self.assertEqual(select_indices(metrics, pick="low", count=5, seed=0), [1, 2])


class EndToEndCliTest(unittest.TestCase):
    def test_renders_one_figure_per_sample_from_a_bulk_layout(self) -> None:
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = [f"/missing/img_{i}.png" for i in range(4)]
            for model in ("ijepa", "radjepa"):
                model_dir = root / "bulk" / "imagenet" / model
                embeddings = model_dir / "embeddings.h5"
                write_embeddings(
                    embeddings,
                    rng.normal(size=(4, 16, 8)).astype("float16"),
                    metadata={"model": model, "input_kind": "image", "settings": {"frames": None}},
                    sources=sources,
                )
                subprocess.run(
                    [
                        sys.executable, "experiments/token_metrics.py",
                        "--model", model, "--embeddings", str(embeddings),
                        "--out-dir", str(model_dir / "token_metrics"),
                        "--cluster-k", "2", "3", "--workers", "1",
                    ],
                    check=True, capture_output=True, text=True,
                )
            out_dir = root / "maps"
            result = subprocess.run(
                [
                    sys.executable, "experiments/compare_bulk_maps.py",
                    str(root / "bulk"), "imagenet", "--out-dir", str(out_dir),
                    "--pick", "low", "--count", "2", "--k", "3",
                ],
                check=True, capture_output=True, text=True,
            )
            self.assertIn("alignment OK", result.stdout)
            self.assertNotIn("WARNING", result.stderr)
            self.assertEqual(len(list(out_dir.glob("imagenet_*.png"))), 2)


if __name__ == "__main__":
    unittest.main()
