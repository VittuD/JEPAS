from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.artifacts import write_embeddings
from experiments.token_metrics import (
    _boundary_fraction,
    _connected_components,
    adjacent_token_stats,
    infer_grid,
)


class InferGridTest(unittest.TestCase):
    def test_image_grid_is_2d_square(self) -> None:
        self.assertEqual(infer_grid(256, input_kind="image", frames=None), (16, 16))

    def test_non_square_image_token_count_rejected(self) -> None:
        with self.assertRaises(ValueError):
            infer_grid(255, input_kind="image", frames=None)

    def test_video_grid_is_3d_tubelets_by_square_spatial(self) -> None:
        # 16 frames / tubelet_size=2 -> 8 tubelets; 2048/8=256 -> 16x16 spatial.
        self.assertEqual(infer_grid(2048, input_kind="video", frames=16), (8, 16, 16))
        self.assertEqual(infer_grid(4608, input_kind="video", frames=16), (8, 24, 24))
        self.assertEqual(infer_grid(1568, input_kind="video", frames=16), (8, 14, 14))

    def test_video_grid_requires_frames(self) -> None:
        with self.assertRaises(ValueError):
            infer_grid(2048, input_kind="video", frames=None)

    def test_unsupported_kind_rejected(self) -> None:
        with self.assertRaises(ValueError):
            infer_grid(256, input_kind="volume", frames=None)


class AdjacentTokenStatsTest(unittest.TestCase):
    def test_identical_tokens_have_zero_distance(self) -> None:
        tokens = np.ones((16, 8), dtype=np.float32)
        stats = adjacent_token_stats(tokens, (4, 4))
        self.assertAlmostEqual(stats["adjacent_cosine_distance"], 0.0, places=5)
        self.assertAlmostEqual(stats["adjacent_cosine_similarity"], 1.0, places=5)

    def test_3d_grid_includes_temporal_axis_neighbors(self) -> None:
        # Two "frames" (tubelets) of otherwise-identical spatial tokens, but the
        # two frames themselves are orthogonal -- a purely-2D (per-frame) metric
        # would report zero distance everywhere; the 3D version must not.
        grid = (2, 2, 2)
        frame_a = np.tile(np.array([1.0, 0.0], dtype=np.float32), (4, 1))
        frame_b = np.tile(np.array([0.0, 1.0], dtype=np.float32), (4, 1))
        tokens = np.concatenate([frame_a, frame_b], axis=0)
        stats = adjacent_token_stats(tokens, grid)
        self.assertGreater(stats["adjacent_cosine_distance"], 0.0)

    def test_2d_grid_matches_original_two_axis_behavior(self) -> None:
        rng = np.random.default_rng(0)
        tokens = rng.normal(size=(16, 8)).astype(np.float32)
        stats = adjacent_token_stats(tokens, (4, 4))
        self.assertIn("adjacent_cosine_distance", stats)
        self.assertGreaterEqual(stats["adjacent_cosine_distance"], 0.0)


class ConnectedComponentsTest(unittest.TestCase):
    def test_uniform_2d_grid_is_one_component(self) -> None:
        labels = np.zeros((4, 4), dtype=int)
        self.assertEqual(_connected_components(labels, 0), 1)

    def test_checkerboard_2d_grid_has_many_components(self) -> None:
        labels = np.indices((4, 4)).sum(axis=0) % 2
        # Each same-label cell only touches opposite-label neighbors (4-conn),
        # so every cell is its own component.
        self.assertEqual(_connected_components(labels, 0), 8)

    def test_uniform_3d_grid_is_one_component(self) -> None:
        labels = np.zeros((2, 3, 3), dtype=int)
        self.assertEqual(_connected_components(labels, 0), 1)

    def test_3d_grid_merges_only_through_temporal_axis(self) -> None:
        # Two spatially-identical frames of label 0, connected only via the
        # tubelet (first) axis -- must still be one component under 6-conn.
        labels = np.zeros((2, 2, 2), dtype=int)
        self.assertEqual(_connected_components(labels, 0), 1)


class BoundaryFractionTest(unittest.TestCase):
    def test_uniform_grid_has_zero_boundary(self) -> None:
        labels = np.zeros((4, 4), dtype=int)
        self.assertEqual(_boundary_fraction(labels), 0.0)

    def test_checkerboard_has_full_boundary(self) -> None:
        labels = np.indices((4, 4)).sum(axis=0) % 2
        self.assertEqual(_boundary_fraction(labels), 1.0)


class EndToEndCliTest(unittest.TestCase):
    def test_writes_summary_for_image_and_video_geometry(self) -> None:
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            embeddings = root / "embeddings.h5"
            n, tokens_per_sample, dim = 3, 16, 4
            tokens = rng.normal(size=(n, tokens_per_sample, dim)).astype(np.float16)
            write_embeddings(
                embeddings,
                tokens,
                metadata={
                    "model": "ijepa",
                    "input_kind": "image",
                    "settings": {"frames": None},
                },
                sources=[f"img_{i}.jpg" for i in range(n)],
            )

            out_dir = root / "token_metrics"
            result = subprocess.run(
                [
                    sys.executable,
                    "experiments/token_metrics.py",
                    "--model",
                    "ijepa",
                    "--embeddings",
                    str(embeddings),
                    "--out-dir",
                    str(out_dir),
                    "--cluster-k",
                    "2",
                    "3",
                    "--workers",
                    "1",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("wrote", result.stdout)

            summary_csv = out_dir / "summary.csv"
            self.assertTrue(summary_csv.exists())
            with summary_csv.open() as f:
                rows = list(csv.DictReader(f))
            # n=3 samples * 2 methods (kmeans, gmm) * cluster_k in (2, 3) = 12 rows
            self.assertEqual(len(rows), 12)
            self.assertEqual({row["model"] for row in rows}, {"ijepa"})
            self.assertEqual({row["source"] for row in rows}, {"img_0.jpg", "img_1.jpg", "img_2.jpg"})
            self.assertTrue((out_dir / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
