from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.artifacts import (
    SCHEMA_NAME,
    cleanup_embeddings,
    valid_visualization,
    write_embeddings,
)


class CleanupTest(unittest.TestCase):
    @staticmethod
    def _metadata() -> dict[str, object]:
        return {
            "schema": SCHEMA_NAME,
            "figure": "visualization.png",
            "frames": [],
            "temporal_expected": False,
            "grid_shape": [2, 2],
            "pca_explained_variance": [0.5, 0.3, 0.2],
            "extraction": {
                "token_shape": [1, 4, 8],
                "token_dtype": "<f4",
                "pooled_shape": [1, 8],
                "pooled_dtype": "<f4",
            },
        }

    def _embeddings(self, directory: Path) -> Path:
        path = directory / "embeddings.h5"
        write_embeddings(
            path,
            np.arange(32, dtype=np.float32).reshape(1, 4, 8),
            metadata={"model": "test"},
        )
        return path

    def test_cleanup_refuses_invalid_visualization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            embeddings = self._embeddings(directory)
            with self.assertRaisesRegex(ValueError, "Refusing to delete"):
                cleanup_embeddings(directory)
            self.assertTrue(embeddings.exists())

    def test_cleanup_is_guarded_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            embeddings = self._embeddings(directory)
            Image.new("RGB", (8, 8)).save(directory / "visualization.png")
            (directory / "metadata.json").write_text(
                json.dumps(self._metadata())
            )
            self.assertTrue(cleanup_embeddings(directory))
            self.assertFalse(embeddings.exists())
            self.assertFalse(cleanup_embeddings(directory))

    def test_declared_temporal_frames_must_exist_before_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            embeddings = self._embeddings(directory)
            Image.new("RGB", (8, 8)).save(directory / "visualization.png")
            metadata = self._metadata()
            metadata.update(
                {
                    "frames": ["visualization_frame_000.png"],
                    "temporal_expected": True,
                }
            )
            (directory / "metadata.json").write_text(
                json.dumps(metadata)
            )
            with self.assertRaises(ValueError):
                cleanup_embeddings(directory)
            self.assertTrue(embeddings.exists())

    def test_visualizer_reads_hdf5_and_writes_valid_static_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            embeddings = directory / "embeddings.h5"
            tokens = np.arange(32, dtype=np.float32).reshape(1, 4, 8)
            tokens[:, :, 1::2] *= -0.5
            write_embeddings(
                embeddings,
                tokens,
                metadata={
                    "model": "test",
                    "input": "input.jpg",
                    "checkpoint": "checkpoint.pt",
                    "settings": {"precision": "fp32"},
                },
            )
            Image.new("RGB", (8, 8), color=(10, 20, 30)).save(
                directory / "input.jpg"
            )
            subprocess.run(
                [
                    sys.executable,
                    "experiments/visualize.py",
                    str(embeddings),
                    "--out-dir",
                    str(directory),
                    "--grid-shape",
                    "2x2",
                    "--image",
                    str(directory / "input.jpg"),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertTrue(
                valid_visualization(directory, require_temporal=False)
            )
            with Image.open(directory / "visualization.png") as visualization:
                self.assertEqual(visualization.width, visualization.height)
            self.assertFalse((directory / ".matplotlib").exists())

    def test_visualizer_writes_numbered_temporal_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            embeddings = directory / "embeddings.h5"
            tokens = np.arange(64, dtype=np.float32).reshape(1, 8, 8)
            write_embeddings(
                embeddings,
                tokens,
                metadata={"model": "test", "settings": {}},
            )
            Image.new("RGB", (8, 8), color=(10, 20, 30)).save(
                directory / "input.jpg"
            )
            subprocess.run(
                [
                    sys.executable,
                    "experiments/visualize.py",
                    str(embeddings),
                    "--out-dir",
                    str(directory),
                    "--grid-shape",
                    "2x2x2",
                    "--image",
                    str(directory / "input.jpg"),
                    "--animate",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            metadata = json.loads((directory / "metadata.json").read_text())
            self.assertEqual(
                metadata["frames"],
                ["visualization_frame_000.png", "visualization_frame_001.png"],
            )
            self.assertTrue(
                valid_visualization(directory, require_temporal=True)
            )


if __name__ == "__main__":
    unittest.main()
