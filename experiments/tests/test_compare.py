from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from experiments.artifacts import SCHEMA_NAME
from experiments.catalog import media_specs, model_specs
from experiments.compare import resolution_for, run_pair, selected_pairs


class ComparisonTest(unittest.TestCase):
    def test_native_resolution_comes_from_catalog(self) -> None:
        spec = model_specs()["vjepa2-1-vitb"]
        self.assertEqual(resolution_for(spec, "native"), 384)

    def test_matched_pairing_uses_domain_examples(self) -> None:
        args = argparse.Namespace(
            models=["ijepa", "echojepa"],
            inputs=None,
            pairing="matched",
        )
        pairs = selected_pairs(args)
        self.assertEqual(
            [(model.name, media.name) for model, media in pairs],
            [("ijepa", "dog"), ("echojepa", "echonet")],
        )

    def test_completed_visualization_resumes_without_extraction(self) -> None:
        model = model_specs()["ijepa"]
        media = media_specs()["dog"]
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "comparison"
            pair_dir = run_dir / media.name / model.name
            pair_dir.mkdir(parents=True)
            Image.new("RGB", (8, 8), color=(32, 64, 96)).save(
                pair_dir / "visualization.png"
            )
            metadata = {
                "schema": SCHEMA_NAME,
                "figure": "visualization.png",
                "video": None,
                "video_expected": False,
                "grid_shape": [2, 2],
                "pca_explained_variance": [0.5, 0.3, 0.2],
                "extraction": {
                    "token_shape": [1, 4, 8],
                    "token_dtype": "<f4",
                    "pooled_shape": [1, 8],
                    "pooled_dtype": "<f4",
                },
            }
            (pair_dir / "metadata.json").write_text(json.dumps(metadata))

            with mock.patch(
                "experiments.compare.extract_one",
                side_effect=AssertionError("extraction should not run"),
            ):
                record = run_pair(
                    model,
                    media,
                    first_frames={},
                    run_dir=run_dir,
                    resolution_value="224",
                    frames=16,
                    device="cpu",
                    precision="float32",
                    force=False,
                    keep_embeddings=False,
                )

            self.assertEqual(record["status"], "reused")
            self.assertIsNone(record["embeddings"])


if __name__ == "__main__":
    unittest.main()
