from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from experiments.artifacts import read_embedding_metadata, valid_embeddings, write_embeddings
from experiments.catalog import media_specs, model_specs
from experiments.extract import extraction_command


class ExtractionCommandTest(unittest.TestCase):
    def test_catalog_drives_vjepa_adapter_command(self) -> None:
        spec = model_specs()["vjepa2-vitl"]
        media = media_specs()["diving48"]
        with tempfile.TemporaryDirectory() as directory:
            command, tokens, preview = extraction_command(
                spec,
                weights=spec.weights_path,
                input_path=media.path,
                output_dir=Path(directory),
                device="cuda",
                precision="auto",
                image_size=384,
                frames=16,
                seed=0,
            )

        self.assertIn("experiments/models/vjepa2.py", command)
        self.assertIn("vjepa2-vitl", command)
        self.assertIn("--video", command)
        self.assertIn("384", command)
        self.assertEqual(tokens.name, ".tokens.npy")
        self.assertEqual(preview.name, "input.jpg")

    def test_hdf5_round_trip_is_exact_and_pooling_is_fp32(self) -> None:
        rng = np.random.default_rng(23)
        tokens = rng.normal(size=(2, 7, 5)).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embeddings.h5"
            write_embeddings(
                path,
                tokens,
                metadata={"model": "test", "settings": {"precision": "float32"}},
            )

            with h5py.File(path, "r") as handle:
                stored_tokens = handle["tokens"][...]
                pooled = handle["pooled"][...]
                self.assertEqual(handle["tokens"].compression, "gzip")
                self.assertTrue(handle["tokens"].shuffle)
                self.assertIsNotNone(handle["tokens"].chunks)

            self.assertTrue(valid_embeddings(path))
            np.testing.assert_array_equal(stored_tokens, tokens)
            self.assertEqual(stored_tokens.dtype, tokens.dtype)
            self.assertEqual(pooled.dtype, np.dtype(np.float32))
            np.testing.assert_array_equal(
                pooled,
                tokens.mean(axis=-2, dtype=np.float32),
            )
            metadata = read_embedding_metadata(path)
            self.assertEqual(metadata["token_shape"], [2, 7, 5])
            self.assertEqual(metadata["token_dtype"], tokens.dtype.str)


if __name__ == "__main__":
    unittest.main()
