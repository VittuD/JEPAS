from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from experiments.compare import load_maps, resolution_for, selected_pairs
from experiments.catalog import model_specs
from experiments.visualize import (
    _prepare_tokens,
    _select_2d_tokens,
    _token_pca_maps,
)


class LoadMapsTest(unittest.TestCase):
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

    def test_uses_first_temporal_slice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.default_rng(7)
            embedding = rng.normal(size=(1, 8, 6)).astype(np.float32)
            embedding[:, 4:, :] += np.asarray(
                [4.0, -3.0, 2.0, 0.0, 1.0, -2.0],
                dtype=np.float32,
            )
            embedding_path = root / "tokens.npy"
            np.save(embedding_path, embedding)

            preview_path = root / "input_frame.jpg"
            Image.new("RGB", (8, 8), color=(32, 64, 96)).save(preview_path)
            record = {
                "tokens": str(embedding_path),
                "spatial_grid": [2, 2],
                "temporal_grid": 2,
                "reference": str(preview_path),
            }
            _, rendered_pca, _, _ = load_maps(record)

            tokens, grid = _prepare_tokens(
                embedding,
                sample_index=None,
                grid_shape=(2, 2, 2),
            )
            tokens = F.layer_norm(
                torch.from_numpy(tokens),
                (tokens.shape[-1],),
            ).numpy()
            selected, grid_2d, _ = _select_2d_tokens(
                tokens,
                grid,
                slice_index=0,
            )
            expected_pca, _, _ = _token_pca_maps(selected, grid_2d)

            np.testing.assert_allclose(rendered_pca, expected_pca)


if __name__ == "__main__":
    unittest.main()
