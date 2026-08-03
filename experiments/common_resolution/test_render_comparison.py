from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from experiments.common_resolution.render_comparison import load_maps
from experiments.visualization.visualize_embedding import (
    _prepare_tokens,
    _select_2d_tokens,
    _token_pca_maps,
)


class LoadMapsTest(unittest.TestCase):
    def test_uses_visualizer_manifest_slice_for_temporal_tokens(self) -> None:
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
            visualization_manifest = root / "visualization_manifest.json"
            visualization_manifest.write_text(json.dumps({"slice_index": "0"}))

            record = {
                "embedding": str(embedding_path),
                "spatial_grid": [2, 2],
                "temporal_grid": 2,
                "figure": str(root / "visualization" / "embedding_visualization.png"),
                "artifacts": {
                    "preview": {"path": str(preview_path)},
                    "visualization_manifest": str(visualization_manifest),
                },
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
