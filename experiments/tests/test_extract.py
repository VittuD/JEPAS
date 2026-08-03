from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

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
        self.assertEqual(tokens.name, "tokens.npy")
        self.assertEqual(preview.name, "input.jpg")


if __name__ == "__main__":
    unittest.main()
