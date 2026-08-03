from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from visualization.catalog import discover_experiments


class BrowserTest(unittest.TestCase):
    def test_discovers_only_visualized_new_schema_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            outputs = Path(temporary)
            experiment = outputs / "comparison_all_384"
            result = experiment / "dog" / "ijepa"
            result.mkdir(parents=True)
            (experiment / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema": "jepas-experiment-v1",
                        "experiment": "All at 384",
                    }
                )
            )
            Image.new("RGB", (8, 8)).save(result / "visualization.png")
            (result / "metadata.json").write_text(
                json.dumps(
                    {
                        "schema": "jepas-experiment-v1",
                        "figure": "visualization.png",
                        "frames": [],
                    }
                )
            )
            extraction_only = experiment / "xray" / "radjepa"
            extraction_only.mkdir(parents=True)
            (extraction_only / "embeddings.h5").touch()
            legacy = outputs / "legacy"
            legacy.mkdir()
            (legacy / "manifest.json").write_text(json.dumps({"results": []}))

            discovered = discover_experiments(outputs)

            self.assertEqual(len(discovered), 1)
            self.assertEqual(discovered[0]["rows"], ["dog"])
            self.assertEqual(discovered[0]["columns"], ["ijepa"])
            self.assertEqual(discovered[0]["results"][0]["frames"], [])
            self.assertEqual(
                discovered[0]["results"][0]["image"],
                result / "visualization.png",
            )

    def test_inference_blueprints_include_visualization_requirements(self) -> None:
        root = Path(__file__).resolve().parents[2]
        for name in ("vjepa2_inference.txt", "vjepa2_inference_cuda.txt"):
            requirements = (root / "requirements" / name).read_text().splitlines()
            self.assertIn("-r visualization.txt", requirements)

    def test_notebook_uses_compact_shared_frame_inspector(self) -> None:
        root = Path(__file__).resolve().parents[2]
        notebook = (root / "visualization" / "browser.py").read_text()

        self.assertIn("mo.ui.run_button(", notebook)
        self.assertIn('label="Refresh experiments"', notebook)
        self.assertIn("refresh_experiments.value", notebook)
        self.assertIn('label="Experiment"', notebook)
        self.assertIn('label="Input"', notebook)
        self.assertIn('label="Frame"', notebook)
        self.assertIn("mo.ui.slider(", notebook)
        self.assertNotIn("mo.carousel(", notebook)
        self.assertNotIn("mo.Html(", notebook)
        self.assertEqual(notebook.count("mo.image("), 1)
        self.assertNotIn("mo.video(", notebook)


if __name__ == "__main__":
    unittest.main()
