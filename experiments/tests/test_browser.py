from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from visualization.serve import DEFAULT_PORT, discover_experiments, parse_args


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
                        "video": None,
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
            self.assertIsNone(discovered[0]["cells"][0]["video"])

    def test_default_and_custom_ports(self) -> None:
        self.assertEqual(parse_args([]).port, DEFAULT_PORT)
        self.assertEqual(parse_args(["--port", "49127"]).port, 49127)


if __name__ == "__main__":
    unittest.main()
