from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from experiments.catalog import model_specs
from experiments.extract_batch import batch_command, resolve_kind, select_inputs


class ResolveKindTest(unittest.TestCase):
    def test_single_kind_model_needs_no_override(self) -> None:
        self.assertEqual(resolve_kind(model_specs()["ijepa"], None), "image")

    def test_multi_kind_model_requires_explicit_kind(self) -> None:
        with self.assertRaises(ValueError):
            resolve_kind(model_specs()["vjepa2-vitl"], None)
        self.assertEqual(resolve_kind(model_specs()["vjepa2-vitl"], "video"), "video")

    def test_kind_must_be_supported_by_the_model(self) -> None:
        with self.assertRaises(ValueError):
            resolve_kind(model_specs()["ijepa"], "video")


class BatchCommandTest(unittest.TestCase):
    def test_image_model_command_omits_list_kind(self) -> None:
        spec = model_specs()["ijepa"]
        with tempfile.TemporaryDirectory() as directory:
            command = batch_command(
                spec,
                weights=spec.weights_path,
                input_list=Path(directory) / "input_list.txt",
                kind="image",
                adapter_output=Path(directory) / ".tokens.npy",
                device="cuda",
                precision="auto",
                image_size=None,
                frames=16,
                batch_size=32,
                seed=0,
            )
        self.assertIn("experiments/models/ijepa.py", command)
        self.assertIn("--input-list", command)
        self.assertNotIn("--list-kind", command)
        self.assertIn("--batch-size", command)

    def test_video_model_command_includes_list_kind_and_frames(self) -> None:
        spec = model_specs()["vjepa2-vitl"]
        with tempfile.TemporaryDirectory() as directory:
            command = batch_command(
                spec,
                weights=spec.weights_path,
                input_list=Path(directory) / "input_list.txt",
                kind="video",
                adapter_output=Path(directory) / ".tokens.npy",
                device="cuda",
                precision="auto",
                image_size=None,
                frames=16,
                batch_size=8,
                seed=0,
            )
        self.assertIn("--list-kind", command)
        self.assertIn("video", command)
        self.assertIn("--frames", command)


class CliDryRunTest(unittest.TestCase):
    def test_dry_run_reports_sample_size_without_touching_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(6):
                (root / f"img_{index}.jpg").write_bytes(b"not a real image")
            result = subprocess.run(
                [
                    sys.executable,
                    "experiments/extract_batch.py",
                    "--model",
                    "ijepa",
                    "--input",
                    str(root),
                    "--n",
                    "3",
                    "--output-dir",
                    str(root / "out"),
                    "--device",
                    "cuda",
                    "--dry-run",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        self.assertIn("n=3 sampled from 6 image candidates", result.stdout)
        self.assertIn("experiments/models/ijepa.py", result.stdout)

    def test_insufficient_candidates_fail_fast(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "img_0.jpg").write_bytes(b"not a real image")
            result = subprocess.run(
                [
                    sys.executable,
                    "experiments/extract_batch.py",
                    "--model",
                    "ijepa",
                    "--input",
                    str(root),
                    "--n",
                    "5",
                    "--output-dir",
                    str(root / "out"),
                    "--device",
                    "cuda",
                    "--dry-run",
                ],
                capture_output=True,
                text=True,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Only 1", result.stderr)


if __name__ == "__main__":
    unittest.main()


class SelectInputsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.candidates = [Path(f"/x/{i:03d}.mp4") for i in range(50)]

    def test_all_valid_matches_plain_seeded_choice(self) -> None:
        import numpy as np

        expected = [self.candidates[i] for i in np.sort(np.random.default_rng(7).choice(50, size=10, replace=False))]
        selected, skipped = select_inputs(self.candidates, 10, 7, lambda p: True)
        self.assertEqual(selected, expected)
        self.assertEqual(skipped, [])

    def test_corrupt_inputs_are_replaced_and_reported(self) -> None:
        base, _ = select_inputs(self.candidates, 10, 7, lambda p: True)
        bad = set(base[:3])
        selected, skipped = select_inputs(self.candidates, 10, 7, lambda p: p not in bad)
        self.assertEqual(len(selected), 10)
        self.assertEqual(set(skipped), bad)
        self.assertTrue(bad.isdisjoint(selected))
        self.assertEqual(selected, sorted(selected))
        again, _ = select_inputs(self.candidates, 10, 7, lambda p: p not in bad)
        self.assertEqual(again, selected)

    def test_too_few_readable_inputs_fail(self) -> None:
        with self.assertRaises(ValueError):
            select_inputs(self.candidates, 10, 7, lambda p: int(p.stem) < 5)
