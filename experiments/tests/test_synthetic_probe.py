from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.synthetic_probe import (
    FAMILY_ID,
    IMAGE_FAMILIES,
    VIDEO_FAMILIES,
    _rng,
    distinct_colors,
    make_image,
    make_video,
)

SIZE = 64


def _image(family: str, idx: int = 0, seed: int = 1):
    return make_image(family, _rng(seed, family, idx), idx, SIZE)


def _video(family: str, idx: int = 0, seed: int = 1):
    return make_video(family, _rng(seed, family, idx), idx, SIZE)


class RegistryTest(unittest.TestCase):
    def test_every_family_has_a_unique_id(self) -> None:
        families = IMAGE_FAMILIES + VIDEO_FAMILIES
        self.assertEqual(set(FAMILY_ID), set(families))
        self.assertEqual(len(set(FAMILY_ID.values())), len(families))


class DeterminismTest(unittest.TestCase):
    def test_same_seed_same_pixels_and_different_seed_differs(self) -> None:
        for family in IMAGE_FAMILIES:
            a, _, _ = _image(family, 3, seed=5)
            b, _, _ = _image(family, 3, seed=5)
            c, _, _ = _image(family, 3, seed=6)
            np.testing.assert_array_equal(a, b)
            if family not in ("uniform",):  # a colour can coincide only by chance
                self.assertFalse(np.array_equal(a, c), family)


class ImageFamiliesTest(unittest.TestCase):
    def test_shapes_and_dtype(self) -> None:
        for family in IMAGE_FAMILIES:
            image, _, _ = _image(family)
            self.assertEqual(image.shape, (SIZE, SIZE, 3), family)
            self.assertEqual(image.dtype, np.uint8, family)

    def test_uniform_has_one_colour(self) -> None:
        image, _, params = _image("uniform")
        self.assertEqual(len(np.unique(image.reshape(-1, 3), axis=0)), 1)
        self.assertEqual(image[0, 0].tolist(), params["color"])

    def test_noise_blocks_is_piecewise_constant_at_the_requested_scale(self) -> None:
        for idx, cells in enumerate((2, 4, 8, 16, 32, 64)):
            image, _, params = _image("noise_blocks", idx)
            self.assertEqual(params["cells"], cells)
            step = SIZE // cells
            blocks = image.reshape(cells, step, cells, step, 3)
            self.assertTrue((blocks == blocks[:, :1, :, :1, :]).all(), cells)

    def test_checker_alternates_between_neighbouring_cells(self) -> None:
        for idx, cells in enumerate((2, 4, 8, 16, 32)):
            image, _, params = _image("checker", idx)
            self.assertEqual(params["cells"], cells)
            step = SIZE // cells
            cell_colors = image[::step, ::step]
            self.assertTrue((cell_colors[:, :-1] != cell_colors[:, 1:]).any(axis=-1).all(), cells)
            self.assertTrue((cell_colors[:-1] != cell_colors[1:]).any(axis=-1).all(), cells)

    def test_regions_mask_matches_flat_colours_and_count(self) -> None:
        for idx in range(4):
            image, mask, params = _image("regions", idx)
            labels = np.unique(mask)
            self.assertLessEqual(len(labels), params["k"])
            for label in labels:
                self.assertEqual(len(np.unique(image[mask == label].reshape(-1, 3), axis=0)), 1)

    def test_shapes_mask_has_background_and_flat_objects(self) -> None:
        image, mask, params = _image("shapes", 2)
        self.assertEqual(params["count"], 3)
        self.assertIn(0, np.unique(mask))
        for label in np.unique(mask):
            self.assertEqual(len(np.unique(image[mask == label].reshape(-1, 3), axis=0)), 1)

    def test_only_regions_and_shapes_have_masks(self) -> None:
        for family in IMAGE_FAMILIES:
            _, mask, _ = _image(family)
            self.assertEqual(mask is not None, family in ("regions", "shapes"), family)

    def test_distinct_colors_respects_min_distance(self) -> None:
        colors = distinct_colors(np.random.default_rng(0), 5).astype(float)
        for i in range(5):
            for j in range(i):
                self.assertGreaterEqual(np.linalg.norm(colors[i] - colors[j]), 90.0)


class VideoFamiliesTest(unittest.TestCase):
    def test_shapes_and_frame_count(self) -> None:
        for family in VIDEO_FAMILIES:
            video, _ = _video(family)
            self.assertEqual(video.shape, (32, SIZE, SIZE, 3), family)

    def test_static_families_repeat_one_frame(self) -> None:
        for family in ("static_pattern", "noise_static"):
            video, _ = _video(family)
            self.assertTrue((video == video[0]).all(), family)

    def test_noise_temporal_changes_every_frame(self) -> None:
        video, _ = _video("noise_temporal")
        self.assertTrue(all(not np.array_equal(video[i], video[i + 1]) for i in range(31)))

    def test_cut_is_constant_before_and_after_the_cut_only(self) -> None:
        video, params = _video("cut")
        c = params["cut_frame"]
        self.assertTrue((video[:c] == video[0]).all())
        self.assertTrue((video[c:] == video[c]).all())
        self.assertFalse(np.array_equal(video[c - 1], video[c]))

    def test_flicker_is_spatially_uniform_and_changes_in_time(self) -> None:
        for idx in (0, 1):
            video, _ = _video("flicker", idx)
            self.assertTrue((video == video[:, :1, :1, :]).all())
            self.assertFalse(np.array_equal(video[0], video[-1]))

    def test_moving_shape_moves_faster_at_higher_speed(self) -> None:
        def changed(idx: int) -> float:
            fractions = []
            for seed in range(6):
                video, params = _video("moving_shape", idx, seed=seed)
                fractions.append((video[0] != video[8]).any(axis=-1).mean())
            return float(np.mean(fractions))

        self.assertLess(changed(0), changed(1))
        self.assertLess(changed(1), changed(2))


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "needs ffmpeg")
class EndToEndCliTest(unittest.TestCase):
    def test_cli_writes_files_masks_and_manifest_and_is_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("a", "b"):
                subprocess.run(
                    [sys.executable, "scripts/make_synthetic_probe.py", "--out-dir", str(root / name),
                     "--per-family", "2", "--size", "32", "--workers", "2", "--seed", "7"],
                    check=True, capture_output=True, text=True,
                )
            out = root / "a"
            records = [json.loads(line) for line in (out / "manifest.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 2 * (len(IMAGE_FAMILIES) + len(VIDEO_FAMILIES)))
            self.assertEqual(len(list((out / "image").rglob("*.png"))), 2 * len(IMAGE_FAMILIES))
            self.assertEqual(len(list((out / "video").rglob("*.mkv"))), 2 * len(VIDEO_FAMILIES))
            # Masks live beside, not inside, the input directories.
            self.assertEqual(len(list((out / "masks").rglob("*.png"))), 2 * 2)
            self.assertEqual(list((out / "image").rglob("mask*")), [])
            # Lossless round trip of a video and a reproducible image.
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                 "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0",
                 str(out / "video" / "static_pattern" / "static_pattern_0000.mkv")],
                check=True, capture_output=True, text=True,
            )
            self.assertEqual(probe.stdout.strip(), "32")
            image_a = np.asarray(Image.open(out / "image" / "checker" / "checker_0001.png"))
            image_b = np.asarray(Image.open(root / "b" / "image" / "checker" / "checker_0001.png"))
            np.testing.assert_array_equal(image_a, image_b)
            self.assertEqual((out / "manifest.jsonl").read_text(), (root / "b" / "manifest.jsonl").read_text().replace(str(root / "b"), str(out)))


if __name__ == "__main__":
    unittest.main()
