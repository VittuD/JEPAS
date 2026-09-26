from __future__ import annotations

import unittest

import numpy as np

from experiments.position_lock import analyse, family_of

GRID = (2, 8, 8)


def _tokens(fn, samples=16, dim=6, seed=0):
    rng = np.random.default_rng(seed)
    return np.stack([fn(rng) for _ in range(samples)]).astype(np.float32)


class PositionLockTest(unittest.TestCase):
    def test_white_noise_has_no_position_lock_and_no_lattice(self) -> None:
        stats = analyse(_tokens(lambda r: r.normal(size=(128, 6))), GRID)
        self.assertLess(stats["pos_variance_fraction"], 0.2)
        self.assertLess(abs(stats["split_half_cosine"]), 0.3)
        self.assertLess(stats["nyquist_ratio"], 1.5)

    def test_a_fixed_checkerboard_plus_noise_is_position_locked_with_period_two(self) -> None:
        board = np.indices((2, 8, 8)).sum(axis=0) % 2 * 2.0 - 1.0
        pattern = np.repeat(board.reshape(128, 1), 6, axis=1)
        stats = analyse(_tokens(lambda r: pattern * 3 + r.normal(size=(128, 6))), GRID)
        self.assertGreater(stats["pos_variance_fraction"], 0.8)
        self.assertGreater(stats["split_half_cosine"], 0.9)
        self.assertGreater(stats["nyquist_ratio"], 3)
        self.assertGreater(stats["nyquist_share"], 0.7)
        self.assertAlmostEqual(stats["dominant_period"], 1 / np.hypot(0.5, 0.5), places=2)

    def test_family_of(self) -> None:
        self.assertEqual(family_of("/x/synthetic_v1_seed42/video/noise_static/a.mkv"), "noise_static")
        self.assertEqual(family_of("/x/kinetics/a.mp4"), "all")


if __name__ == "__main__":
    unittest.main()
