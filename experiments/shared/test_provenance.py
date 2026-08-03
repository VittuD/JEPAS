from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.shared.provenance import (
    ROOT_DIR,
    portable_path,
    resolve_portable_path,
)


class PortablePathTest(unittest.TestCase):
    def test_repository_path_is_relative_and_round_trips(self) -> None:
        path = ROOT_DIR / "weights" / "example.pt"

        serialized = portable_path(path)

        self.assertEqual(serialized, "weights/example.pt")
        self.assertEqual(resolve_portable_path(serialized), path)

    def test_external_path_remains_absolute(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.pt"

            serialized = portable_path(path)

            self.assertEqual(serialized, str(path.resolve()))
            self.assertEqual(resolve_portable_path(serialized), path.resolve())

    def test_repository_symlink_keeps_its_portable_launcher_path(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT_DIR) as directory:
            launcher = Path(directory) / "python"
            launcher.symlink_to("/usr/bin/python3")

            serialized = portable_path(launcher)

            self.assertFalse(Path(serialized).is_absolute())
            self.assertEqual(resolve_portable_path(serialized), launcher.resolve())


if __name__ == "__main__":
    unittest.main()
