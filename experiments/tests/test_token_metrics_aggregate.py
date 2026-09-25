from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from experiments.token_metrics_aggregate import (
    add_null_normalization,
    aggregate_rows,
    compare_pair,
    enrich_row,
    infer_dataset,
    infer_modality,
    match_key,
)


def _row(**overrides: Any) -> dict[str, str]:
    base = {
        "model": "modelA",
        "dataset_index": "0",
        "sample_id": "modelA_00000",
        "source": "clip_0.mp4",
        "method": "kmeans",
        "params": "k=2",
        "n_clusters": "2",
        "noise_count": "0",
        "cluster_entropy": "0.9",
        "boundary_fraction": "0.3",
        "silhouette": "0.1",
        "calinski_harabasz": "100.0",
        "davies_bouldin": "2.0",
        "counts": json.dumps({"0": 10, "1": 6}),
        "proportions": json.dumps({"0": 0.625, "1": 0.375}),
        "connected_components": json.dumps({"0": 2, "1": 3}),
        "pca_explained_variance": json.dumps([0.5, 0.2, 0.1]),
        "adjacent_cosine_distance": "0.4",
        "adjacent_cosine_distance_sd": "0.05",
        "adjacent_cosine_similarity": "0.6",
    }
    base.update({k: str(v) for k, v in overrides.items()})
    return base


class InferDatasetModalityTest(unittest.TestCase):
    def test_infers_known_dataset_fragments(self) -> None:
        cases = {
            "/shared/datasets/classification/imagenet/val/n01/x.JPEG": "imagenet",
            "/shared/datasets/medical/rsna-pneumonia-detection-challenge/stage_2_test_images_png/a.png": "rsna",
            "/shared/datasets/detection/coco/val2017/000000000139.jpg": "coco",
            "/x/datasets_derived/synthetic_v1_seed42/image/checker/checker_0003.png": "syn-checker",
            "/x/datasets_derived/synthetic_v1_seed7/video/cut/cut_0000.mkv": "syn-cut",
            "/shared/datasets/video/diving48/rgb/clip.mp4": "diving48",
            "/leonardo/prod/data/ai/kinetics/kinetics400/256/val/abseiling/x.mp4": "kinetics",
            "/leonardo/prod/data/ai/epic-kitchens/epic-kitchens-100/P01/videos/P01_01.MP4": "epic-kitchens",
            "/shared/eidos-datasets/data-registry/medical/cardio/catdx/StanfordDataset/Videos/x.avi": "echonet",
        }
        for source, expected in cases.items():
            self.assertEqual(infer_dataset(source), expected)

    def test_unknown_source_falls_back(self) -> None:
        self.assertEqual(infer_dataset("/some/other/path/file.jpg"), "unknown")

    def test_infers_modality_from_catalog(self) -> None:
        self.assertEqual(infer_modality("ijepa"), "image")
        self.assertEqual(infer_modality("radjepa"), "image")
        self.assertEqual(infer_modality("vjepa2-vitl"), "video")
        self.assertEqual(infer_modality("echojepa"), "video")

    def test_unknown_model_falls_back(self) -> None:
        self.assertEqual(infer_modality("not-a-real-model"), "unknown")


class NullNormalizationTest(unittest.TestCase):
    def test_adds_z_and_ratio_joined_by_model_method_params(self) -> None:
        aggregates = [
            {"model": "ijepa", "dataset": "imagenet", "method": "kmeans", "params": "k=4", "boundary_fraction_mean": 0.6},
            {"model": "ijepa", "dataset": "rsna", "method": "kmeans", "params": "k=4", "boundary_fraction_mean": 0.4},
        ]
        null_aggregates = [
            {"model": "ijepa", "method": "kmeans", "params": "k=4", "boundary_fraction_mean": 0.5, "boundary_fraction_sd": 0.1},
        ]
        result = add_null_normalization(aggregates, null_aggregates, metrics=("boundary_fraction",))
        # Both dataset rows for the same model join to the same (dataset-free) null baseline.
        self.assertAlmostEqual(result[0]["boundary_fraction_null_mean"], 0.5)
        self.assertAlmostEqual(result[0]["boundary_fraction_z"], 1.0)  # (0.6-0.5)/0.1
        self.assertAlmostEqual(result[0]["boundary_fraction_ratio"], 1.2)  # 0.6/0.5
        self.assertAlmostEqual(result[1]["boundary_fraction_z"], -1.0)  # (0.4-0.5)/0.1

    def test_missing_null_baseline_leaves_row_unmodified(self) -> None:
        aggregates = [{"model": "unmatched", "method": "kmeans", "params": "k=4", "boundary_fraction_mean": 0.6}]
        result = add_null_normalization(aggregates, [], metrics=("boundary_fraction",))
        self.assertNotIn("boundary_fraction_z", result[0])


class EnrichRowTest(unittest.TestCase):
    def test_derives_connected_component_summaries_and_pca_components(self) -> None:
        enriched = enrich_row(_row(), Path("summary.csv"))
        self.assertEqual(enriched["total_connected_components"], 5.0)
        self.assertEqual(enriched["extra_connected_components"], 3.0)  # (2-1) + (3-1)
        self.assertEqual(enriched["max_connected_components"], 3.0)
        self.assertAlmostEqual(enriched["pca_pc1"], 0.5)
        self.assertAlmostEqual(enriched["pca_pc2"], 0.2)
        self.assertAlmostEqual(enriched["pca_pc3"], 0.1)
        self.assertAlmostEqual(enriched["pca_top3_sum"], 0.8)

    def test_adds_dataset_and_modality(self) -> None:
        enriched = enrich_row(_row(model="ijepa", source="/x/imagenet/val/a.jpg"), Path("summary.csv"))
        self.assertEqual(enriched["dataset"], "imagenet")
        self.assertEqual(enriched["modality"], "image")


class AggregateRowsTest(unittest.TestCase):
    def test_groups_and_summarizes_by_model(self) -> None:
        rows = [
            enrich_row(_row(model="modelA", boundary_fraction="0.2"), Path("a.csv")),
            enrich_row(_row(model="modelA", boundary_fraction="0.4"), Path("a.csv")),
            enrich_row(_row(model="modelB", boundary_fraction="0.9"), Path("b.csv")),
        ]
        aggregates = aggregate_rows(rows, group_by=("model",), metrics=("boundary_fraction",))
        by_model = {row["model"]: row for row in aggregates}
        self.assertAlmostEqual(by_model["modelA"]["boundary_fraction_mean"], 0.3)
        self.assertEqual(by_model["modelA"]["boundary_fraction_n"], 2)
        self.assertAlmostEqual(by_model["modelB"]["boundary_fraction_mean"], 0.9)


class ComparePairTest(unittest.TestCase):
    def test_matches_rows_by_source_not_by_index(self) -> None:
        # Same clip, different dataset_index between the two models -- match_key
        # must join on source, not on index, or this pair would be silently missed.
        baseline = enrich_row(
            _row(model="vjepa2-vitl", dataset_index="7", source="clip_x.mp4", boundary_fraction="0.5"),
            Path("a.csv"),
        )
        other = enrich_row(
            _row(model="vjepa2-1-vitb", dataset_index="3", source="clip_x.mp4", boundary_fraction="0.3"),
            Path("b.csv"),
        )
        comparisons = compare_pair(
            [baseline, other],
            baseline="vjepa2-vitl",
            other="vjepa2-1-vitb",
            metrics=("boundary_fraction",),
        )
        self.assertEqual(len(comparisons), 1)
        row = comparisons[0]
        self.assertEqual(row["n_matched"], 1)
        self.assertAlmostEqual(row["boundary_fraction_baseline_mean"], 0.5)
        self.assertAlmostEqual(row["boundary_fraction_other_mean"], 0.3)
        self.assertAlmostEqual(row["boundary_fraction_delta_mean"], -0.2)

    def test_no_overlap_returns_empty_with_warning(self) -> None:
        baseline = enrich_row(_row(model="modelA", source="only_in_a.mp4"), Path("a.csv"))
        other = enrich_row(_row(model="modelB", source="only_in_b.mp4"), Path("b.csv"))
        comparisons = compare_pair(
            [baseline, other], baseline="modelA", other="modelB", metrics=("boundary_fraction",)
        )
        self.assertEqual(comparisons, [])

    def test_match_key_uses_source_method_params(self) -> None:
        row = _row(source="clip.mp4", method="gmm", params="k=4,cov=diag")
        self.assertEqual(match_key(row), ("clip.mp4", "gmm", "k=4,cov=diag"))


class CliEndToEndTest(unittest.TestCase):
    def test_writes_aggregates_and_comparisons(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fieldnames = list(_row())

            for model, source, boundary in (
                ("vjepa2-vitl", "clip_0.mp4", "0.5"),
                ("vjepa2-1-vitb", "clip_0.mp4", "0.3"),
            ):
                csv_path = root / model / "summary.csv"
                csv_path.parent.mkdir(parents=True)
                with csv_path.open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerow(_row(model=model, source=source, boundary_fraction=boundary))

            out_dir = root / "aggregate"
            result = subprocess.run(
                [
                    sys.executable,
                    "experiments/token_metrics_aggregate.py",
                    str(root),
                    "--compare",
                    "vjepa2-vitl",
                    "vjepa2-1-vitb",
                    "--out-dir",
                    str(out_dir),
                    "--metrics",
                    "boundary_fraction",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("Matched Comparisons", result.stdout)
            self.assertTrue((out_dir / "aggregates.csv").exists())
            self.assertTrue((out_dir / "comparisons.csv").exists())
            comparisons = json.loads((out_dir / "comparisons.json").read_text())
            self.assertEqual(len(comparisons), 1)
            self.assertEqual(comparisons[0]["n_matched"], 1)


if __name__ == "__main__":
    unittest.main()
