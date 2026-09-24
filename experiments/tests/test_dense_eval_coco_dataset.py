from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools.cocoeval import COCOeval

from experiments.dense_eval.coco_dataset import CocoDetectionDataset


def _write_fake_coco(root: Path, *, orig_size: tuple[int, int]) -> tuple[Path, Path]:
    """One image at a non-square, non-224 original resolution (the exact
    condition that exposed the GT/prediction coordinate-space bug: Cityscapes
    images are 2048x1024, nothing near the 224x224 training resolution)."""
    orig_w, orig_h = orig_size
    images_dir = root / "images"
    images_dir.mkdir()
    Image.new("RGB", (orig_w, orig_h), color=(120, 40, 200)).save(images_dir / "img_0.jpg")

    coco_json = {
        "images": [{"id": 1, "file_name": "img_0.jpg", "width": orig_w, "height": orig_h}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [orig_w * 0.1, orig_h * 0.2, orig_w * 0.3, orig_h * 0.15],
                "area": (orig_w * 0.3) * (orig_h * 0.15),
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "thing"}],
    }
    ann_path = root / "instances.json"
    ann_path.write_text(json.dumps(coco_json))
    return ann_path, images_dir


class RescaleRegressionTest(unittest.TestCase):
    """Regression coverage for the bug this session actually hit: __getitem__
    rescaled boxes for training targets, but pycocotools' COCOeval was handed
    the dataset's *un-rescaled* `.coco` object directly as ground truth,
    silently reporting ~0 mAP regardless of model quality -- indistinguishable
    from "the model hasn't learned anything" without this kind of check."""

    def test_getitem_boxes_match_rescaled_coco_object(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ann_path, images_dir = _write_fake_coco(root, orig_size=(2048, 1024))
            ds = CocoDetectionDataset(annotations_path=ann_path, images_root=images_dir, image_size=224)

            _, target = ds[0]
            coco_ann = ds.coco.loadAnns(ds.coco.getAnnIds(imgIds=ds.image_ids[0]))[0]
            x, y, w, h = coco_ann["bbox"]
            expected_box = torch.tensor([x, y, x + w, y + h])

            self.assertTrue(torch.allclose(target["boxes"][0], expected_box, atol=1e-3))
            # And that box must NOT still be in the original 2048x1024 space.
            self.assertLessEqual(float(target["boxes"][0].max()), 224.0)

    def test_perfect_predictions_score_near_one_map(self) -> None:
        """The decisive check: feed the ground truth straight back as
        'predictions' (score=1.0) and confirm pycocotools reports ~1.0 mAP.
        Before the fix, this scored ~0 because dataset.coco (used as
        COCOeval's ground truth) was still in original-image coordinates
        while the fed-back boxes were in the 224x224 resized space."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ann_path, images_dir = _write_fake_coco(root, orig_size=(2048, 1024))
            ds = CocoDetectionDataset(annotations_path=ann_path, images_root=images_dir, image_size=224)

            _, target = ds[0]
            x1, y1, x2, y2 = target["boxes"][0].tolist()
            results = [
                {
                    "image_id": ds.image_ids[0],
                    "category_id": 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": 1.0,
                }
            ]
            coco_dt = ds.coco.loadRes(results)
            coco_eval = COCOeval(ds.coco, coco_dt, iouType="bbox")
            coco_eval.params.imgIds = ds.image_ids
            coco_eval.evaluate()
            coco_eval.accumulate()
            precision = coco_eval.eval["precision"][:, :, 0, 0, -1]
            map_score = precision[precision > -1].mean() if (precision > -1).any() else 0.0
            self.assertGreater(float(map_score), 0.9, "a perfect predictor must score near 1.0 mAP")

    def test_zip_backed_loading_matches_plain_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ann_path, images_dir = _write_fake_coco(root, orig_size=(640, 480))

            zip_path = root / "images.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.write(images_dir / "img_0.jpg", "images/img_0.jpg")

            ds_plain = CocoDetectionDataset(annotations_path=ann_path, images_root=images_dir, image_size=224)
            ds_zip = CocoDetectionDataset(
                annotations_path=ann_path, images_root=images_dir, image_size=224, image_zip=zip_path
            )
            img_plain, target_plain = ds_plain[0]
            img_zip, target_zip = ds_zip[0]
            self.assertTrue(np.allclose(img_plain.numpy(), img_zip.numpy()))
            self.assertTrue(torch.allclose(target_plain["boxes"], target_zip["boxes"]))


if __name__ == "__main__":
    unittest.main()
