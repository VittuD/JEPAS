#!/usr/bin/env python3
"""Convert Cityscapes gtFine polygon annotations to a COCO-format instances
JSON, so the existing CocoDetectionDataset/train_detector.py pipeline can be
reused as-is for Cityscapes instance segmentation/detection instead of
building a second dataset/training codepath.

Only Cityscapes' 8 official "thing" classes get instance annotations (the
rest -- road, sky, building, etc. -- are "stuff", no individual instances).
"*group" labels (crowds too dense to label individually) become COCO
iscrowd=1 annotations, mirroring how COCO itself handles crowds.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any

# Cityscapes' 8 official instance ("thing") classes, in the paper's order.
THING_CLASSES = ["person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
CATEGORY_IDS = {name: index + 1 for index, name in enumerate(THING_CLASSES)}
GROUP_SUFFIX = "group"


def polygon_to_bbox(polygon: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    return x0, y0, x1 - x0, y1 - y0


def polygon_area(polygon: list[list[float]]) -> float:
    # Shoelace formula.
    area = 0.0
    n = len(polygon)
    for i in range(n):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % n]
        area += x0 * y1 - x1 * y0
    return abs(area) / 2.0


def convert_split(gt_zip: zipfile.ZipFile, split: str) -> dict[str, Any]:
    json_names = sorted(
        n for n in gt_zip.namelist() if n.startswith(f"gtFine/{split}/") and n.endswith("_polygons.json")
    )
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    next_image_id = 1
    next_ann_id = 1

    for json_name in json_names:
        with gt_zip.open(json_name) as f:
            data = json.load(f)
        # gtFine/train/jena/jena_000092_000019_gtFine_polygons.json ->
        # leftImg8bit/train/jena/jena_000092_000019_leftImg8bit.png
        image_name = json_name.replace("gtFine/", "leftImg8bit/").replace(
            "_gtFine_polygons.json", "_leftImg8bit.png"
        )
        image_id = next_image_id
        next_image_id += 1
        images.append(
            {
                "id": image_id,
                "file_name": image_name,
                "width": data["imgWidth"],
                "height": data["imgHeight"],
            }
        )

        for obj in data["objects"]:
            label = obj["label"]
            is_crowd = label.endswith(GROUP_SUFFIX)
            base_label = label[: -len(GROUP_SUFFIX)] if is_crowd else label
            if base_label not in CATEGORY_IDS:
                continue
            polygon = obj["polygon"]
            if len(polygon) < 3:
                continue
            x, y, w, h = polygon_to_bbox(polygon)
            if w <= 0 or h <= 0:
                continue
            annotations.append(
                {
                    "id": next_ann_id,
                    "image_id": image_id,
                    "category_id": CATEGORY_IDS[base_label],
                    "bbox": [x, y, w, h],
                    "area": polygon_area(polygon),
                    "iscrowd": int(is_crowd),
                    "segmentation": [[coord for point in polygon for coord in point]],
                }
            )
            next_ann_id += 1

    categories = [{"id": cat_id, "name": name} for name, cat_id in CATEGORY_IDS.items()]
    return {"images": images, "annotations": annotations, "categories": categories}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-zip", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.gt_zip) as gt_zip:
        for split in args.splits:
            coco_json = convert_split(gt_zip, split)
            out_path = args.out_dir / f"instances_{split}.json"
            out_path.write_text(json.dumps(coco_json))
            n_images, n_anns = len(coco_json["images"]), len(coco_json["annotations"])
            print(f"[cityscapes_to_coco] {split}: {n_images} images, {n_anns} annotations -> {out_path}")


if __name__ == "__main__":
    main()
