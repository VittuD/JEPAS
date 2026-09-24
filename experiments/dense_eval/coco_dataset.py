"""COCO detection dataset -- val2017 from plain files, train2017 read directly
out of train2017.zip (never extracted, see the Gluster many-small-files
lesson in this session's history) via Python's zipfile, no materialized
per-image files needed on disk for the train split.

Unlike experiments/extract.py's adapters (center-crop, matching a
whole-image/mean-pooled representation), detection needs the *entire* image
with boxes intact, so this does a plain square resize (no crop) and rescales
box coordinates by the same (possibly non-uniform) x/y factors.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset

_IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
_IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)


def build_category_mapping(coco: COCO) -> dict[int, int]:
    """COCO category ids have gaps (up to 90 for 80 classes); map to a
    contiguous 1..80 range (0 reserved for background, torchvision convention)."""
    cat_ids = sorted(coco.getCatIds())
    return {cat_id: index + 1 for index, cat_id in enumerate(cat_ids)}


class CocoDetectionDataset(Dataset):
    def __init__(
        self,
        *,
        annotations_path: Path,
        images_root: Path,
        image_size: int,
        image_zip: Path | None = None,
        limit: int | None = None,
        zip_path_prefix: str | None = None,
    ) -> None:
        self.coco = COCO(str(annotations_path))
        self.images_root = images_root
        self.image_zip = image_zip
        self.image_size = image_size
        # COCO's file_name values are bare ("000000000009.jpg"), stored in the
        # zip under "train2017/000000000009.jpg" -- needs a prefix. Datasets
        # whose converter already writes the full zip-relative path (e.g.
        # Cityscapes, nested by city) pass zip_path_prefix="" to use file_name
        # verbatim.
        self.zip_path_prefix = f"{images_root.name}/" if zip_path_prefix is None else zip_path_prefix
        self.category_mapping = build_category_mapping(self.coco)
        # Only keep images that actually have at least one annotation -- an
        # empty-target image is valid COCO but not useful for a first pass.
        image_ids = [
            img_id for img_id in sorted(self.coco.imgs) if len(self.coco.getAnnIds(imgIds=img_id)) > 0
        ]
        self.image_ids = image_ids[:limit] if limit is not None else image_ids
        self._zip_handle: zipfile.ZipFile | None = None
        self._rescale_coco_to_image_size()

    def _rescale_coco_to_image_size(self) -> None:
        """Rescale self.coco's stored boxes/image dims to the fixed square
        image_size in place, so the SAME coco object used for __getitem__'s
        training targets is also correct when handed to pycocotools'
        COCOeval as ground truth for evaluation -- otherwise COCOeval
        compares model predictions (in resized-image space) against
        original-resolution boxes and silently reports ~0 IoU everywhere,
        indistinguishable from "the model just hasn't learned anything" (see
        this session's history for how that actually played out)."""
        for image_id in self.image_ids:
            info = self.coco.imgs[image_id]
            orig_w, orig_h = info["width"], info["height"]
            scale_x, scale_y = self.image_size / orig_w, self.image_size / orig_h
            for ann in self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id)):
                x, y, w, h = ann["bbox"]
                ann["bbox"] = [x * scale_x, y * scale_y, w * scale_x, h * scale_y]
                ann["area"] = ann["area"] * scale_x * scale_y
            info["width"], info["height"] = self.image_size, self.image_size

    def __len__(self) -> int:
        return len(self.image_ids)

    def _zip(self) -> zipfile.ZipFile:
        # Lazily opened per-process (safe under DataLoader worker forking,
        # unlike sharing one handle opened before fork).
        if self._zip_handle is None:
            self._zip_handle = zipfile.ZipFile(self.image_zip)
        return self._zip_handle

    def _load_image(self, file_name: str) -> Image.Image:
        if self.image_zip is not None:
            with self._zip().open(f"{self.zip_path_prefix}{file_name}") as f:
                return Image.open(io.BytesIO(f.read())).convert("RGB")
        return Image.open(self.images_root / file_name).convert("RGB")

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        image_id = self.image_ids[index]
        info = self.coco.imgs[image_id]
        image = self._load_image(info["file_name"])
        # self.coco's boxes/dims were already rescaled to image_size in
        # _rescale_coco_to_image_size(); resizing the actual pixels here
        # (not preserving aspect ratio) is what makes that rescale correct.
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes, labels, areas, iscrowd = [], [], [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.category_mapping[ann["category_id"]])
            areas.append(ann["area"])
            iscrowd.append(ann.get("iscrowd", 0))

        tensor = torch.from_numpy(
            __import__("numpy").asarray(image).copy()
        ).permute(2, 0, 1).float().div(255.0)
        tensor = (tensor - _IMAGENET_MEAN) / _IMAGENET_STD

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([image_id]),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.int64),
        }
        return tensor, target


def collate_fn(batch: list[tuple[torch.Tensor, dict[str, Any]]]) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
    images, targets = zip(*batch)
    return list(images), list(targets)
