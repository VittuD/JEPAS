#!/usr/bin/env python3
"""Train a Faster R-CNN detection head (frozen JEPA backbone) on COCO and
report standard pycocotools mAP.

First-pass validation harness for whether token-fragmentation metrics
(experiments/token_metrics.py) predict real dense-task performance -- see
that discussion for the full context. Only the ViTDet-style neck + RPN + ROI
heads train; the backbone is frozen throughout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from experiments.dense_eval.backbones import build_backbone  # noqa: E402
from experiments.dense_eval.coco_dataset import CocoDetectionDataset, collate_fn  # noqa: E402
from experiments.dense_eval.simple_fpn import SimpleFeaturePyramid  # noqa: E402


# Override on clusters where datasets are not under /shared/datasets.
DATASETS_ROOT = Path(os.environ.get("JEPAS_DATASETS_ROOT", "/shared/datasets"))
COCO_ROOT = DATASETS_ROOT / "detection" / "coco"
CITYSCAPES_ROOT = DATASETS_ROOT / "detection" / "cityscapes"

MODEL_CHECKPOINTS = {
    "ijepa": ROOT_DIR / "weights" / "ijepa" / "IN1K-vit.h.14-300e.pth.tar",
}
MODEL_IMAGE_SIZE = {
    "ijepa": 224,
}

# num_classes includes the background class (torchvision convention), so
# COCO's 80 things -> 81, Cityscapes' 8 things -> 9.
DATASET_CONFIGS = {
    "coco": {
        "num_classes": 81,
        "train_annotations": COCO_ROOT / "annotations" / "instances_train2017.json",
        "train_images_root": Path("train2017"),
        "train_zip": COCO_ROOT / "train2017.zip",
        "train_zip_prefix": None,  # default: f"{images_root.name}/"
        "val_annotations": COCO_ROOT / "annotations" / "instances_val2017.json",
        "val_images_root": COCO_ROOT / "val2017",
        "val_zip": None,
        "val_zip_prefix": None,
    },
    "cityscapes": {
        "num_classes": 9,
        "train_annotations": CITYSCAPES_ROOT / "annotations" / "instances_train.json",
        "train_images_root": Path("leftImg8bit"),
        "train_zip": CITYSCAPES_ROOT / "leftImg8bit_trainvaltest.zip",
        "train_zip_prefix": "",  # converter already writes the full zip-relative path
        "val_annotations": CITYSCAPES_ROOT / "annotations" / "instances_val.json",
        "val_images_root": Path("leftImg8bit"),
        "val_zip": CITYSCAPES_ROOT / "leftImg8bit_trainvaltest.zip",
        "val_zip_prefix": "",
    },
}


def build_model(model_name: str, *, num_classes: int, device: torch.device) -> FasterRCNN:
    image_size = MODEL_IMAGE_SIZE[model_name]
    backbone = build_backbone(model_name, checkpoint=MODEL_CHECKPOINTS[model_name], image_size=image_size, device=device)
    neck = SimpleFeaturePyramid(backbone, out_channels=256).to(device)

    anchor_gen = AnchorGenerator(sizes=((16,), (32,), (64,), (128,)), aspect_ratios=((0.5, 1.0, 2.0),) * 4)
    roi_pool = MultiScaleRoIAlign(featmap_names=["p2", "p3", "p4", "p5"], output_size=7, sampling_ratio=2)
    model = FasterRCNN(
        neck,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_gen,
        box_roi_pool=roi_pool,
        min_size=image_size,
        max_size=image_size,
    )
    return model.to(device)


def train_one_epoch(
    model: FasterRCNN, loader: DataLoader, optimizer: torch.optim.Optimizer, *, device: torch.device, log_every: int
) -> list[float]:
    model.train()
    losses: list[float] = []
    t0 = time.time()
    for step, (images, targets) in enumerate(loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        if (step + 1) % log_every == 0:
            elapsed = time.time() - t0
            print(f"  step {step + 1}/{len(loader)} loss={loss.item():.4f} ({elapsed:.1f}s elapsed)", flush=True)
    return losses


@torch.no_grad()
def evaluate(model: FasterRCNN, dataset: CocoDetectionDataset, loader: DataLoader, *, device: torch.device) -> dict:
    model.eval()
    results = []
    for images, targets in loader:
        images = [img.to(device) for img in images]
        outputs = model(images)
        inverse_mapping = {v: k for k, v in dataset.category_mapping.items()}
        for target, output in zip(targets, outputs):
            image_id = int(target["image_id"].item())
            boxes = output["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            labels = output["labels"].cpu().numpy()
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box.tolist()
                results.append(
                    {
                        "image_id": image_id,
                        "category_id": inverse_mapping[int(label)],
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score),
                    }
                )
    if not results:
        print("warning: model produced zero detections across the eval set", file=sys.stderr)
        return {"mAP": 0.0, "mAP_50": 0.0, "mAP_75": 0.0, "n_detections": 0}

    coco_dt = dataset.coco.loadRes(results)
    coco_eval = COCOeval(dataset.coco, coco_dt, iouType="bbox")
    coco_eval.params.imgIds = dataset.image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return {
        "mAP": float(coco_eval.stats[0]),
        "mAP_50": float(coco_eval.stats[1]),
        "mAP_75": float(coco_eval.stats[2]),
        "mAP_small": float(coco_eval.stats[3]),
        "mAP_medium": float(coco_eval.stats[4]),
        "mAP_large": float(coco_eval.stats[5]),
        "n_detections": len(results),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_CHECKPOINTS))
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_CONFIGS))
    parser.add_argument("--train-limit", type=int, default=2000)
    parser.add_argument("--val-limit", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = MODEL_IMAGE_SIZE[args.model]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = DATASET_CONFIGS[args.dataset]
    print(f"[dense_eval] model={args.model} dataset={args.dataset} device={device} image_size={image_size}")
    model = build_model(args.model, num_classes=cfg["num_classes"], device=device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"[dense_eval] trainable params: {trainable/1e6:.1f}M, frozen params: {frozen/1e6:.1f}M")

    train_ds = CocoDetectionDataset(
        annotations_path=cfg["train_annotations"],
        images_root=cfg["train_images_root"],
        image_size=image_size,
        image_zip=cfg["train_zip"],
        zip_path_prefix=cfg["train_zip_prefix"],
        limit=args.train_limit,
    )
    val_ds = CocoDetectionDataset(
        annotations_path=cfg["val_annotations"],
        images_root=cfg["val_images_root"],
        image_size=image_size,
        image_zip=cfg["val_zip"],
        zip_path_prefix=cfg["val_zip_prefix"],
        limit=args.val_limit,
    )
    print(f"[dense_eval] train={len(train_ds)} val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    all_losses: list[float] = []
    for epoch in range(args.epochs):
        print(f"[dense_eval] epoch {epoch + 1}/{args.epochs}")
        epoch_losses = train_one_epoch(model, train_loader, optimizer, device=device, log_every=args.log_every)
        all_losses.extend(epoch_losses)
        print(f"[dense_eval] epoch {epoch + 1} mean loss: {sum(epoch_losses) / len(epoch_losses):.4f}")

    print("[dense_eval] evaluating...")
    metrics = evaluate(model, val_ds, val_loader, device=device)
    print(f"[dense_eval] metrics: {metrics}")

    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (args.out_dir / "loss_curve.json").write_text(json.dumps(all_losses, indent=2) + "\n")
    torch.save(
        {k: v for k, v in model.state_dict().items() if "backbone.encoder" not in k},
        args.out_dir / "trainable_weights.pt",
    )
    print(f"[dense_eval] wrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
