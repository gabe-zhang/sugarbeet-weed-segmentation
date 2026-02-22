"""Run YOLO instance seg and convert to semantic masks.

Produces semantic PNGs (0=soil, 1=crop, 2=weed) compatible
with PhenoBench CodaLab submission and submit.py evaluation.

Usage:
    # Predict on val with mIoU eval:
    uv run src/predict_yolo.py \
        --weights runs/yolo-seg-v1/weights/best.pt \
        --split val --export_dir runs

    # Predict on test for CodaLab:
    uv run src/predict_yolo.py \
        --weights runs/yolo-seg-v1/weights/best.pt \
        --split test --export_dir runs
"""

import argparse
import os
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

# YOLO class → semantic class
# yolo 0 (sugarbeet) → semantic 1 (crop)
# yolo 1 (weed)      → semantic 2 (weed)
YOLO_TO_SEM = {0: 1, 1: 2}

CLASS_COLORS = np.array([[0, 0, 0], [0, 255, 0], [255, 0, 0]], dtype=np.uint8)
CLASS_NAMES = ["soil", "crop", "weed"]
NUM_CLASSES = 3


def parse_args() -> dict:
    parser = argparse.ArgumentParser(
        description="YOLO seg → semantic prediction."
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="Path to YOLO seg weights (.pt).",
    )
    parser.add_argument(
        "--data_path",
        default="data/PhenoBench",
        help="Path to PhenoBench root.",
    )
    parser.add_argument(
        "--split",
        choices=["test", "val"],
        default="test",
        help="Dataset split (default: test).",
    )
    parser.add_argument(
        "--export_dir",
        default="runs",
        help="Output directory.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="NMS IoU threshold (default: 0.7).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1024,
        help="Inference image size (default: 1024).",
    )
    return vars(parser.parse_args())


def instances_to_semantic(
    result,
    orig_h: int,
    orig_w: int,
) -> np.ndarray:
    """Convert YOLO instance seg result to semantic mask.

    Instances sorted by confidence (lowest first) so higher
    confidence instances overwrite lower ones on overlap.
    """
    semantic = np.zeros((orig_h, orig_w), dtype=np.uint8)

    if result.masks is None or len(result.masks) == 0:
        return semantic

    masks = result.masks.data.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()

    # Sort by confidence ascending (lowest painted first)
    order = np.argsort(confs)
    for idx in order:
        yolo_cls = classes[idx]
        sem_cls = YOLO_TO_SEM.get(yolo_cls)
        if sem_cls is None:
            continue

        mask = masks[idx]
        # Resize mask to original image size if needed
        if mask.shape != (orig_h, orig_w):
            mask = cv2.resize(
                mask,
                (orig_w, orig_h),
                interpolation=cv2.INTER_LINEAR,
            )
        semantic[mask > 0.5] = sem_cls

    return semantic


def accumulate_confusion(
    pred: np.ndarray,
    gt: np.ndarray,
    num_classes: int,
    inter: np.ndarray,
    union: np.ndarray,
) -> None:
    """Accumulate global intersection/union for IoU."""
    for c in range(num_classes):
        p = pred == c
        g = gt == c
        inter[c] += np.logical_and(p, g).sum()
        union[c] += np.logical_or(p, g).sum()


def main() -> None:
    args = parse_args()

    data_root = Path(args["data_path"])
    img_dir = data_root / args["split"] / "images"
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Not found: {img_dir}")

    has_gt = args["split"] == "val"
    gt_dir = data_root / args["split"] / "semantics"
    if has_gt and not gt_dir.is_dir():
        raise FileNotFoundError(f"Not found: {gt_dir}")

    # Output dirs
    weights_name = Path(args["weights"]).parent.parent.name
    run_dir = Path(args["export_dir"]) / weights_name
    sub_name = f"submission_{args['split']}"
    sem_dir = run_dir / sub_name / "semantics"
    overlay_dir = run_dir / sub_name / "overlay"
    sem_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = YOLO(args["weights"])

    fnames = sorted(f for f in os.listdir(img_dir) if f.endswith(".png"))
    print(f"Running YOLO seg on {len(fnames)} {args['split']} images...")

    inter = np.zeros(NUM_CLASSES, dtype=np.int64)
    union = np.zeros(NUM_CLASSES, dtype=np.int64)
    alpha = 0.4

    for fname in tqdm(fnames):
        img_path = str(img_dir / fname)

        # Run YOLO inference
        results = model(
            img_path,
            imgsz=args["imgsz"],
            conf=args["conf"],
            iou=args["iou"],
            verbose=False,
        )
        result = results[0]
        orig_h, orig_w = result.orig_shape

        # Convert instances → semantic mask
        pred = instances_to_semantic(result, orig_h, orig_w)

        # Save semantic prediction
        Image.fromarray(pred).save(sem_dir / fname)

        # Save overlay
        img_np = np.array(Image.open(img_path))
        color_map = CLASS_COLORS[pred]
        blended = (img_np * (1 - alpha) + color_map * alpha).astype(np.uint8)
        Image.fromarray(blended).save(overlay_dir / fname)

        # Compute IoU against GT
        if has_gt:
            gt = np.array(Image.open(gt_dir / fname))
            if len(gt.shape) > 2:
                gt = gt[:, :, 0]
            gt[gt == 3] = 1
            gt[gt == 4] = 2
            accumulate_confusion(pred, gt, NUM_CLASSES, inter, union)

    # Print val metrics
    if has_gt and union.sum() > 0:
        iou_per_class = np.where(union > 0, inter / union, 0.0)
        miou = iou_per_class.mean()
        print(f"\n{'=' * 40}")
        print("  Val Results (YOLO seg → semantic)")
        print(f"{'=' * 40}")
        for i, name in enumerate(CLASS_NAMES):
            print(f"  {name:>5}: {iou_per_class[i]:.5f}")
        print(f"  {'mIoU':>5}: {miou:.5f}")
        print(f"{'=' * 40}")

    # Create submission zip for test
    if args["split"] == "test":
        zip_path = run_dir / f"{weights_name}_submission.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.mkdir("semantics")
            for fname in sorted(os.listdir(sem_dir)):
                fpath = sem_dir / fname
                zf.write(fpath, f"semantics/{fname}")
        print(f"\nSubmission zip: {zip_path}")

    print(f"\nPredictions: {sem_dir}")
    print(f"Overlays:    {overlay_dir}")


if __name__ == "__main__":
    main()
