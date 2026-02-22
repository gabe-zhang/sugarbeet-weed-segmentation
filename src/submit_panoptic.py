"""Generate CodaLab panoptic submission for PhenoBench.

Runs YOLO instance seg, produces both semantic and instance
PNGs, and packages them into a zip for panoptic evaluation.
On val split, computes PQ+ = (IoU(soil) + PQ(crop) + PQ(weed)) / 3.

Submission zip layout:
    semantics/*.png        uint8  (0=soil, 1=crop, 2=weed)
    plant_instances/*.png  uint16 (0=bg, 1..N=instance IDs)

Usage:
    # Evaluate on val:
    uv run src/submit_panoptic.py \
        --weights runs/yolo26x-baseline/weights/best.pt \
        --split val --export_dir runs

    # Submit on test:
    uv run src/submit_panoptic.py \
        --weights runs/yolo26x-baseline/weights/best.pt \
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

# fmt: off
YOLO_TO_SEM = {0: 1, 1: 2}   # yolo 0=sugarbeet→sem 1, yolo 1=weed→sem 2
CLASS_NAMES  = ["soil", "crop", "weed"]
NUM_CLASSES  = 3
# fmt: on


def parse_args() -> dict:
    parser = argparse.ArgumentParser(
        description="YOLO panoptic submission for PhenoBench."
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


def instances_to_panoptic(
    result,
    orig_h: int,
    orig_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert YOLO result to semantic + instance maps.

    Returns:
        semantic: uint8 [H, W], values 0/1/2.
        instances: uint16 [H, W], 0=bg, 1..N=instance IDs.
    """
    semantic = np.zeros((orig_h, orig_w), dtype=np.uint8)
    instances = np.zeros((orig_h, orig_w), dtype=np.uint16)

    if result.masks is None or len(result.masks) == 0:
        return semantic, instances

    masks = result.masks.data.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()

    # Sort by confidence ascending (lowest first, highest overwrites)
    order = np.argsort(confs)
    inst_id = 0
    for idx in order:
        yolo_cls = classes[idx]
        sem_cls = YOLO_TO_SEM.get(yolo_cls)
        if sem_cls is None:
            continue

        mask = masks[idx]
        if mask.shape != (orig_h, orig_w):
            mask = cv2.resize(
                mask,
                (orig_w, orig_h),
                interpolation=cv2.INTER_LINEAR,
            )
        binary = mask > 0.5
        inst_id += 1
        semantic[binary] = sem_cls
        instances[binary] = inst_id

    return semantic, instances


# ── Local PQ evaluation (mirrors PhenoBench evaluator) ──────


def compute_pq_single_class(
    pred_inst: np.ndarray,
    gt_inst: np.ndarray,
) -> float:
    """PQ for one semantic class (instance-level matching).

    Uses IoU > 0.5 matching threshold, same as PhenoBench.
    """
    pred_ids = np.unique(pred_inst)
    pred_ids = pred_ids[pred_ids > 0]
    gt_ids = np.unique(gt_inst)
    gt_ids = gt_ids[gt_ids > 0]

    if len(gt_ids) == 0 and len(pred_ids) == 0:
        return 1.0
    if len(gt_ids) == 0:
        return 0.0

    gt_masks = [(gt_inst == gid) for gid in gt_ids]
    gt_areas = [m.sum() for m in gt_masks]
    matched = [False] * len(gt_ids)

    iou_sum = 0.0
    tp = 0
    fp = 0

    for pid in pred_ids:
        pred_mask = pred_inst == pid
        pred_area = pred_mask.sum()
        best_iou = 0.0
        best_idx = -1

        for j in range(len(gt_ids)):
            if matched[j]:
                continue
            inter = np.logical_and(pred_mask, gt_masks[j]).sum()
            union = pred_area + gt_areas[j] - inter
            if union == 0:
                continue
            iou = inter / union
            if iou > best_iou:
                best_iou = iou
                best_idx = j

        if best_iou > 0.5:
            tp += 1
            iou_sum += best_iou
            matched[best_idx] = True
        else:
            fp += 1

    fn = len(gt_ids) - tp
    denom = tp + 0.5 * fp + 0.5 * fn
    if denom == 0:
        return 0.0
    return iou_sum / denom


def compute_pq_image(
    pred_sem: np.ndarray,
    gt_sem: np.ndarray,
    pred_inst: np.ndarray,
    gt_inst: np.ndarray,
) -> dict[int, float]:
    """Compute per-class PQ for one image.

    Returns dict {semantic_class: pq_value} for classes
    present in GT (classes 1=crop, 2=weed).
    """
    results = {}
    for cls in (1, 2):
        gt_cls_mask = gt_sem == cls
        if not gt_cls_mask.any():
            continue
        pred_cls_mask = pred_sem == cls
        pred_inst_cls = (pred_inst * pred_cls_mask).astype(np.uint16)
        gt_inst_cls = (gt_inst * gt_cls_mask).astype(np.uint16)
        pq = compute_pq_single_class(pred_inst_cls, gt_inst_cls)
        results[cls] = pq
    return results


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
    split = args["split"]
    img_dir = data_root / split / "images"
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Not found: {img_dir}")

    has_gt = split == "val"
    gt_sem_dir = data_root / split / "semantics"
    gt_inst_dir = data_root / split / "plant_instances"

    # Output dirs
    weights_name = Path(args["weights"]).parent.parent.name
    run_dir = Path(args["export_dir"]) / weights_name
    sub_name = f"panoptic_{split}"
    sem_dir = run_dir / sub_name / "semantics"
    inst_dir = run_dir / sub_name / "plant_instances"
    sem_dir.mkdir(parents=True, exist_ok=True)
    inst_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args["weights"])

    fnames = sorted(f for f in os.listdir(img_dir) if f.endswith(".png"))
    print(f"Running panoptic prediction on {len(fnames)} {split} images...")

    # IoU accumulators for soil
    inter = np.zeros(NUM_CLASSES, dtype=np.int64)
    union = np.zeros(NUM_CLASSES, dtype=np.int64)

    # PQ accumulators: running average per class
    pq_sum: dict[int, float] = {}
    pq_count: dict[int, int] = {}

    for fname in tqdm(fnames):
        img_path = str(img_dir / fname)

        results = model(
            img_path,
            imgsz=args["imgsz"],
            conf=args["conf"],
            iou=args["iou"],
            verbose=False,
        )
        result = results[0]
        orig_h, orig_w = result.orig_shape

        pred_sem, pred_inst = instances_to_panoptic(result, orig_h, orig_w)

        # Save predictions
        Image.fromarray(pred_sem).save(sem_dir / fname)
        Image.fromarray(pred_inst).save(inst_dir / fname)

        # Evaluate against GT
        if has_gt:
            gt_sem = np.array(Image.open(gt_sem_dir / fname))
            if len(gt_sem.shape) > 2:
                gt_sem = gt_sem[:, :, 0]
            gt_sem[gt_sem == 3] = 1
            gt_sem[gt_sem == 4] = 2

            gt_inst = np.array(Image.open(gt_inst_dir / fname))
            if len(gt_inst.shape) > 2:
                gt_inst = gt_inst[:, :, 0]

            # IoU for soil (and all classes)
            accumulate_confusion(pred_sem, gt_sem, NUM_CLASSES, inter, union)

            # PQ per class
            pq_img = compute_pq_image(pred_sem, gt_sem, pred_inst, gt_inst)
            for cls, pq_val in pq_img.items():
                if cls not in pq_sum:
                    pq_sum[cls] = 0.0
                    pq_count[cls] = 0
                pq_sum[cls] += pq_val
                pq_count[cls] += 1

    # Print val metrics
    if has_gt and union.sum() > 0:
        iou_per_class = np.where(union > 0, inter / union, 0.0)
        iou_soil = iou_per_class[0]

        pq_crop = (
            pq_sum.get(1, 0.0) / pq_count[1] if pq_count.get(1, 0) else 0.0
        )
        pq_weed = (
            pq_sum.get(2, 0.0) / pq_count[2] if pq_count.get(2, 0) else 0.0
        )
        pq_plus = (iou_soil * 100 + pq_crop * 100 + pq_weed * 100) / 3

        print(f"\n{'=' * 44}")
        print("  Val Results (Panoptic)")
        print(f"{'=' * 44}")
        for i, name in enumerate(CLASS_NAMES):
            print(f"  IoU  {name:>5}: {iou_per_class[i]:.5f}")
        miou = iou_per_class.mean()
        print(f"  IoU   mIoU: {miou:.5f}")
        print(f"  PQ   crop : {pq_crop * 100:.2f}")
        print(f"  PQ   weed : {pq_weed * 100:.2f}")
        print(f"  PQ+       : {pq_plus:.2f}")
        print(f"{'=' * 44}")

    # Create submission zip
    if split == "test":
        zip_path = run_dir / f"{weights_name}_panoptic.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            for d, prefix in (
                (sem_dir, "semantics"),
                (inst_dir, "plant_instances"),
            ):
                zf.mkdir(prefix)
                for f in sorted(os.listdir(d)):
                    zf.write(d / f, f"{prefix}/{f}")
        print(f"\nSubmission zip: {zip_path}")
        print("\nValidate with:")
        print(
            f"  phenobench-validator --task panoptic"
            f" --phenobench_dir {data_root}"
            f" --zipfile {zip_path}"
        )

    print(f"\nSemantics:  {sem_dir}")
    print(f"Instances:  {inst_dir}")


if __name__ == "__main__":
    main()
