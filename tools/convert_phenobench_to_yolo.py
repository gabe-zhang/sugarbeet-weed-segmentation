"""Convert PhenoBench plant instances to YOLO segmentation format.

Reads plant_instances + semantics masks and produces per-image
polygon label files compatible with Ultralytics instance seg.

Output structure:
    data/PhenoBench_yolo_seg/
    ├── images/{train,val}/  (symlinks to originals)
    ├── labels/{train,val}/  (txt polygon labels)
    └── data.yaml

Usage:
    uv run tools/convert_phenobench_to_yolo.py \
        --src data/PhenoBench --dst data/PhenoBench_yolo_seg
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

# PhenoBench semantic → YOLO class mapping
# semantic 1 (crop/sugarbeet) → yolo 0
# semantic 2 (weed)           → yolo 1
# semantic 3 → 1 (partial crop), semantic 4 → 2 (partial weed)
SEM_TO_YOLO = {1: 0, 2: 1, 3: 0, 4: 1}

MIN_CONTOUR_PTS = 3  # YOLO needs at least 3 polygon points


def instance_to_polygons(
    inst_mask: np.ndarray,
    sem_mask: np.ndarray,
    h: int,
    w: int,
) -> list[str]:
    """Convert instance mask to YOLO polygon lines.

    Returns list of strings, each: '<cls> <x1> <y1> ... <xn> <yn>'
    with coordinates normalized to [0, 1].
    """
    lines: list[str] = []
    for iid in np.unique(inst_mask):
        if iid == 0:
            continue

        binary = (inst_mask == iid).astype(np.uint8)

        # Determine semantic class for this instance
        sem_vals = np.unique(sem_mask[binary == 1])
        sem_cls = int(sem_vals[0])
        if sem_cls not in SEM_TO_YOLO:
            continue
        yolo_cls = SEM_TO_YOLO[sem_cls]

        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue

        # Take largest contour
        contour = max(contours, key=cv2.contourArea)
        if len(contour) < MIN_CONTOUR_PTS:
            continue

        # Normalize to 0-1 and flatten
        pts = contour.reshape(-1, 2).astype(np.float64)
        pts[:, 0] /= w
        pts[:, 1] /= h
        coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in pts)
        lines.append(f"{yolo_cls} {coords}")

    return lines


def convert_split(src: Path, dst: Path, split: str) -> tuple[int, int]:
    """Convert one split. Returns (n_images, n_instances)."""
    img_src = src / split / "images"
    inst_src = src / split / "plant_instances"
    sem_src = src / split / "semantics"

    img_dst = dst / "images" / split
    lbl_dst = dst / "labels" / split
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    fnames = sorted(f.name for f in img_src.glob("*.png"))
    total_instances = 0

    for fname in tqdm(fnames, desc=split):
        stem = Path(fname).stem

        # Symlink image
        link = img_dst / fname
        if not link.exists():
            link.symlink_to((img_src / fname).resolve())

        # Load masks
        inst = np.array(Image.open(inst_src / fname))
        sem = np.array(Image.open(sem_src / fname))
        h, w = inst.shape[:2]

        # Convert to polygon labels
        lines = instance_to_polygons(inst, sem, h, w)
        total_instances += len(lines)

        # Write label file (empty file if no instances)
        with open(lbl_dst / f"{stem}.txt", "w") as f:
            f.write("\n".join(lines))
            if lines:
                f.write("\n")

    return len(fnames), total_instances


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert PhenoBench to YOLO seg format."
    )
    parser.add_argument(
        "--src",
        type=str,
        default="data/PhenoBench",
        help="Source PhenoBench directory.",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default="data/PhenoBench_yolo_seg",
        help="Output YOLO seg directory.",
    )
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val"):
        n_imgs, n_inst = convert_split(src, dst, split)
        print(f"{split}: {n_imgs} images, {n_inst} instances")

    # Write data.yaml
    data_cfg = {
        "path": str(dst.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "sugarbeet", 1: "weed"},
    }
    data_yaml = dst / "data.yaml"
    with open(data_yaml, "w") as f:
        yaml.dump(data_cfg, f, default_flow_style=False)
    print(f"\nData config: {data_yaml}")


if __name__ == "__main__":
    main()
