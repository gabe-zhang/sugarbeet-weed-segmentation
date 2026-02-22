"""Generate CodaLab submission for PhenoBench semantics.

Runs inference on test or val split, saves argmax predictions
as PNGs, and packages them into a zip file ready for upload.
When using --split val, also computes IoU metrics.

Usage:
    uv run src/submit.py \
        --config config/erfnet_finetune_phenobench.yaml \
        --ckpt_path runs/.../best.ckpt \
        --export_dir runs

    # Validate on val set with TTA:
    uv run src/submit.py \
        --config config/erfnet_finetune_phenobench.yaml \
        --ckpt_path runs/.../best.ckpt \
        --export_dir runs --split val --tta
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent),
)

import cv2
import numpy as np
import oyaml as yaml
import torch
import torch.nn.functional as functional
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from datasets.image_normalizer import (
    ImageNormalizer,
    get_image_normalizer,
)
from modules import get_backbone, get_criterion, module

# Class colors: soil=black, crop=green, weed=red
CLASS_COLORS = np.array([[0, 0, 0], [0, 255, 0], [255, 0, 0]], dtype=np.uint8)
CLASS_NAMES = ["soil", "crop", "weed"]


class ImageFolderDataset(Dataset):
    """Load images from a directory, no augmentation."""

    def __init__(
        self,
        img_dir: str,
        fnames: list[str],
        normalizer: ImageNormalizer,
    ) -> None:
        """Create dataset from image directory and normalizer."""
        self.img_dir = Path(img_dir)
        self.fnames = fnames
        self.normalizer = normalizer
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.fnames)

    def __getitem__(self, idx: int) -> dict:
        fname = self.fnames[idx]
        img_pil = Image.open(self.img_dir / fname).convert("RGB")
        img = self.to_tensor(img_pil)
        img_norm = self.normalizer.normalize(img)
        return {
            "image": img_norm,
            "image_raw": img,
            "fname": fname,
        }


def parse_args() -> dict:
    parser = argparse.ArgumentParser(
        description="Generate CodaLab submission zip."
    )
    parser.add_argument(
        "--config",
        required=True,
        action="append",
        help="Path to config (repeat for ensemble).",
    )
    parser.add_argument(
        "--ckpt_path",
        required=True,
        action="append",
        help="Path to checkpoint (repeat for ensemble).",
    )
    parser.add_argument(
        "--export_dir",
        required=True,
        help="Directory to save submission files.",
    )
    parser.add_argument(
        "--data_path",
        required=False,
        default=None,
        help="Override path to dataset root.",
    )
    parser.add_argument(
        "--split",
        choices=["test", "val"],
        default="test",
        help="Dataset split to run on (default: test).",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Enable test-time augmentation (multi-scale + flip).",
    )
    parser.add_argument(
        "--tta_scales",
        type=float,
        nargs="+",
        default=[0.75, 1.0, 1.25],
        help="TTA scale factors (default: 0.75 1.0 1.25).",
    )
    parser.add_argument(
        "--opening",
        type=int,
        default=0,
        help="Morphological opening kernel size (0=off).",
    )
    parser.add_argument(
        "--erosion",
        type=int,
        default=0,
        help="Morphological erosion kernel size (0=off).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for inference (default: 4).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader workers (default: 4).",
    )
    return vars(parser.parse_args())


def load_config(path: str) -> dict:
    """Load YAML config from path."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with p.open() as f:
        return yaml.safe_load(f)


def predict_batch(
    model: torch.nn.Module,
    batch: torch.Tensor,
) -> torch.Tensor:
    """Forward pass on a batch, return softmax probs.

    Args:
        model: segmentation model.
        batch: [B, C, H, W] normalized input.

    Returns:
        [B, n_classes, H, W] softmax probabilities.

    """
    logits = model(batch)
    return functional.softmax(logits, dim=1)


def predict_tta_batch(
    model: torch.nn.Module,
    batch: torch.Tensor,
    scales: list[float],
) -> torch.Tensor:
    """Multi-scale + flip TTA on a batch.

    For each scale: predict original + hflip + vflip,
    resize back to original size, average all probs.

    Args:
        model: segmentation model.
        batch: [B, C, H, W] normalized input.
        scales: list of scale factors.

    Returns:
        [B, n_classes, H, W] averaged softmax probs.

    """
    b, _, orig_h, orig_w = batch.shape
    accum = torch.zeros(
        b,
        model.network.num_classes,
        orig_h,
        orig_w,
        device=batch.device,
    )
    count = 0

    for scale in scales:
        if scale != 1.0:
            sh = int(orig_h * scale)
            sw = int(orig_w * scale)
            scaled = functional.interpolate(
                batch,
                size=(sh, sw),
                mode="bilinear",
                align_corners=False,
            )
        else:
            scaled = batch

        # Original
        probs = predict_batch(model, scaled)
        if scale != 1.0:
            probs = functional.interpolate(
                probs,
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            )
        accum += probs
        count += 1

        # Horizontal flip
        flipped_h = torch.flip(scaled, dims=[3])
        probs_h = predict_batch(model, flipped_h)
        probs_h = torch.flip(probs_h, dims=[3])
        if scale != 1.0:
            probs_h = functional.interpolate(
                probs_h,
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            )
        accum += probs_h
        count += 1

        # Vertical flip
        flipped_v = torch.flip(scaled, dims=[2])
        probs_v = predict_batch(model, flipped_v)
        probs_v = torch.flip(probs_v, dims=[2])
        if scale != 1.0:
            probs_v = functional.interpolate(
                probs_v,
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            )
        accum += probs_v
        count += 1

    return accum / count


def apply_opening(pred: np.ndarray, kernel_size: int) -> np.ndarray:
    """Per-class morphological opening to remove noise."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    classes = np.unique(pred)
    out = np.zeros_like(pred)
    for cls in classes:
        mask = (pred == cls).astype(np.uint8)
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        out[opened == 1] = cls
    return out


def apply_erosion(pred: np.ndarray, kernel_size: int) -> np.ndarray:
    """Per-class morphological erosion to shrink boundaries."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    classes = np.unique(pred)
    out = np.zeros_like(pred)
    for cls in classes:
        mask = (pred == cls).astype(np.uint8)
        eroded = cv2.erode(mask, kernel)
        out[eroded == 1] = cls
    return out


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


def _prepare_paths(cfg: dict, export_dir: str, split: str):
    """Prepare and validate dataset + output directories."""
    data_root = Path(cfg["data"]["path_to_dataset"])
    img_dir = data_root / split / "images"
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Images not found: {img_dir}")

    has_gt = split == "val"
    gt_dir = data_root / split / "semantics" if has_gt else None
    if has_gt and not gt_dir.is_dir():
        raise FileNotFoundError(f"GT not found: {gt_dir}")

    run_dir = Path(export_dir) / cfg["experiment"]["id"]
    sub_name = f"submission_{split}"
    sem_dir = run_dir / sub_name / "semantics"
    overlay_dir = run_dir / sub_name / "overlay"
    sem_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    return data_root, img_dir, gt_dir, run_dir, sem_dir, overlay_dir, has_gt


def _load_segmentation_module(cfg: dict, ckpt_path: str, device: torch.device):
    """Instantiate backbone+Lightning module and load checkpoint."""
    network = get_backbone(cfg)
    criterion = get_criterion(cfg)
    seg_module = module.SegmentationNetwork(
        network,
        criterion,
        cfg["train"]["learning_rate"],
        cfg["train"]["weight_decay"],
    )

    ckpt = torch.load(ckpt_path, map_location=device)
    seg_module.load_state_dict(ckpt["state_dict"])
    seg_module = seg_module.to(device)
    seg_module.eval()

    return seg_module


def _run_inference(
    loader: DataLoader,
    models: list[module.SegmentationNetwork],
    sem_dir: Path,
    overlay_dir: Path,
    has_gt: bool,
    gt_dir: Path | None,
    num_classes: int,
    use_tta: bool,
    tta_scales: list[float],
    opening_ks: int,
    erosion_ks: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference loop and save semantic maps + overlays.

    Returns (intersection, union) arrays for global IoU.
    """
    inter = np.zeros(num_classes, dtype=np.int64)
    union = np.zeros(num_classes, dtype=np.int64)
    alpha = 0.4

    device = next(models[0].parameters()).device
    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader)):
            imgs = batch["image"].to(device)
            imgs_raw = batch["image_raw"]
            batch_fnames = batch["fname"]

            # Ensemble: average softmax probs across models
            accum = None
            for seg_module in models:
                if use_tta:
                    probs = predict_tta_batch(seg_module, imgs, tta_scales)
                else:
                    probs = predict_batch(seg_module, imgs)
                if accum is None:
                    accum = probs
                else:
                    accum += probs
            preds = (accum / len(models)).argmax(dim=1)

            preds_np = preds.cpu().numpy().astype(np.uint8)

            # Per-image postprocessing and saving
            for j, fname in enumerate(batch_fnames):
                pred_np = preds_np[j]

                if opening_ks > 0:
                    pred_np = apply_opening(pred_np, opening_ks)
                if erosion_ks > 0:
                    pred_np = apply_erosion(pred_np, erosion_ks)

                # Save semantic prediction
                Image.fromarray(pred_np).save(sem_dir / fname)

                # Save overlay visualization
                raw = imgs_raw[j]  # [C, H, W] in [0,1]
                orig_np = raw.permute(1, 2, 0).mul(255).byte().numpy()
                color_map = CLASS_COLORS[pred_np]
                blended = (orig_np * (1 - alpha) + color_map * alpha).astype(
                    np.uint8
                )
                Image.fromarray(blended).save(overlay_dir / fname)

                # Accumulate global confusion for IoU
                if has_gt:
                    gt = np.array(Image.open(gt_dir / fname))
                    if len(gt.shape) > 2:
                        gt = gt[:, :, 0]
                    gt[gt == 3] = 1
                    gt[gt == 4] = 2
                    accumulate_confusion(
                        pred_np, gt, num_classes, inter, union
                    )

    return inter, union


def main() -> None:
    """CLI entrypoint for generating a CodaLab submission zip or
    validating on the `val` split."""
    args = parse_args()
    configs = args["config"]
    ckpt_paths = args["ckpt_path"]
    if len(configs) != len(ckpt_paths):
        raise ValueError(
            f"Mismatched --config ({len(configs)}) and "
            f"--ckpt_path ({len(ckpt_paths)}) counts."
        )

    cfg = load_config(configs[0])
    if args["data_path"] is not None:
        cfg["data"]["path_to_dataset"] = args["data_path"]

    data_root, img_dir, gt_dir, run_dir, sem_dir, overlay_dir, has_gt = (
        _prepare_paths(cfg, args["export_dir"], args["split"])
    )

    # Load all models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models: list[module.SegmentationNetwork] = []
    for cfg_path, ckpt_path in zip(configs, ckpt_paths):
        c = load_config(cfg_path)
        m = _load_segmentation_module(c, ckpt_path, device)
        models.append(m)
        print(f"Loaded: {cfg_path} -> {ckpt_path}")

    img_normalizer = get_image_normalizer(cfg)
    num_classes = cfg["backbone"]["num_classes"]

    use_tta = args["tta"]
    tta_scales = args["tta_scales"]
    opening_ks = args["opening"]
    erosion_ks = args["erosion"]
    flags: list[str] = []
    if len(models) > 1:
        flags.append(f"ensemble x{len(models)}")
    if use_tta:
        flags.append(f"TTA scales={tta_scales} flip=hv")
    if opening_ks > 0:
        flags.append(f"opening k={opening_ks}")
    if erosion_ks > 0:
        flags.append(f"erosion k={erosion_ks}")
    mode_str = ", ".join(flags) if flags else "single pass"

    # Get filenames and build DataLoader
    fnames = sorted(f for f in os.listdir(img_dir) if f.endswith(".png"))
    dataset = ImageFolderDataset(img_dir, fnames, img_normalizer)
    loader = DataLoader(
        dataset,
        batch_size=args["batch_size"],
        num_workers=args["num_workers"],
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )
    print(
        f"Running inference on {len(fnames)}"
        f" {args['split']} images [{mode_str}]..."
    )

    inter, union = _run_inference(
        loader,
        models,
        sem_dir,
        overlay_dir,
        has_gt,
        gt_dir,
        num_classes,
        use_tta,
        tta_scales,
        opening_ks,
        erosion_ks,
    )

    # Print val metrics (global IoU, matching PhenoBench eval)
    if has_gt and union.sum() > 0:
        iou_per_class = np.where(union > 0, inter / union, 0.0)
        miou = iou_per_class.mean()
        print(f"\n{'=' * 40}")
        print(f"  Val Results ({mode_str})")
        print(f"{'=' * 40}")
        for i, name in enumerate(CLASS_NAMES):
            print(f"  {name:>5}: {iou_per_class[i]:.5f}")
        print(f"  {'mIoU':>5}: {miou:.5f}")
        print(f"{'=' * 40}")

    # Create zip (for test split)
    if args["split"] == "test":
        zip_path = run_dir / f"{cfg['experiment']['id']}_submission.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.mkdir("semantics")
            for fname in sorted(os.listdir(sem_dir)):
                fpath = sem_dir / fname
                zf.write(fpath, f"semantics/{fname}")
        print(f"\nSubmission zip: {zip_path}")
        print("\nValidate with:")
        print(
            f"  phenobench-validator --task semantics "
            f"--phenobench_dir {data_root} "
            f"--zipfile {zip_path}"
        )

    print(f"\nPredictions: {sem_dir}")
    print(f"Overlays:    {overlay_dir}")


if __name__ == "__main__":
    main()
