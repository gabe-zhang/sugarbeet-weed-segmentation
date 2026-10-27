"""Compare a checkpoint and TensorRT exports on unlabeled images."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch_tensorrt  # Registers TensorRT TorchScript operators.

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modules.erfnet.erfnet_modified import ERFNetModel  # noqa: E402

CLASSES = ["background", "crop", "weed"]
COLORS = np.array([[0, 0, 0], [0, 255, 0], [255, 0, 0]], dtype=np.uint8)
MODELS = ["checkpoint_fp32", "tensorrt_fp32", "tensorrt_fp16"]


def confusion_metrics(confusion):
    diagonal = np.diag(confusion).astype(float)
    union = confusion.sum(0) + confusion.sum(1) - diagonal
    iou = np.divide(diagonal, union, out=np.full(3, np.nan), where=union != 0)
    return {
        "pixel_agreement_pct": float(100 * diagonal.sum() / confusion.sum()),
        "mean_iou_vs_checkpoint": float(np.nanmean(iou)),
        "iou_vs_checkpoint": dict(
            zip(CLASSES, [float(x) if np.isfinite(x) else None for x in iou])
        ),
        "confusion_rows_checkpoint_columns_candidate": confusion.tolist(),
    }


def normalize(rgb):
    # Matches ToTensor + SingleImageNormalizer, including sample std.
    tensor = (
        torch.from_numpy(np.array(rgb, copy=True))
        .permute(2, 0, 1)
        .contiguous()
        .float()
        / 255
    )
    mean = tensor.mean(dim=(1, 2), keepdim=True)
    std = tensor.flatten(1).std(dim=1).view(3, 1, 1)
    return ((tensor - mean) / (std + 1e-17)).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "models/semantic-seg-erfnet.ckpt",
    )
    parser.add_argument(
        "--trt-fp32", type=Path, default=ROOT / "models/erfnet_tensorrt.ts"
    )
    parser.add_argument(
        "--trt-fp16",
        type=Path,
        default=ROOT / "models/erfnet_half_tensorrt.ts",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 1:
        parser.error("warmup and repeats must be positive")
    files = sorted(
        p
        for p in args.images.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not files:
        parser.error("No images found")
    if len({p.stem for p in files}) != len(files):
        parser.error("Image stems must be unique")
    for path in files:
        with Image.open(path) as image:
            if image.size != (1920, 1080):
                parser.error(f"{path}: expected 1920x1080, got {image.size}")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for these existing TensorRT engines"
        )

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    paths = dict(zip(MODELS, [args.checkpoint, args.trt_fp32, args.trt_fp16]))
    ckpt = torch.load(str(args.checkpoint), map_location="cpu")
    state = {
        k[len("network.") :] if k.startswith("network.") else k: v
        for k, v in ckpt.get("state_dict", ckpt).items()
    }
    baseline = ERFNetModel(num_classes=3)
    baseline.load_state_dict(state, strict=True)
    models = {MODELS[0]: baseline.float().eval().cuda()}
    del state, ckpt, baseline
    for name in MODELS[1:]:
        models[name] = torch.jit.load(str(paths[name])).eval()

    args.output.mkdir(parents=True, exist_ok=False)
    for name in MODELS:
        for kind in ["masks", "overlays"]:
            (args.output / name / kind).mkdir(parents=True)
    for name in MODELS[1:]:
        (args.output / name / "disagreements").mkdir()
    (args.output / "comparisons").mkdir()
    metadata = {
        "images": str(args.images.resolve()),
        "image_count": len(files),
        "input_shape": [1, 3, 1080, 1920],
        "input_memory_layout": "contiguous NCHW, as with torchvision ToTensor",
        "torch": torch.__version__,
        "torch_tensorrt": torch_tensorrt.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "normalization": (
            "RGB / 255; per-image per-channel (x-mean)/(sample_std+1e-17); "
            "no resize or crop"
        ),
        "classes": dict(enumerate(CLASSES)),
        "colors": COLORS.tolist(),
        "warmup_for_each_model": args.warmup,
        "timed_forwards_per_image": args.repeats,
        "timing": (
            "CUDA events for forward only, batch=1; excludes decode, "
            "normalization, transfers, argmax and saving"
        ),
        "tf32_enabled_in_pytorch": False,
        "note": (
            "Agreement and IoU use checkpoint predictions as reference, "
            "not ground truth. Existing TensorRT engine build precision "
            "is unchanged."
        ),
        "models": {
            name: {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in paths.items()
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2))
    aggregate = {name: np.zeros((3, 3), dtype=np.int64) for name in MODELS[1:]}
    times = {name: [] for name in MODELS}
    errors = {
        name: {"sum_abs": 0.0, "max_abs": 0.0, "elements": 0}
        for name in MODELS[1:]
    }
    rows = []
    start_time = time.perf_counter()
    with torch.inference_mode():
        sample = normalize(Image.open(files[0]).convert("RGB")).cuda()
        for name, model in models.items():
            x = sample.half() if name.endswith("fp16") else sample
            for _ in range(args.warmup):
                model(x)
            torch.cuda.synchronize()
            print(f"Warmed up {name}", flush=True)
        del sample, x
        for index, path in enumerate(files):
            with Image.open(path) as image:
                rgb = image.convert("RGB")
            x32 = normalize(rgb).cuda()
            x16 = x32.half()
            outputs, masks, per_image_times = {}, {}, {}
            # Rotate measurement order to reduce systematic ordering effects.
            order = MODELS[index % 3 :] + MODELS[: index % 3]
            for name in order:
                x = x16 if name.endswith("fp16") else x32
                events = []
                for _ in range(args.repeats):
                    before, after = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    before.record()
                    output = models[name](x)
                    after.record()
                    events.append((before, after))
                torch.cuda.synchronize()
                durations = [
                    before.elapsed_time(after) for before, after in events
                ]
                times[name].extend(durations)
                per_image_times[name] = float(np.mean(durations))
                if (
                    tuple(output.shape) != (1, 3, 1080, 1920)
                    or not torch.isfinite(output).all().item()
                ):
                    raise RuntimeError(
                        f"Invalid output from {name} for {path}"
                    )
                outputs[name] = output.float().cpu().numpy()[0]
                masks[name] = outputs[name].argmax(axis=0).astype(np.uint8)
                metadata["models"][name]["output_dtype"] = str(output.dtype)
                del output
            reference = masks[MODELS[0]]
            row = {"image": path.name}
            panels = [rgb.resize((640, 360))]
            titles = ["Original RGB"]
            rgb_array = np.array(rgb)
            for name in MODELS:
                mask = masks[name]
                Image.fromarray(mask).save(
                    args.output / name / "masks" / (path.stem + ".png")
                )
                overlay = rgb_array.copy()
                selected = mask != 0
                overlay[selected] = (
                    0.55 * rgb_array[selected] + 0.45 * COLORS[mask[selected]]
                ).astype(np.uint8)
                overlay = Image.fromarray(overlay)
                overlay.save(
                    args.output / name / "overlays" / (path.stem + ".jpg"),
                    quality=92,
                )
                panels.append(overlay.resize((640, 360)))
                row[name + "_forward_ms"] = per_image_times[name]
                title = name
                if name != MODELS[0]:
                    confusion = np.bincount(
                        3 * reference.ravel().astype(np.int64) + mask.ravel(),
                        minlength=9,
                    ).reshape(3, 3)
                    aggregate[name] += confusion
                    metrics = confusion_metrics(confusion)
                    row[name + "_agreement_pct"] = metrics[
                        "pixel_agreement_pct"
                    ]
                    row[name + "_mean_iou_vs_checkpoint"] = metrics[
                        "mean_iou_vs_checkpoint"
                    ]
                    for cls, iou in metrics["iou_vs_checkpoint"].items():
                        row[name + "_" + cls + "_iou_vs_checkpoint"] = iou
                    delta = np.abs(outputs[name] - outputs[MODELS[0]])
                    errors[name]["sum_abs"] += float(
                        delta.sum(dtype=np.float64)
                    )
                    errors[name]["elements"] += delta.size
                    errors[name]["max_abs"] = max(
                        errors[name]["max_abs"], float(delta.max())
                    )
                    row[name + "_logit_mae"] = float(
                        delta.mean(dtype=np.float64)
                    )
                    row[name + "_logit_max_abs_error"] = float(delta.max())
                    Image.fromarray(
                        (mask != reference).astype(np.uint8) * 255
                    ).save(
                        args.output
                        / name
                        / "disagreements"
                        / (path.stem + ".png")
                    )
                    title += (
                        f" | agreement {metrics['pixel_agreement_pct']:.3f}%"
                    )
                titles.append(title)
            sheet = Image.new("RGB", (1280, 780), "white")
            draw = ImageDraw.Draw(sheet)
            for i, (panel, title) in enumerate(zip(panels, titles)):
                px, py = (i % 2) * 640, (i // 2) * 390
                draw.text((px + 8, py + 8), title, fill="black")
                sheet.paste(panel, (px, py + 30))
            sheet.save(
                args.output / "comparisons" / (path.stem + ".jpg"), quality=93
            )
            rows.append(row)
            print(
                f"[{index + 1}/{len(files)}] FP32 agreement "
                f"{row['tensorrt_fp32_agreement_pct']:.4f}% | FP16 "
                f"{row['tensorrt_fp16_agreement_pct']:.4f}%",
                flush=True,
            )

    summary = {
        "image_count": len(files),
        "wall_time_seconds": time.perf_counter() - start_time,
        "models": {},
    }
    baseline_ms = float(np.mean(times[MODELS[0]]))
    for name in MODELS:
        values = times[name]
        result = {
            "mean_forward_ms": float(np.mean(values)),
            "median_forward_ms": float(np.median(values)),
            "p95_forward_ms": float(np.percentile(values, 95)),
            "fps_forward_only": 1000 / float(np.mean(values)),
            "speedup_vs_checkpoint": baseline_ms / float(np.mean(values)),
            "timed_forwards": len(values),
        }
        if name != MODELS[0]:
            result.update(confusion_metrics(aggregate[name]))
            result["logit_mae"] = (
                errors[name]["sum_abs"] / errors[name]["elements"]
            )
            result["logit_max_abs_error"] = errors[name]["max_abs"]
        summary["models"][name] = result
    with (args.output / "per_image.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2))
    report = [
        "# ERFNet export comparison",
        "",
        f"{len(files)} images, 1920x1080, batch 1, {metadata['device']}.",
        "",
        "All models use identical RGB per-image channel normalization. "
        "No resizing or cropping.",
        "",
        "IoU and agreement compare to checkpoint predictions, not ground "
        "truth; they measure export drift, not task accuracy loss.",
        "",
        "| Model | Forward ms | FPS | Speedup | Pixel agreement "
        "| Crop IoU | Weed IoU |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in summary["models"].items():
        agreement = (
            f"{m['pixel_agreement_pct']:.4f}%"
            if "pixel_agreement_pct" in m
            else "reference"
        )
        iou = m.get("iou_vs_checkpoint", {})
        vals = [
            f"{iou[c]:.6f}" if iou.get(c) is not None else "-"
            for c in ["crop", "weed"]
        ]
        report.append(
            f"| {name} | {m['mean_forward_ms']:.2f} | "
            f"{m['fps_forward_only']:.2f} | "
            f"{m['speedup_vs_checkpoint']:.2f}x | {agreement} | "
            f"{vals[0]} | {vals[1]} |"
        )
    report += [
        "",
        f"Timing uses synchronized CUDA events over {args.repeats} "
        f"forwards per image after {args.warmup} warmups per model. "
        "Decode, preprocessing, transfers and output saving are excluded. "
        "Model order rotates between images. Existing TensorRT build "
        "settings are preserved; PyTorch TF32 is disabled.",
        "",
        "Masks store class IDs: 0 background, 1 crop, 2 weed. "
        "Overlays show crop in green and weed in red. Disagreement maps "
        "are white wherever an export changes the checkpoint's "
        "class prediction.",
        "",
        "[Per-image metrics](per_image.csv) | "
        "[Summary JSON](summary.json) | [Run manifest](manifest.json)",
        "",
        "## Visual comparisons",
        "",
    ]
    for path in files:
        report.extend(
            [
                f"### {path.name}",
                "",
                f"![comparison](comparisons/{path.stem}.jpg)",
                "",
            ]
        )
    (args.output / "report.md").write_text("\n".join(report))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Results: {args.output}", flush=True)


if __name__ == "__main__":
    main()
