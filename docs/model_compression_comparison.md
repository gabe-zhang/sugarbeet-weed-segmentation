# ERFNet compression comparison

The TensorRT exports closely preserve the original pretrained checkpoint's
predictions on this image set. The results suggest **marginal degradation after
compression**, with FP16 changing only **0.0393% of pixels** while providing a
**6.64× inference speedup**. FP32 predictions are effectively identical.

These results measure agreement with the original model. Ground-truth labels
would be needed to establish the actual segmentation accuracy change.

## Results

Evaluated on 2026-10-27 using all **47 images** from
`output/2026-10-22_12-38/images`: **1920×1080**, batch size **1**, NVIDIA
**Xavier**. The comparison covers **97,459,200 pixels**.

| Model | Mean GPU ms/image | FPS | Speedup | Pixels changed vs. checkpoint |
|---|---:|---:|---:|---:|
| Original checkpoint (FP32) | 600.4 | 1.67 | 1.00× | Reference |
| TensorRT FP32 | 332.5 | 3.01 | 1.81× | 6 (0.000006%) |
| TensorRT FP16 | 90.4 | 11.06 | 6.64× | 38,273 (0.0393%) |

Per-class IoU helps assess differences in crop and weed predictions separately
from the much larger background area. All values below use the checkpoint's
predictions as the reference, rather than ground truth.

| Export | Pixel agreement | Crop IoU | Weed IoU |
|---|---:|---:|---:|
| TensorRT FP32 | 99.999994% | 99.99985% | 99.99985% |
| TensorRT FP16 | 99.9607% | 99.26% | 98.25% |

The largest FP16 disagreement on an individual image was **0.1594% of pixels**.
The observed differences are small on this set, although consistency with the
checkpoint does not establish performance on other datasets.

## Method and artifacts

- Checkpoint: `models/semantic-seg-erfnet.ckpt`.
- FP32 export: `models/erfnet_tensorrt.ts`.
- FP16 export: `models/erfnet_half_tensorrt.ts`.
- Existing exports were evaluated directly, without fine-tuning or recompilation.
- All models received identical RGB preprocessing: divide by 255, then normalize
  each image channel by its own mean and sample standard deviation. Inputs used
  contiguous NCHW layout with no resizing or cropping.
- GPU forward timing used CUDA events, 10 warmups per model, and three timed
  passes per image. Model order rotated between images. Timing excludes image
  loading, preprocessing, transfers, argmax, and saving; PyTorch TF32 was disabled.
- Aggregate agreement and confusion matrices were independently verified against
  all saved prediction masks.

The reusable runner is [src/compare_erfnet.py](../src/compare_erfnet.py).
Local outputs are under
`prediction/erfnet_comparison_2026-10-22_12-38/`: `report.md` includes visual
comparisons and the largest FP16 differences; `per_image.csv`, `summary.json`, and
`manifest.json` record metrics and run details. Masks, overlays, and disagreement
maps are saved per model. These generated outputs are ignored by Git.
