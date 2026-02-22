# Sugarbeet Weed Segmentation

[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](LICENSE) [![Python](https://img.shields.io/badge/Python-%3E%3D3.12-blue.svg)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-orange.svg)](https://pytorch.org/) [![Lightning](https://img.shields.io/badge/Lightning-2.6.1-blueviolet.svg)](https://lightning.ai/) [![W&B](https://img.shields.io/badge/W%26B-sugarbeet--weed--segmentation-ff69b4?logo=wandb&logoColor=white)](https://wandb.ai/yuanzzhang/sugarbeet-weed-segmentation)

A segmentation framework for agricultural weed detection in **sugarbeet** fields, supporting **semantic**, **instance**, and **panoptic** segmentation across three categories: **soil** (background), **crop** (sugarbeet), and **weed**.

Trained on the [PhenoBench dataset](https://www.phenobench.org/) with CodaLab benchmark submission support.


## Models

| Task | Model | Framework |
|------|-------|-----------|
| Semantic segmentation | [ERFNet](https://github.com/Eromera/erfnet_pytorch), [DeepLabV3+](https://github.com/VainF/DeepLabV3Plus-Pytorch) | PyTorch Lightning |
| Instance / Panoptic segmentation | [YOLO26](https://docs.ultralytics.com/) | Ultralytics |

## Branches

| Branch | Platform | Description |
|--------|----------|-------------|
| `main` | Desktop / Cloud GPU | Training, inference, and submission scripts |
| `brain` | NVIDIA Jetson Xavier NX | TensorRT edge deployment |

## Setup

```bash
git clone https://github.com/gabe-zhang/sugarbeet-weed-segmentation.git
cd sugarbeet-weed-segmentation
uv sync
```

Download [PhenoBench](https://www.phenobench.org/) and update the data path in the YAML config files under `config/`.

## Usage

### Semantic Segmentation (ERFNet / DeepLabV3+)

```bash
# Train
uv run src/train.py --config config/<config>.yaml --export_dir runs

# Validate / submit
uv run src/submit.py \
    --config config/<config>.yaml \
    --ckpt_path runs/<run>/checkpoints/<best>.ckpt \
    --export_dir runs --split val --tta
```

Supports ensemble (`--config`/`--ckpt_path` repeated), TTA (`--tta`, `--tta_scales`), and morphological postprocessing (`--opening`).

### Instance / Panoptic Segmentation (YOLO26)

```bash
# Convert PhenoBench to YOLO format
uv run tools/convert_phenobench_to_yolo.py

# Train
uv run src/train_yolo.py --config config/yolo26x-baseline.yaml

# Predict (instance -> semantic)
uv run src/predict_yolo.py \
    --weights runs/<run>/weights/best.pt --split val

# Panoptic submission
uv run src/submit_panoptic.py \
    --weights runs/<run>/weights/best.pt --split test
```

### Validate Submission

```bash
uv run tools/validator.py --task semantics \
    --phenobench_dir data/PhenoBench --zipfile <submission>.zip
```

## Classes

| Class | ID | Color (RGB) |
|-------|------|-------------|
| Soil (background) | 0 | (0, 0, 0) |
| Crop (sugarbeet) | 1 | (0, 255, 0) |
| Weed | 2 | (255, 0, 0) |

## Benchmark (Jetson Xavier NX)

Input: 1920x1080 | ERFNet model

| Model | Latency (ms) | FPS | Speedup |
|-------|-------------|-----|---------|
| PyTorch | 717.9 | 1.4 | 1.0x |
| TensorRT FP32 | 309.3 | 3.2 | 2.3x |
| TensorRT FP16 | 83.6 | 12.0 | 8.6x |

## License

This project is licensed under the [MIT License](LICENSE).

## Attribution

Code adapted from [PRBonn/phenobench-baselines](https://github.com/PRBonn/phenobench-baselines/tree/main/semantic_segmentation). Trained on the [PhenoBench dataset](https://www.phenobench.org/) ([CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)).

If you use the PhenoBench dataset and/or models, please cite accordingly (see [CITATION.md](CITATION.md)).
