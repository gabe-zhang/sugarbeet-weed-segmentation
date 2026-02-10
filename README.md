# Sugarbeet Weed Segmentation

A semantic segmentation framework for agricultural weed detection in **sugarbeet** fields, classifying pixels into three categories: **soil** (background), **crop** (sugarbeet), and **weed**.

> **Note**: Models are trained specifically on sugarbeet imagery from the PhenoBench dataset and are not designed for other crop types.

## Branches

| Branch | Target Platform | Python | Description |
|--------|-----------------|--------|-------------|
| `main` | Desktop / Cloud GPU | 3.12+ | Full training and inference with `uv` managed dependencies |
| `brain` | NVIDIA Jetson Xavier NX | 3.8 | Edge deployment with TensorRT optimization |

### Main Branch
- Full dependency management via `uv add`
- Training, validation, and inference scripts
- Supports [ERFNet](https://github.com/Eromera/erfnet_pytorch) and [DeepLabV3+](https://github.com/VainF/DeepLabV3Plus-Pytorch)

### Brain Branch
- Dependencies managed via `uv pip install` with `--system-site-packages`
- TensorRT model conversion (`conversion.ipynb`)
- Benchmark script for PyTorch vs TensorRT comparison (`benchmark.py`)
- Optimized for real-time inference on edge devices

## Setup

### Main Branch (Desktop/Cloud)

```bash
git clone https://github.com/gabe-zhang/sugarbeet-weed-segmentation.git
cd sugarbeet-weed-segmentation
uv sync
```

### Brain Branch (Jetson Xavier NX)

```bash
# Ensure system packages are available (CUDA, TensorRT, PyTorch)
uv venv --system-site-packages
uv pip install -r requirements.txt
```

### Dataset Configuration

Update `data.path_to_dataset` in the YAML config files under `config/`.

## Project Structure

```
sugarbeet-weed-segmentation/
├── src/              # Entry points (train, val, predict)
├── config/           # YAML experiment configs
├── models/           # Pretrained checkpoints
├── runs/             # Training outputs and logs
├── modules/          # Model architectures and losses
├── datasets/         # Data loaders and augmentations
├── callbacks/        # Training callbacks
├── scripts/          # Shell scripts
└── tools/            # Utility scripts
```

## Usage

Shell scripts under `scripts/` wrap common workflows:

```bash
./scripts/train.sh    # Train a model
./scripts/val.sh      # Run validation
./scripts/predict.sh  # Run inference
```

Script calls `uv run src/<script>.py` with these flags:

| Flag | Description |
|------|-------------|
| `--config` | Path to YAML experiment config |
| `--ckpt_path` | Checkpoint path (auto-downloaded if missing) |
| `--export_dir` | Output directory for logs and checkpoints |
| `--resume` | Resume training (auto-detects last checkpoint) |

## Pretrained Models

Pretrained weights from PRBonn (auto-downloaded when using `--ckpt_path`):
- [ERFNet](https://www.ipb.uni-bonn.de/html/projects/phenobench/semantic_segmentation/semantic-seg-erfnet.ckpt) (24 MB)
- [DeepLabV3+](https://www.ipb.uni-bonn.de/html/projects/phenobench/semantic_segmentation/semantic-seg-deeplab.ckpt) (456 MB)

## Benchmark (Jetson Xavier NX)

Input: 1920×1080 | ERFNet model

| Model | Latency (ms) | FPS | Speedup |
|-------|-------------|-----|---------|
| PyTorch | 717.9 | 1.4 | 1.0× |
| TensorRT FP32 | 309.3 | 3.2 | 2.3× |
| TensorRT FP16 | 83.6 | 12.0 | 8.6× |

## Classes

| Class | ID | Color (RGB) |
|-------|------|-------------|
| Soil (background) | 0 | (0, 0, 0) |
| Crop (sugarbeet) | 1 | (0, 255, 0) |
| Weed | 2 | (255, 0, 0) |

## License

This project is licensed under the [MIT License](LICENSE).

## Attribution

Code adapted from [PRBonn/phenobench-baselines](https://github.com/PRBonn/phenobench-baselines/tree/main/semantic_segmentation). Trained on the [PhenoBench dataset](https://www.phenobench.org/) ([CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)).

If you use the PhenoBench dataset and/or models, please cite accordingly (see [CITATION.md](CITATION.md)).
