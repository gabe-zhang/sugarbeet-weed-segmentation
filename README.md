# Sugarbeet Weed Segmentation

A semantic segmentation framework for agricultural weed detection in **sugarbeet** fields, classifying pixels into three categories: **soil** (background), **crop** (sugarbeet), and **weed**.

> **Note**: This model is trained specifically on sugarbeet imagery from the PhenoBench dataset and is not designed for other crop types.

## Branches

| Branch | Target Platform | Python | Description |
|--------|-----------------|--------|-------------|
| `main` | Desktop / Cloud GPU | 3.9+ | Full training and inference with `uv` managed dependencies |
| `brain` | NVIDIA Jetson Xavier NX | 3.8 | Edge deployment with TensorRT optimization |

### Main Branch
- Full dependency management via `uv add`
- Training, validation, and inference scripts
- Supports ERFNet, DeepLabV3+, and UNet architectures

### Brain Branch
- Dependencies managed via `uv pip install` with `--system-site-packages`
- TensorRT model conversion (`conversion.ipynb`)
- Benchmark script for PyTorch vs TensorRT comparison (`benchmark.py`)
- Optimized for real-time inference on edge devices

## Setup

### Main Branch (Desktop/Cloud)

```bash
# Clone and enter directory
git clone https://github.com/gabe-zhang/sugarbeet-weed-segmentation.git
cd sugarbeet-weed-segmentation

# Install dependencies with uv
uv sync
```

### Brain Branch (Jetson Xavier NX)

```bash
# Ensure system packages are available (CUDA, TensorRT, PyTorch)
# Then install Python dependencies
uv venv --system-site-packages
uv pip install -r requirements.txt
```

### Dataset Configuration

Update the dataset path in the configuration files:
```bash
./config/config_erfnet.yaml
./config/config_deeplab.yaml
```

## Project Structure

```
sugarbeet-weed-segmentation/
├── src/              # Entry point scripts
│   ├── train.py      # Training script
│   ├── test.py       # Testing script
│   ├── val.py        # Validation script
│   └── predict.py    # Prediction script
├── config/           # YAML configurations
├── models/           # Pretrained checkpoints
├── modules/          # Model architectures (ERFNet, DeepLabV3+, UNet)
├── datasets/         # Data loaders and augmentations
├── callbacks/        # Training callbacks
├── scripts/          # Shell scripts
│   └── val.sh        # Validation runner
└── tools/            # Python utility scripts
    ├── calculate_class_weights.py
    ├── calculate_means_stds.py
    └── visualize.py
```

## Usage

### Train ERFNet

```bash
uv run src/train.py --config ./config/config_erfnet.yaml --export_dir <path-to-export-directory>
```

### Train DeepLabV3+

```bash
uv run src/train.py --config ./config/config_deeplab.yaml --export_dir <path-to-export-directory>
```

### Test

```bash
uv run src/test.py --config ./config/config_erfnet.yaml --ckpt_path <path-to-ckpt> --export_dir <path-to-export-directory>
```

### Inference

```bash
uv run src/predict.py --config ./config/config_erfnet.yaml --ckpt_path <path-to-ckpt> --export_dir <path-to-export-directory>
```

### Benchmark (Brain Branch Only)

```bash
uv run src/benchmark.py --config config/erfnet_predict.yaml --num_warmup 10 --num_passes 3
```

## Pretrained Models

Pretrained weights from PRBonn:
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

This project is based on [PRBonn/phenobench-baselines](https://github.com/PRBonn/phenobench-baselines/tree/main/semantic_segmentation) and uses the [PhenoBench dataset](https://www.phenobench.org/) (licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)). The original codebase has been modified for custom training, edge deployment, and testing purposes.

If you use this code, please cite the original PhenoBench work and the respective model architectures (see [CITATION.md](CITATION.md)).
