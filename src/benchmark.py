"""Benchmark script to compare inference speed between PyTorch and TensorRT.

Compares: erfnet.ckpt (PyTorch), erfnet_tensorrt.ts (FP32),
erfnet_half_tensorrt.ts (FP16)
"""

import argparse
import os
import sys
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Dict, List, Tuple

import oyaml as yaml
import torch
import torch_tensorrt  # noqa: F401

from datasets import get_data_module
from modules import get_backbone


def parse_args() -> Dict[str, str]:
    parser = argparse.ArgumentParser(description="Benchmark ERFNet models")
    parser.add_argument(
        "--config",
        default="config/erfnet_predict.yaml",
        help="Path to configuration file (*.yaml)",
    )
    parser.add_argument(
        "--num_warmup",
        type=int,
        default=10,
        help="Number of warm-up iterations",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=100,
        help="Number of inference runs for timing",
    )
    return vars(parser.parse_args())


def load_config(path_to_config_file: str) -> Dict:
    assert os.path.exists(path_to_config_file), f"Config not found: {path_to_config_file}"
    with open(path_to_config_file) as istream:
        config = yaml.safe_load(istream)
    return config


class TensorRTWrapper:
    """Wrapper to make TensorRT model compatible with the benchmark interface."""

    def __init__(self, model_path: str, num_classes: int = 3):
        self.network = torch.jit.load(model_path)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def eval(self):
        return self

    def to(self, device):
        self.network = self.network.to(device)
        return self

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


def load_models(cfg: Dict, device: torch.device) -> Dict[str, torch.nn.Module]:
    """Load all three models for benchmarking."""
    models = {}

    # 1. PyTorch ERFNet (baseline)
    print("Loading PyTorch ERFNet from checkpoint...")
    pytorch_model = get_backbone(cfg)
    ckpt = torch.load("models/semantic-seg-erfnet.ckpt", map_location=device)
    # Extract the network weights from the checkpoint
    state_dict = {}
    for key, value in ckpt["state_dict"].items():
        if key.startswith("network."):
            state_dict[key.replace("network.", "")] = value
    pytorch_model.load_state_dict(state_dict)
    pytorch_model = pytorch_model.to(device).eval()
    models["pytorch_erfnet"] = pytorch_model

    # 2. TensorRT FP32
    print("Loading TensorRT FP32 model...")
    trt_fp32 = TensorRTWrapper("models/erfnet_tensorrt.ts", num_classes=3)
    trt_fp32 = trt_fp32.to(device).eval()
    models["tensorrt_fp32"] = trt_fp32

    # 3. TensorRT FP16 (half precision)
    print("Loading TensorRT FP16 model...")
    trt_fp16 = TensorRTWrapper("models/erfnet_half_tensorrt.ts", num_classes=3)
    trt_fp16 = trt_fp16.to(device).eval()
    models["tensorrt_fp16"] = trt_fp16

    return models


def warmup(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    device: torch.device,
    num_iterations: int,
    use_half: bool = False,
) -> None:
    """Warm up the model with empty (zeros) input tensors."""
    print(f"  Warming up with {num_iterations} iterations...")
    dummy_input = torch.zeros(input_shape, device=device)
    if use_half:
        dummy_input = dummy_input.half()

    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)
    torch.cuda.synchronize()


def benchmark_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_runs: int,
    use_half: bool = False,
) -> Dict[str, float]:
    """Benchmark a model using CUDA events for accurate GPU timing."""
    # Get a sample batch for shape
    sample_batch = next(iter(dataloader))
    input_tensor = sample_batch["input_image"].to(device)
    if use_half:
        input_tensor = input_tensor.half()

    # Create CUDA events for timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    times = []

    with torch.no_grad():
        for i in range(num_runs):
            # Synchronize before starting timer
            torch.cuda.synchronize()
            start_event.record()

            _ = model(input_tensor)

            end_event.record()
            torch.cuda.synchronize()

            elapsed_ms = start_event.elapsed_time(end_event)
            times.append(elapsed_ms)

    # Calculate statistics
    times_tensor = torch.tensor(times)
    return {
        "mean_ms": float(times_tensor.mean()),
        "std_ms": float(times_tensor.std()),
        "min_ms": float(times_tensor.min()),
        "max_ms": float(times_tensor.max()),
        "fps": 1000.0 / float(times_tensor.mean()),
    }


def benchmark_on_dataset(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    use_half: bool = False,
) -> Dict[str, float]:
    """Benchmark model on actual dataset images."""
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    total_time_ms = 0.0
    num_images = 0

    with torch.no_grad():
        for batch in dataloader:
            input_tensor = batch["input_image"].to(device)
            if use_half:
                input_tensor = input_tensor.half()

            torch.cuda.synchronize()
            start_event.record()

            _ = model(input_tensor)

            end_event.record()
            torch.cuda.synchronize()

            total_time_ms += start_event.elapsed_time(end_event)
            num_images += input_tensor.shape[0]

    avg_time_ms = total_time_ms / num_images
    return {
        "total_time_ms": total_time_ms,
        "num_images": num_images,
        "avg_time_ms": avg_time_ms,
        "fps": 1000.0 / avg_time_ms,
    }


def print_results(results: Dict[str, Dict[str, float]], title: str) -> None:
    """Print benchmark results in a formatted table."""
    print(f"\n{'=' * 70}")
    print(f"{title:^70}")
    print("=" * 70)

    # Check if results have std_ms (repeated runs) or avg_time_ms (dataset)
    has_std = any("std_ms" in m for m in results.values())

    if has_std:
        print(f"{'Model':<20} {'Mean (ms)':<12} {'Std (ms)':<12} {'FPS':<12} {'Speedup':<12}")
    else:
        print(f"{'Model':<20} {'Avg (ms)':<12} {'Total (ms)':<12} {'FPS':<12} {'Speedup':<12}")
    print("-" * 70)

    baseline_fps = results.get("pytorch_erfnet", {}).get("fps", 1.0)

    for model_name, metrics in results.items():
        speedup = metrics["fps"] / baseline_fps if baseline_fps > 0 else 0
        if has_std:
            print(
                f"{model_name:<20} "
                f"{metrics['mean_ms']:<12.2f} "
                f"{metrics.get('std_ms', 0):<12.2f} "
                f"{metrics['fps']:<12.1f} "
                f"{speedup:<12.2f}x"
            )
        else:
            print(
                f"{model_name:<20} "
                f"{metrics['avg_time_ms']:<12.2f} "
                f"{metrics.get('total_time_ms', 0):<12.0f} "
                f"{metrics['fps']:<12.1f} "
                f"{speedup:<12.2f}x"
            )
    print("=" * 70)


def main():
    args = parse_args()
    cfg = load_config(args["config"])

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type != "cuda":
        print("WARNING: CUDA not available. Benchmarking on CPU may not be representative.")

    # Load dataset
    print("\nLoading dataset...")
    datamodule = get_data_module(cfg)
    datamodule.setup(stage="predict")
    dataloader = datamodule.predict_dataloader()
    print(f"Dataset: {len(datamodule)} images")

    # Get input shape from first batch
    sample_batch = next(iter(dataloader))
    input_shape = sample_batch["input_image"].shape
    print(f"Input shape: {input_shape}")

    # Load all models
    print("\nLoading models...")
    models = load_models(cfg, device)

    # Benchmark each model
    print("\n" + "=" * 70)
    print("BENCHMARKING")
    print("=" * 70)

    repeated_results = {}
    dataset_results = {}

    for model_name, model in models.items():
        print(f"\n[{model_name}]")
        use_half = "fp16" in model_name

        # Warm-up with empty input
        warmup(model, input_shape, device, args["num_warmup"], use_half=use_half)

        # Benchmark with repeated runs on same input
        print(f"  Running {args['num_runs']} repeated inference runs...")
        repeated_results[model_name] = benchmark_model(
            model, dataloader, device, args["num_runs"], use_half=use_half
        )

        # Benchmark on full dataset
        print(f"  Running inference on full dataset...")
        dataset_results[model_name] = benchmark_on_dataset(
            model, dataloader, device, use_half=use_half
        )

    # Print results
    print_results(repeated_results, "REPEATED INFERENCE (Same Input)")
    print_results(dataset_results, "FULL DATASET INFERENCE")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    baseline = repeated_results["pytorch_erfnet"]["mean_ms"]
    for model_name, metrics in repeated_results.items():
        speedup = baseline / metrics["mean_ms"]
        print(f"{model_name}: {metrics['mean_ms']:.2f}ms ({speedup:.2f}x vs PyTorch)")


if __name__ == "__main__":
    main()
