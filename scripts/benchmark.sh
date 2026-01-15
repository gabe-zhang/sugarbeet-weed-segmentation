#!/bin/bash

# Benchmark script for comparing PyTorch vs TensorRT ERFNet models
# Run on NVIDIA Jetson Xavier NX

uv run benchmark.py \
    --config config/erfnet_predict.yaml \
    --num_warmup 10 \
    --num_runs 100
