#!/bin/bash
# Run from project root directory

uv run src/val.py \
    --config ./config/config_erfnet.yaml \
    --ckpt_path ./models/semantic-seg-erfnet.ckpt \
    --export_dir ./prediction/semantics
