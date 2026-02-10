#!/bin/bash

uv run src/predict.py \
    --config config/config_erfnet.yaml \
    --ckpt_path models/semantic-seg-erfnet.ckpt \
    --export_dir runs