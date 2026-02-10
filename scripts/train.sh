#!/bin/bash

uv run src/train.py \
    --config config/config_erfnet.yaml \
    --ckpt_path models/semantic-seg-erfnet.ckpt \
    --export_dir runs \
    --resume false
