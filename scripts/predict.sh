#!/bin/bash

uv run src/predict.py \
    --config config/erfnet_predict.yaml \
    --ckpt_path models/semantic-seg-erfnet.ckpt \
    --export_dir prediction/semantics