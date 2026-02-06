#!/bin/bash

uv run src/train.py \
    --config config/erfnet_finetune_phenobench.yaml \
    --ckpt_path models/semantic-seg-erfnet.ckpt \
    --export_dir runs
