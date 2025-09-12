#!/bin/bash

uv run val.py \
    --config /home/zy/proj/weed-detect/semantic_segmentation/config/config_erfnet.yaml \
    --ckpt_path /home/zy/proj/weed-detect/semantic_segmentation/models/semantic-seg-erfnet.ckpt \
    --export_dir /home/zy/proj/weed-detect/semantic_segmentation/prediction/semantics