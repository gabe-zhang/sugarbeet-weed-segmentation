from typing import Dict

import lightning.pytorch as pl

from .dataset import SegmentationDataModule


def get_data_module(cfg: Dict) -> pl.LightningDataModule:
    dataset_name = cfg["data"]["name"]
    if dataset_name == "phenobench":
        return SegmentationDataModule(cfg)
    raise ValueError(f"Unknown dataset: {dataset_name}")
