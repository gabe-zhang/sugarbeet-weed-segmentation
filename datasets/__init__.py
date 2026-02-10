from typing import Dict

import lightning.pytorch as pl

from .pdc import PhenoBenchModule


def get_data_module(cfg: Dict) -> pl.LightningDataModule:
    dataset_name = cfg["data"]["name"]
    if dataset_name == "phenobench":
        return PhenoBenchModule(cfg)
    raise ValueError(f"Unknown dataset: {dataset_name}")
