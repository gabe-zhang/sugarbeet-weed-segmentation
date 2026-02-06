from typing import Dict

import lightning.pytorch as pl

from .mydataset import MyDatasetModule
from .pdc import PDCModule


def get_data_module(cfg: Dict) -> pl.LightningDataModule:
    dataset_name = cfg["data"]["name"]
    if dataset_name == "phenobench":
        return PDCModule(cfg)
    else:
        return MyDatasetModule(cfg)
