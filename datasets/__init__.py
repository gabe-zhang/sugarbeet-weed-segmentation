from typing import Dict

import pytorch_lightning as pl

from .pdc import PDCModule
from .mydataset import MyDatasetModule


def get_data_module(cfg: Dict) -> pl.LightningDataModule:
	dataset_name = cfg['data']['name']
	if dataset_name == 'phenobench':
		return PDCModule(cfg)
	else:
		return MyDatasetModule(cfg)
