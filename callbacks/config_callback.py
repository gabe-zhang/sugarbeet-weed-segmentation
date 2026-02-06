import os
from typing import Dict

import oyaml as yaml
from lightning.pytorch.callbacks import Callback


class ConfigCallback(Callback):
    """Callback to save the config file of a model."""

    def __init__(self, cfg: Dict):
        self.cfg = cfg

    def setup(self, trainer, pl_module, stage=None) -> None:
        log_dir = trainer.log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        fpath = os.path.join(log_dir, "config.yaml")
        with open(fpath, "w") as ostream:
            yaml.dump(self.cfg, ostream)
