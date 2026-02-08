from typing import Dict

import torch.nn as nn

from modules.erfnet.erfnet_modified import ERFNetModel
from modules.losses import get_criterion


def get_backbone(cfg: Dict) -> nn.Module:
    num_classes = cfg["backbone"]["num_classes"]
    pretrained = cfg["backbone"]["pretrained"]
    if cfg["backbone"]["name"] == "erfnet":
        return ERFNetModel(num_classes, pretrained=pretrained)

    raise ValueError("The requested backbone is not supported.")
