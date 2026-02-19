from typing import Dict

import torch
import torch.nn as nn

from modules.deeplab.modeling import deeplabv3plus_resnet50
from modules.erfnet.erfnet_modified import ERFNetModel
from modules.losses import get_criterion as get_criterion

# fmt: off
torch.set_float32_matmul_precision("high")  # use 2 BF16 nums for 1 FP32 FMA
torch.backends.cudnn.allow_tf32 = True      # use TF32 for convolutions
# fmt: on


def get_backbone(cfg: Dict) -> nn.Module:
    num_classes = cfg["backbone"]["num_classes"]
    pretrained = cfg["backbone"]["pretrained"]
    name = cfg["backbone"]["name"]
    if name == "erfnet":
        return ERFNetModel(num_classes, pretrained=pretrained)
    if name == "deeplabv3plus_resnet50":
        return deeplabv3plus_resnet50(
            num_classes=num_classes,
            pretrained_backbone=pretrained,
        )

    raise ValueError("The requested backbone is not supported.")
