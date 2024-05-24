from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.base import OutputBlock
from ..modules.pyramid import ASPP_v2
from ..modules.resnet import resnet101


class DeepLabV2(nn.Module):
    """DeepLabV2: Dilated ResNet + ASPP

    ASPP output stride is fixed at 8.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        pretrained: bool,
        atrous_rates: list[int],
        init_type: Optional[str],
    ) -> None:
        super().__init__()

        self.backbone = resnet101(input_dim, pretrained)
        self.ASPP = ASPP_v2(2048, output_dim, atrous_rates, init_type=init_type)
        self.output_block = OutputBlock(output_dim, output_dim, init_type=init_type)

    def forward(self, input: torch.Tensor):
        x = self.ASPP(self.backbone(input))
        return self.output_block(
            F.interpolate(
                x, size=input.size()[-2:], mode="bilinear", align_corners=False
            )
        )

    def get_params(self, keywords: Dict) -> list[Dict]:
        params = [
            {
                "params": self.backbone.get_params(),
                "lr": keywords["lr"],
                "weight_decay": keywords["weight_decay"],
            },
            {
                "params": self.ASPP.get_weight(),
                "lr": 10 * keywords["lr"],
                "weight_decay": keywords["weight_decay"],
            },
        ]
        if self.ASPP.is_bias:
            params.append(
                {
                    "params": self.ASPP.get_bias(),
                    "lr": 20 * keywords["lr"],
                    "weight_decay": 0.0,
                },
            )
        return params

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def freeze_backbone_bn(self):
        self.backbone.freeze_bn()
