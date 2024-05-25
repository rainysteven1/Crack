from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbone.resnet import resnet101
from ..modules.pyramid import ASPP_v2


class DeepLabV2(nn.Module):
    """DeepLabV2: Dilated ResNet + ASPP"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        pretrained: bool,
        strides: Optional[List[int]],
        dilations: Optional[List[int]],
        atrous_rates: List[int],
        init_type: Optional[str],
    ) -> None:
        super().__init__()

        self.backbone = resnet101(input_dim, pretrained, strides, dilations)
        aspp_input_dim = (
            self.backbone.stage_cfg["dims"][-1] * self.backbone.block_type.expansion
        )
        self.ASPP = ASPP_v2(
            aspp_input_dim, output_dim, atrous_rates, init_type=init_type
        )

    def forward(self, input: torch.Tensor):
        x = self.ASPP(self.backbone(input))
        return F.sigmoid(
            F.interpolate(
                x, size=input.size()[-2:], mode="bilinear", align_corners=True
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
