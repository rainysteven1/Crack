from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.base import BasicBlock
from ..modules.pyramid import ASPP_v3
from ..modules.resnet import resnet101


class DeepLabV3(nn.Module):
    """DeepLab v3: Dilated ResNet with multi-grid + improved ASPP"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        pretrained: bool,
        strides: Optional[List[int]],
        dilations: Optional[List[int]],
        multi_grids: Optional[List[int]],
        atrous_rates: List[int],
        init_type: Optional[str],
        return_intermediate: bool = False,
    ):
        super().__init__()
        self.middle_dim = 256

        self.backbone = resnet101(
            input_dim, pretrained, strides, dilations, multi_grids, return_intermediate
        )

        aspp_input_dim = (
            self.backbone.stage_cfg["dims"][-1] * self.backbone.block_type.expansion
        )
        self.ASPP = ASPP_v3(
            aspp_input_dim, self.middle_dim, atrous_rates, init_type=init_type
        )

        concat_dim = self.middle_dim * (len(atrous_rates) + 2)
        self.fc1 = BasicBlock(
            concat_dim,
            self.middle_dim,
            kernel_size=1,
            padding=0,
            is_bias=False,
            init_type=init_type,
        )
        self.fc2 = BasicBlock(
            self.middle_dim,
            output_dim,
            kernel_size=1,
            padding=0,
            is_bn=False,
            is_relu=False,
            is_bias=False,
            init_type=init_type,
        )

    def forward(self, input: torch.Tensor):
        x = self.ASPP(self.backbone(input))
        x = self.fc2(self.fc1(x))
        return F.sigmoid(
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


class DeepLabV3Plus(DeepLabV3):
    """DeepLab v3+: Dilated ResNet with multi-grid + improved ASPP + decoder."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        pretrained: bool,
        strides: Optional[List[int]],
        dilations: Optional[List[int]],
        multi_grids: Optional[List[int]],
        atrous_rates: List[int],
        init_type: Optional[str],
    ):
        super().__init__(
            input_dim,
            output_dim,
            pretrained,
            strides,
            dilations,
            multi_grids,
            atrous_rates,
            init_type,
            return_intermediate=True,
        )
        reduce_dim = 48

        self.reduce = BasicBlock(
            self.middle_dim,
            reduce_dim,
            kernel_size=1,
            padding=0,
            is_bias=False,
            init_type=init_type,
        )
        self.fc2 = nn.Sequential(
            BasicBlock(
                self.middle_dim + reduce_dim,
                self.middle_dim,
                is_bias=False,
                init_type=init_type,
            ),
            BasicBlock(
                self.middle_dim, self.middle_dim, is_bias=False, init_type=init_type
            ),
            self.fc2,
        )

    def forward(self, input: torch.Tensor):
        x_list = self.backbone(input)
        x_ = self.reduce(x_list[1])
        x = F.interpolate(
            self.fc1(self.ASPP(x_list[-1])),
            size=x_.size()[-2:],
            mode="bilinear",
            align_corners=True,
        )
        x = torch.cat((x, x_), dim=1)
        x = F.interpolate(
            self.fc2(x), size=input.size()[-2:], mode="bilinear", align_corners=True
        )
        return F.sigmoid(x)
