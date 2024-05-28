from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.base import BasicBlock, IntermediateSequential
from ..modules.pyramid import ASPP_v3
from ._base import DeepLabHead


class DeepLabV3(DeepLabHead):
    """DeepLab v3: Dilated ResNet with multi-grid + improved ASPP"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        middle_dim: Optional[int],
        backbone: IntermediateSequential,
        atrous_rates: List[int],
        init_type: Optional[str],
    ):
        super().__init__(
            input_dim,
            output_dim,
            middle_dim,
            backbone,
            ASPP_v3,
            atrous_rates,
            init_type,
        )

        self.project = BasicBlock(
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
        return F.sigmoid(
            F.interpolate(
                self.project(super().forward(input)),
                size=input.size()[-2:],
                mode="bilinear",
                align_corners=False,
            )
        )


class DeepLabV3Plus(DeepLabV3):
    """DeepLab v3+: Dilated ResNet with multi-grid + improved ASPP + decoder."""

    reduce_dim = 48

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        middle_dim: Optional[int],
        backbone: nn.Sequential,
        atrous_rates: List[int],
        init_type: Optional[str],
        return_intermediate_index: int,
    ):
        super().__init__(
            input_dim, output_dim, middle_dim, backbone, atrous_rates, init_type
        )
        self.return_intermediate_index = return_intermediate_index

        self.reduce = BasicBlock(
            self.middle_dim,
            self.reduce_dim,
            kernel_size=1,
            padding=0,
            is_bias=False,
            init_type=init_type,
        )
        self.project = nn.Sequential(
            BasicBlock(
                self.middle_dim + self.reduce_dim,
                self.middle_dim,
                is_bias=False,
                init_type=init_type,
            ),
            BasicBlock(
                self.middle_dim, self.middle_dim, is_bias=False, init_type=init_type
            ),
            self.project,
        )

    def forward(self, input: torch.Tensor):
        x_list = self.backbone(input)
        x_ = self.reduce(x_list[self.return_intermediate_index])
        x = F.interpolate(
            self.ASPP(x_list[-1]),
            size=x_.size()[-2:],
            mode="bilinear",
            align_corners=True,
        )
        x = F.interpolate(
            self.project(torch.cat((x, x_), dim=1)),
            size=input.size()[-2:],
            mode="bilinear",
            align_corners=True,
        )
        return F.sigmoid(x)
