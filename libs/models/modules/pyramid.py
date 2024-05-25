from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BasicBlock


class _ASPPPooling(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        init_type: Optional[str],
    ) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            BasicBlock(
                input_dim,
                output_dim,
                kernel_size=1,
                padding=0,
                is_bias=False,
                init_type=init_type,
            ),
        )

    def forward(self, input: torch.Tensor):
        return F.interpolate(
            self.layers(input),
            size=input.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )


class ASPP_v2(nn.Module):
    """Atrous spatial pyramid pooling (ASPP)
    version: DeepLabV2
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        atrous_rates: list[int],
        init_type: Optional[str],
    ) -> None:
        super().__init__()
        self.is_bias = not (init_type is not None and init_type == "kaiming")

        self.layers = nn.ModuleList(
            [
                BasicBlock(
                    input_dim,
                    output_dim,
                    padding=rate,
                    dilation=rate,
                    is_bn=not self.is_bias,
                    is_relu=not self.is_bias,
                    is_bias=self.is_bias,
                    init_type=init_type,
                )
                for rate in atrous_rates
            ]
        )

    def forward(self, input: torch.Tensor):
        return sum([layer(input) for layer in self.layers])

    def get_weight(self):
        for layer in self.layers:
            for module in layer.modules():
                if isinstance(module, nn.Conv2d):
                    yield module.weight

    def get_bias(self):
        for layer in self.layers:
            for module in layer.modules():
                if isinstance(module, nn.Conv2d):
                    yield module.bias


class ASPP_v3(nn.Module):
    """Atrous spatial pyramid pooling with image-level feature
    version: DeepLabV3
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        atrous_rates: list[int],
        init_type: Optional[str],
    ) -> None:
        super().__init__()

        self.module_list = nn.ModuleList(
            [
                BasicBlock(
                    input_dim,
                    output_dim,
                    padding=rate,
                    dilation=rate,
                    init_type=init_type,
                )
                for rate in atrous_rates
            ]
        )
        self.module_list.insert(
            0,
            BasicBlock(
                input_dim,
                output_dim,
                kernel_size=1,
                padding=0,
                is_bias=False,
                init_type=init_type,
            ),
        )
        self.module_list.append(_ASPPPooling(input_dim, output_dim, init_type))

    def forward(self, input: torch.Tensor):
        return torch.cat([module(input) for module in self.module_list], dim=1)
