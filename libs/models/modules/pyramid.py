from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BasicBlock


class _ASPPPooling(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            BasicBlock(input_dim, output_dim, kernel_size=1, padding=0, is_bias=False),
        )

    def forward(self, input: torch.Tensor):
        return F.interpolate(self.layers(input), size=input.shape[-2:], mode="bilinear")


class ASPP_v2(nn.Module):
    """
    Atrous spatial pyramid pooling (ASPP)
    version: DeepLabv2
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
    """
    version: DeepLabv3
    """

    dilations = [1, 6, 12, 18]

    def __init__(self, input_dim: int, output_dim: int, rate_scale: int = 1) -> None:
        super().__init__()

        self.module_list = nn.ModuleList(
            [
                BasicBlock(
                    input_dim,
                    output_dim,
                    kernel_size=3,
                    padding="same",
                    dilation=dilation * rate_scale,
                )
                for dilation in self.dilations
            ]
        )
        self.module_list.insert(
            0,
            BasicBlock(input_dim, output_dim, kernel_size=1, padding=0, is_bias=False),
        )
        self.module_list.append(_ASPPPooling(input_dim, output_dim))

        self.output_block = nn.Sequential(
            BasicBlock(
                len(self.module_list) * output_dim,
                output_dim,
                kernel_size=1,
                padding=0,
                is_bias=False,
            ),
            nn.Dropout(0.5),
        )

    def forward(self, input: torch.Tensor):
        x = torch.cat([module(input) for module in self.module_list], dim=1)
        return self.output_block(x)
