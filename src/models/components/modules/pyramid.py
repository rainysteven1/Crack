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

    dilations = [1, 6, 12, 18]

    def __init__(self, input_dim: int, output_dim: int, rate_scale: int = 1) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            [
                BasicBlock(
                    input_dim,
                    output_dim,
                    kernel_size=3,
                    padding="same",
                    dilation=dilation * rate_scale,
                    is_relu=False,
                )
                for dilation in self.dilations
            ]
        )
        self.output_block = BasicBlock(
            output_dim, output_dim, kernel_size=1, padding=0, is_bn=False, is_relu=False
        )

    def forward(self, input: torch.Tensor):
        return self.output_block(sum([layer(input) for layer in self.layers]))


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
