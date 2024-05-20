from typing import Optional, Union

import torch
import torch.nn as nn

from ..modules.base import BasicBlock


class ConvBlock(nn.Sequential):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        middle_dim: Optional[int] = None,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Union[str, int] = 1,
        is_bn: bool = True,
        init_type: Optional[str] = None,
    ) -> None:
        middle_dim = middle_dim or output_dim

        super().__init__(
            BasicBlock(
                input_dim,
                middle_dim,
                kernel_size,
                stride,
                padding,
                is_bn=is_bn,
                init_type=init_type,
            ),
            BasicBlock(
                middle_dim,
                output_dim,
                kernel_size,
                stride,
                padding,
                is_bn=is_bn,
                init_type=init_type,
            ),
        )


class EncoderBlock(nn.Sequential):
    def __init__(
        self, input_dim: int, output_dim: int, init_type: Optional[str] = None
    ) -> None:
        super().__init__(
            nn.MaxPool2d(kernel_size=2),
            ConvBlock(input_dim, output_dim, init_type=init_type),
        )


class Encoder(nn.Module):
    """按照层的顺序返回结果."""

    def __init__(
        self, input_dim: int, filters: list, init_type: Optional[str] = None
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(ConvBlock(input_dim, filters[0], init_type=init_type))
        for idx in range(0, len(filters) - 1):
            self.layers.append(
                EncoderBlock(filters[idx], filters[idx + 1], init_type=init_type)
            )

    def forward(self, input: torch.Tensor):
        x = input
        x_list = list()
        for module in self.layers:
            x = module(x)
            x_list.append(x)
        return x_list
