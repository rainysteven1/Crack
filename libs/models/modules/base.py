import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_2_t

from ...utils.init import InitModule


class _Conv2dSame(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t | str = 0,
        dilation: _size_2_t = 1,
        bias=True,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            0 if isinstance(padding, str) else padding,
            dilation,
            bias=bias,
        )
        self._padding = padding

    def _calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._padding == "same":
            ih, iw = x.size()[-2:]

            pad_h = self._calc_same_pad(
                i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0]
            )
            pad_w = self._calc_same_pad(
                i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1]
            )

            if pad_h > 0 or pad_w > 0:
                x = F.pad(
                    x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
                )
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class SqueezeExciteBlock(nn.Module):
    def __init__(self, filters: int, radio: int = 8) -> None:
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.layers = nn.Sequential(
            nn.Linear(filters, filters // radio, bias=False),
            nn.ReLU(),
            nn.Linear(filters // radio, filters, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input: torch.Tensor):
        n, c, _, _ = input.shape
        x = self.pool(input).view(n, c)
        x = self.layers(x).view(n, c, 1, 1)
        return input * x


class BasicBlock(InitModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Union[int, str] = 1,
        dilation: int = 1,
        is_bn: bool = True,
        is_relu: bool = True,
        is_bias: bool = True,
        reversed: bool = False,
        init_type: Optional[str] = None,
    ):
        super().__init__(init_type)
        layer_list = [
            _Conv2dSame(
                input_dim, output_dim, kernel_size, stride, padding, dilation, is_bias
            ),
        ]
        if reversed:
            layer_list.insert(0, nn.BatchNorm2d(input_dim))
            layer_list.insert(1, nn.ReLU())
        else:
            if is_relu:
                layer_list.append(nn.ReLU())
            if is_bn:
                layer_list.insert(1, nn.BatchNorm2d(output_dim))
        self.layers = nn.Sequential(*layer_list)

        if self.init:
            self._initialize_weights()

    def forward(self, input: torch.Tensor):
        return self.layers(input)

    def _initialize_weights(self):
        self.layers.apply(lambda m: self.init(m))


class OutputBlock(nn.Sequential):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int = 1,
        is_bn: bool = False,
        init_type: Optional[str] = None,
    ) -> None:
        super().__init__(
            *[
                BasicBlock(
                    input_dim,
                    output_dim,
                    kernel_size,
                    padding="same",
                    is_bn=is_bn,
                    is_relu=False,
                    init_type=init_type,
                ),
                nn.Sigmoid(),
            ],
        )
