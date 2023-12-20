import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
from torch.nn.common_types import _size_2_t

from ...utils.init import InitModule


class Conv2dSame(nn.Conv2d):
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
            0 if type(padding) == str else padding,
            dilation,
            bias=bias,
        )
        self._padding = padding

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if type(self._padding) == str:
            ih, iw = x.size()[-2:]

            pad_h = self.calc_same_pad(
                i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0]
            )
            pad_w = self.calc_same_pad(
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


class BasicBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int,
        stride: int = 1,
        padding: str | int = "same",
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            Conv2dSame(
                input_dim,
                output_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
        )

    def forward(self, x):
        return self.layers(x)


class RedisualBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int = 3,
        skip_kernel_size: int = 1,
        stride: int = 1,
        padding: str | int = "same",
    ) -> None:
        super().__init__()

        self.conv_block = nn.Sequential(
            BasicBlock(input_dim, output_dim, kernel_size, stride, padding),
            BasicBlock(output_dim, output_dim, kernel_size, 1, padding),
        )
        self.skip_block = nn.Sequential(
            Conv2dSame(input_dim, output_dim, skip_kernel_size, stride, padding),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):
        return torch.add(self.conv_block(x), self.skip_block(x))


class SqueezeExciteBlock(nn.Module):
    def __init__(self, filters: int, radio: int = 8) -> None:
        super().__init__()

        self.average_pooling = nn.AdaptiveAvgPool2d(1)
        self.layers = nn.Sequential(
            nn.Linear(filters, filters // radio, bias=False),
            nn.ReLU(),
            nn.Linear(filters // radio, filters, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input: torch.Tensor):
        n, c, _, _ = input.shape
        x0 = self.average_pooling(input).view(n, c)
        x1 = self.layers(x0).view(n, c, 1, 1)
        output = torch.mul(input, x1)
        return output


class OutputBlock(InitModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int = 1,
        is_bn: bool = False,
        init_type: Optional[str] = None,
    ) -> None:
        super().__init__(init_type)

        self.layer_list = [
            nn.Sequential(
                Conv2dSame(input_dim, output_dim, kernel_size, padding="same"),
                nn.Sigmoid(),
            )
        ]
        if is_bn:
            self.layer_list.insert(1, nn.BatchNorm2d(output_dim))
        self.layers = nn.Sequential(*self.layer_list)

        if self.init_type:
            self._initialize_weights()

    def forward(self, input):
        return self.layers(input)

    def _initialize_weights(self):
        self.layers.apply(lambda m: self.init(m))
