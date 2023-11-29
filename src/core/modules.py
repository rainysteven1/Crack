import math, sys
import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t
import torch.nn.functional as F


class Conv2dSame(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t | str = 0,
        dilation: _size_2_t = 1,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            0 if type(padding) == str else padding,
            dilation,
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
        stride: int,
        padding: str | int,
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


class InputBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernal_size: int = 3,
        skip_kernal_size: int = 1,
        stride: int = 1,
        padding: str | int = "same",
        skip_bn: bool = True,
    ) -> None:
        super().__init__()
        self.skip_bn = skip_bn
        self.conv_block = nn.Sequential(
            Conv2dSame(input_dim, output_dim, kernal_size, stride, padding),
            BasicBlock(output_dim, output_dim, kernal_size, stride, padding),
        )
        self.layers = [
            Conv2dSame(
                input_dim,
                output_dim,
                skip_kernal_size,
                stride,
                padding,
            )
        ]
        if skip_bn:
            self.layers.append(nn.BatchNorm2d(output_dim))
        self.conv_skip = nn.Sequential(*self.layers)

    def forward(self, input):
        return torch.add(self.conv_block(input), self.conv_skip(input))


class RedisualBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int = 3,
        skip_kernel_size: int = 1,
        stride: int = 1,
        padding: str | int = "same",
        is_bridge: bool = False,
    ) -> None:
        self.is_bridge = is_bridge
        super().__init__()
        self.conv_block = nn.Sequential(
            BasicBlock(input_dim, output_dim, kernel_size, stride, padding),
            BasicBlock(output_dim, output_dim, kernel_size, 1, padding),
        )
        self.conv_skip = nn.Sequential(
            Conv2dSame(input_dim, output_dim, skip_kernel_size, stride, padding),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):
        return (
            self.conv_block(x)
            if self.is_bridge
            else torch.add(self.conv_block(x), self.conv_skip(x))
        )


class DecoderBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        skip_dim: int,
        kernel_size: int = 3,
        skip_kernel_size: int = 1,
        stride: int = 1,
        r_stride: int = 1,
        padding: str | int = "same",
        is_upsample: bool = True,
    ) -> None:
        super().__init__()
        self.upsample = (
            nn.Upsample(scale_factor=2, mode="nearest")
            if is_upsample
            else nn.ConvTranspose2d(
                input_dim, input_dim, kernel_size=kernel_size, stride=stride
            )
        )
        self.redisual = RedisualBlock(
            input_dim + skip_dim,
            output_dim,
            kernel_size,
            skip_kernel_size,
            r_stride,
            padding,
        )

    def forward(self, input, skip):
        x0 = self.upsample(input)
        x1 = torch.cat((x0, skip), dim=1)
        x2 = self.redisual(x1)
        return x2


class DiceLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        y_pred_f = torch.flatten(y_pred)
        y_true_f = torch.flatten(y_true)
        intersection = torch.sum(y_pred_f * y_true_f)
        dice_coef = (2.0 * intersection + self.smooth) / (
            torch.sum(y_pred_f) + torch.sum(y_true_f) + self.smooth
        )
        return 1.0 - dice_coef
