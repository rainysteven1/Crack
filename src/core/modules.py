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
