import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dSame(nn.Conv2d):
    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        input_dim,
        output_dim,
        kernel_size,
        stride,
        padding,
    ) -> None:
        super().__init__()
        bn = nn.BatchNorm2d(input_dim)
        relu = nn.ReLU()
        conv = nn.Conv2d(
            input_dim,
            output_dim,
            kernel_size,
            stride,
            padding,
        )
        self.layers = nn.Sequential(bn, relu, conv)

    def forward(self, x):
        return self.layers(x)


class InputBlock0(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1),
            BasicBlock(output_dim, output_dim, kernel_size=3, stride=1, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return torch.add(self.conv_block(x), self.conv_skip(x))


class RedisualBlock0(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        stride,
        padding,
    ) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            BasicBlock(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            BasicBlock(output_dim, output_dim, kernel_size=3, stride=1, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):
        return torch.add(self.conv_block(x), self.conv_skip(x))


class DecoderBlock0(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            input_dim, input_dim, kernel_size=kernel_size, stride=stride
        )
        self.redisual = RedisualBlock0(
            input_dim + output_dim, output_dim, stride=1, padding=1
        )

    def forward(self, x, skip):
        x0 = self.upsample(x)
        x1 = torch.cat((x0, skip), dim=1)
        x2 = self.redisual(x1)
        return x2


class BasicBlock1(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        kernel_size,
        stride,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            Conv2dSame(input_dim, output_dim, kernel_size=kernel_size, stride=stride),
        )

    def forward(self, x):
        return self.layers(x)


class InputBlock1(nn.Module):
    def __init__(self, input_dim, output_dim, kernal_size=3, stride=1) -> None:
        super().__init__()

        self.conv_block = nn.Sequential(
            Conv2dSame(input_dim, output_dim, kernel_size=kernal_size, stride=stride),
            BasicBlock1(output_dim, output_dim, kernel_size=kernal_size, stride=stride),
        )

        self.conv_skip = nn.Sequential(
            Conv2dSame(input_dim, output_dim, kernel_size=1, stride=stride),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, input):
        return torch.add(self.conv_block(input), self.conv_skip(input))


class RedisualBlock1(nn.Module):
    def __init__(self, input_dim, output_dim, kernal_size=3, stride=1) -> None:
        super().__init__()

        self.conv_block = nn.Sequential(
            BasicBlock1(input_dim, output_dim, kernel_size=kernal_size, stride=stride),
            BasicBlock1(output_dim, output_dim, kernel_size=kernal_size, stride=1),
        )

        self.conv_skip = nn.Sequential(
            Conv2dSame(input_dim, output_dim, kernel_size=1, stride=stride),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, input):
        return torch.add(self.conv_block(input), self.conv_skip(input))


class DecoderBlock1(nn.Module):
    def __init__(self, input_dim, output_dim, skip_dim) -> None:
        super().__init__()
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.redisual = RedisualBlock1(input_dim + skip_dim, output_dim, stride=1)

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
