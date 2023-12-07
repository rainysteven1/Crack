import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from typing import Union
from torch.nn.common_types import _size_2_t


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


class OutputBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, is_bn: bool = False) -> None:
        super().__init__()
        self.layers = (
            nn.Sequential(
                Conv2dSame(input_dim, output_dim, kernel_size=1, padding="same"),
                nn.Sigmoid(),
            )
            if not is_bn
            else nn.Sequential(
                Conv2dSame(input_dim, output_dim, kernel_size=1, padding="same"),
                nn.BatchNorm2d(output_dim),
                nn.Sigmoid(),
            )
        )

    def forward(self, input):
        return self.layers(input)


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


class AttentionBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, skip_dim: int) -> None:
        super().__init__()

        def gen_block(in_c):
            return nn.Sequential(
                nn.BatchNorm2d(in_c),
                nn.ReLU(),
                Conv2dSame(in_c, output_dim, kernel_size=3, padding="same"),
            )

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_block = gen_block(input_dim)
        self.skip_block = gen_block(skip_dim)
        self.output_block = gen_block(output_dim)

    def forward(self, input, skip):
        x1 = self.conv_block(input)

        x2 = self.skip_block(skip)
        x3 = self.max_pool(x2)

        x4 = torch.add(x1, x3)
        x5 = self.output_block(x4)
        output = torch.multiply(x5, input)
        return output


class ASPP(nn.Module):
    """
    version: DeepLabv2
    """

    def __init__(self, input_dim: int, output_dim: int, rate_scale: int = 1) -> None:
        super().__init__()

        def gen_conv_block(dilation: int):
            return nn.Sequential(
                Conv2dSame(
                    input_dim,
                    output_dim,
                    kernel_size=3,
                    padding="same",
                    dilation=dilation,
                ),
                nn.BatchNorm2d(output_dim),
            )

        self.conv_block1 = gen_conv_block(6 * rate_scale)
        self.conv_block2 = gen_conv_block(12 * rate_scale)
        self.conv_block3 = gen_conv_block(18 * rate_scale)
        self.conv_block4 = gen_conv_block(1)
        self.output_layer = Conv2dSame(
            output_dim, output_dim, kernel_size=1, padding="same"
        )

    def forward(self, input):
        x1 = self.conv_block1(input)
        x2 = self.conv_block2(input)
        x3 = self.conv_block3(input)
        x4 = self.conv_block4(input)

        x5 = x1 + x2 + x3 + x4
        output = self.output_layer(x5)
        return output


class ASPPPooling(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv2dSame(input_dim, output_dim, kernel_size=1, padding="same"),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
        )

    def forward(self, input):
        return F.interpolate(
            self.layers(input),
            size=input.shape[-2:],
            mode="bilinear",
        )


class ASPP_v3(nn.Module):
    """
    version: DeepLabv3
    """

    def __init__(self, input_dim: int, output_dim: int, rate_scale: int = 1) -> None:
        super().__init__()
        dilations = [1, 6, 12, 18]

        # ASPP_CONV
        def gen_conv_block(dilation):
            return nn.Sequential(
                Conv2dSame(
                    input_dim,
                    output_dim,
                    kernel_size=3,
                    padding="same",
                    dilation=dilation,
                ),
                nn.BatchNorm2d(output_dim),
                nn.ReLU(),
            )

        self.module_list = nn.ModuleList(
            [gen_conv_block(dilation * rate_scale) for dilation in dilations]
        )

        # ASPP_POOL
        self.module_list.append(ASPPPooling(input_dim, output_dim))

        # Output
        self.output_layer = nn.Sequential(
            Conv2dSame(
                len(self.module_list) * output_dim,
                output_dim,
                kernel_size=1,
                padding="same",
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, input):
        x_list = [module(input) for module in self.module_list]
        x1 = torch.cat(x_list, dim=1)
        output = self.output_layer(x1)
        return output


def init_batchNorm(module):
    init.normal_(module.weight.data, 1.0, 0.02)
    init.constant_(module.bias.data, 0.0)


class InitWeights_He:
    def __init__(self, neg_slope=0):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.BatchNorm2d):
            init_batchNorm(module)
        if isinstance(module, Conv2dSame) or isinstance(module, nn.ConvTranspose2d):
            module.weight = init.kaiming_normal_(
                module.weight, a=self.neg_slope, mode="fan_in"
            )
            if module.bias is not None:
                module.bias = init.constant_(module.bias, 0)


class InitWeights_XavierUniform:
    def __init__(self, gain=1):
        self.gain = gain

    def __call__(self, module):
        if isinstance(module, nn.BatchNorm2d):
            init_batchNorm(module)
        if isinstance(module, Conv2dSame) or isinstance(module, nn.ConvTranspose2d):
            module.weight = init.xavier_uniform_(module.weight, self.gain)
            if module.bias is not None:
                module.bias = init.constant_(module.bias, 0)


class InitModule(nn.Module):
    def __init__(self, init_type: Union[str, None] = None):
        super().__init__()

        self.init_type = init_type

        if self.init_type:
            if self.init_type == "kaiming":
                self.init = InitWeights_He()
            elif self.init_type == "xavier":
                self.init = InitWeights_XavierUniform()

    def _initialize_weights(self):
        pass
