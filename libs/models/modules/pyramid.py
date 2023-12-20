import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Conv2dSame


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


class _ASPPPooling(nn.Module):
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


class ASPP_v2(nn.Module):
    """
    Atrous spatial pyramid pooling (ASPP) \n
    version: DeepLabv2
    """

    def __init__(
        self, input_dim: int, output_dim: int, rates: list, rate_scale: int = 1
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        for rate in rates:
            self.layers.append(
                nn.Sequential(
                    Conv2dSame(
                        input_dim,
                        output_dim,
                        kernel_size=3,
                        padding="same",
                        dilation=rate * rate_scale,
                    ),
                    nn.BatchNorm2d(output_dim),
                )
            )

        self.output_layer = Conv2dSame(
            output_dim, output_dim, kernel_size=1, padding="same"
        )

    def forward(self, input):
        x1 = sum([layer(input) for layer in self.layers])
        output = self.output_layer(x1)
        return output


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
        self.module_list.append(_ASPPPooling(input_dim, output_dim))

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
