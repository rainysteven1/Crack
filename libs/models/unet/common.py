import torch
import torch.nn as nn

from ..modules.base import Conv2dSame, SingleBlock


class ConvBlock(nn.Sequential):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        middle_dim: int = None,
        kernel_size: int = 3,
        stride: int = 1,
        padding: str | int = 1,
        is_batchNorm=True,
        init_type=None,
    ) -> None:
        if not middle_dim:
            middle_dim = output_dim

        super().__init__(
            *[
                SingleBlock(
                    input_dim,
                    middle_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    is_batchNorm=is_batchNorm,
                    init_type=init_type,
                ),
                SingleBlock(
                    middle_dim,
                    output_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    is_batchNorm=is_batchNorm,
                    init_type=init_type,
                ),
            ]
        )


class EncoderBlock(nn.Sequential):
    def __init__(self, input_dim: int, output_dim: int, init_type=None) -> None:
        super().__init__(
            *[
                nn.MaxPool2d(kernel_size=2),
                ConvBlock(input_dim, output_dim, init_type=init_type),
            ]
        )


class Encoder(nn.Module):
    """
    按照层的顺序返回结果
    """

    def __init__(self, input_dim: int, filters: list, init_type=None) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(ConvBlock(input_dim, filters[0], init_type=init_type))
        for idx in range(0, len(filters) - 1):
            self.layers.append(
                EncoderBlock(filters[idx], filters[idx + 1], init_type=init_type)
            )

    def forward(self, input):
        x = input
        x_list = list()
        for module in self.layers:
            x = module(x)
            x_list.append(x)
        return x_list


class AttentionBlock(nn.Module):
    """
    Soft Attention
    """

    def __init__(self, F_g: int, F_l: int, F_int: int) -> None:
        super().__init__()

        self.W_g = nn.Sequential(
            Conv2dSame(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            Conv2dSame(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            Conv2dSame(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU()

    def forward(self, input, skip):
        x1 = self.W_g(input)
        x2 = self.W_x(skip)
        x3 = self.relu(torch.add(x1, x2))
        x4 = self.psi(x3)
        output = torch.multiply(skip, x4)
        return output
