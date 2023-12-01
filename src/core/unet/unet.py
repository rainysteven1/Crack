import torch
import torch.nn as nn
import torch.nn.functional as F
from ..modules import Conv2dSame, OutputBlock


class BasicBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        middle_dim: int = None,
        kernel_size: int = 3,
        stride: int = 1,
        padding: str | int = 1,
    ) -> None:
        super().__init__()
        if not middle_dim:
            middle_dim = output_dim
        self.layers = nn.Sequential(
            Conv2dSame(
                input_dim,
                middle_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            Conv2dSame(
                middle_dim,
                output_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class EncoderBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            BasicBlock(input_dim, output_dim),
        )

    def forward(self, x):
        return self.layers(x)


class DecoderBlock(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, is_bilinear: bool = True
    ) -> None:
        super().__init__()
        if is_bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = BasicBlock(input_dim, output_dim, input_dim // 2)
        else:
            self.up = nn.ConvTranspose2d(
                input_dim, input_dim // 2, kernel_size=2, stride=2
            )
            self.conv = BasicBlock(input_dim, output_dim)

    def forward(self, input, skip):
        x1 = self.up(input)
        # input is CHW
        if skip.shape != x1.shape:
            diff_y = skip.size()[2] - x1.size()[2]
            diff_x = skip.size()[3] - x1.size()[3]
            x1 = F.pad(
                x1,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )
        x = torch.cat([skip, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, filters: list = [64, 128, 256, 512, 1024]
    ) -> None:
        super().__init__()
        is_bilinear = False
        factor = 2 if is_bilinear else 1

        # Encoder
        self.input_layer = BasicBlock(input_dim, filters[0])
        self.e1 = EncoderBlock(filters[0], filters[1])
        self.e2 = EncoderBlock(filters[1], filters[2])
        self.e3 = EncoderBlock(filters[2], filters[3])

        # Bridge
        self.b1 = EncoderBlock(filters[3], filters[4] // factor)

        # Decoder
        self.d1 = DecoderBlock(filters[4], filters[3] // factor, is_bilinear)
        self.d2 = DecoderBlock(filters[3], filters[2] // factor, is_bilinear)
        self.d3 = DecoderBlock(filters[2], filters[1] // factor, is_bilinear)
        self.d4 = DecoderBlock(filters[1], filters[0], is_bilinear)

        # Output
        self.output_layer = OutputBlock(filters[0], output_dim)

    def forward(self, input):
        # Encoder
        x1 = self.input_layer(input)
        x2 = self.e1(x1)
        x3 = self.e2(x2)
        x4 = self.e3(x3)

        # Bridge
        x5 = self.b1(x4)

        # Decoder
        x6 = self.d1(x5, x4)
        x7 = self.d2(x6, x3)
        x8 = self.d3(x7, x2)
        x9 = self.d4(x8, x1)

        # output
        output = self.output_layer(x9)

        return output
