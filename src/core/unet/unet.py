import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import ConvBlock, Encoder, EncoderBlock
from ..modules import OutputBlock


class DecoderBlock(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, is_upsample: bool = True
    ) -> None:
        super().__init__()
        if is_upsample:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = ConvBlock(input_dim, output_dim, input_dim // 2)
        else:
            self.up = nn.ConvTranspose2d(
                input_dim, input_dim // 2, kernel_size=2, stride=2
            )
            self.conv = ConvBlock(input_dim, output_dim)

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
        self,
        input_dim: int,
        output_dim: int,
        filters: list = [64, 128, 256, 512, 1024],
        init_type=None,
    ) -> None:
        super().__init__()
        is_upsample = False
        factor = 2 if is_upsample else 1

        # Encoder
        self.e = Encoder(input_dim, filters[:-1], init_type=init_type)

        # Bridge
        self.b1 = EncoderBlock(filters[3], filters[4] // factor)

        # Decoder
        self.d1 = DecoderBlock(filters[4], filters[3] // factor, is_upsample)
        self.d2 = DecoderBlock(filters[3], filters[2] // factor, is_upsample)
        self.d3 = DecoderBlock(filters[2], filters[1] // factor, is_upsample)
        self.d4 = DecoderBlock(filters[1], filters[0], is_upsample)

        # Output
        self.output_layer = OutputBlock(filters[0], output_dim)

    def forward(self, input):
        # Encoder
        x1, x2, x3, x4 = self.e(input)

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
