from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.base import OutputBlock
from .common import ConvBlock, Encoder, EncoderBlock


class _DecoderBlock(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, is_upsample: bool = True
    ) -> None:
        super().__init__()
        if is_upsample:
            self.upsample = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            )
            self.conv_block = ConvBlock(input_dim, output_dim, input_dim // 2)
        else:
            self.upsample = nn.ConvTranspose2d(
                input_dim, input_dim // 2, kernel_size=2, stride=2
            )
            self.conv_block = ConvBlock(input_dim, output_dim)

    def forward(self, input: torch.Tensor, skip: torch.Tensor):
        x = self.upsample(input)
        # input is CHW
        if skip.shape != x.shape:
            diff_y = skip.size()[2] - x.size()[2]
            diff_x = skip.size()[3] - x.size()[3]
            x = F.pad(
                x,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )
        return self.conv_block(torch.cat((skip, x), dim=1))


class UNet(nn.Module):

    layer_configurations = [64, 128, 256, 512, 1024]

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        init_type: Optional[str] = None,
    ) -> None:
        super().__init__()
        length = len(self.layer_configurations)
        is_upsample = False
        factor = 2 if is_upsample else 1

        self.encoder_blocks = Encoder(
            input_dim, self.layer_configurations[:-1], init_type
        )
        self.bridge = EncoderBlock(
            self.layer_configurations[3], self.layer_configurations[4] // factor
        )
        self.decoder_blocks = nn.ModuleList(
            [
                _DecoderBlock(
                    self.layer_configurations[i],
                    (
                        self.layer_configurations[i - 1]
                        if i == 0
                        else self.layer_configurations[i - 1] // factor
                    ),
                    is_upsample,
                )
                for i in range(length - 1, 0, -1)
            ]
        )
        self.output_block = OutputBlock(
            self.layer_configurations[0], output_dim, init_type=init_type
        )

    def forward(self, input: torch.Tensor):
        x_list = self.encoder_blocks(input)
        x = self.bridge(x_list[-1])
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, x_list.pop())
        return self.output_block(x)
