from typing import Optional

import torch
import torch.nn as nn

from ..modules.attention import AttentionBlock
from ..modules.base import BasicBlock, InitModule, OutputBlock
from .common import ConvBlock, Encoder


class _DecoderBlock(InitModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        skip_dim: int | None = None,
        is_upsample: bool = True,
        init_type: str | None = None,
    ) -> None:
        super().__init__()
        if not skip_dim:
            skip_dim = output_dim

        self.up_conv = nn.Sequential(
            (
                nn.Upsample(scale_factor=2)
                if is_upsample
                else nn.ConvTranspose2d(
                    input_dim, output_dim, kernel_size=4, stride=2, padding=1
                )
            ),
            BasicBlock(input_dim, output_dim, init_type=init_type),
        )
        self.attention_block = AttentionBlock(
            F_g=output_dim, F_l=skip_dim, F_int=output_dim // 2
        )
        self.conv_block = ConvBlock(
            skip_dim + output_dim, output_dim, init_type=init_type
        )

        if self.init_type:
            self._initialize_weights()

    def forward(self, input, skip):
        x1 = self.up_conv(input)
        x2 = self.attention_block(x1, skip)
        x3 = torch.cat((x2, x1), dim=1)
        output = self.conv_block(x3)
        return output

    def _initialize_weights(self):
        self.init(self.up_conv.children[0])


class AttentionUNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        filters: list,
        is_upsample: bool,
        init_type: Optional[str] = None,
    ) -> None:
        super().__init__()
        length = len(filters)

        self.encoder_blocks = Encoder(input_dim, filters, init_type)
        self.decoder_blocks = nn.ModuleList(
            [
                _DecoderBlock(
                    filters[i],
                    filters[i - 1],
                    is_upsample=is_upsample,
                    init_type=init_type,
                )
                for i in range(length - 1, 0, -1)
            ]
        )
        self.output_block = OutputBlock(filters[0], output_dim, init_type=init_type)

    def forward(self, input: torch.Tensor):
        x_list = self.encoder_blocks(input)
        x = x_list.pop()
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, x_list.pop())
        return self.output_block(x)
