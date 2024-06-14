from typing import List, Optional

import torch
import torch.nn as nn

from ..modules.attention import AttentionBlock
from ..modules.base import BasicBlock, InitModule, OutputBlock
from ._base import Decoder, DoubleConv, Encoder


class DecoderBlock(InitModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        _: int,
        is_upsample: bool = True,
        init_type: Optional[str] = None,
    ) -> None:
        super().__init__()
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
            F_g=output_dim, F_l=skip_dim, F_int=output_dim // 2, init_type=init_type
        )
        self.conv_block = DoubleConv(
            skip_dim + output_dim, output_dim, init_type=init_type
        )

    def forward(self, input: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x1 = self.up_conv(input)
        x2 = self.attention_block(x1, skip)
        x3 = torch.cat((x2, x1), dim=1)
        return self.conv_block(x3)

    def _initialize_weights(self) -> None:
        self.init(self.up_conv.children[0])


class AttentionUNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dims: List[int],
        is_upsample: bool = True,
        init_type: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.encoder = Encoder(input_dim, dims, init_type=init_type)
        self.decoder = Decoder(dims, DecoderBlock, init_type, is_upsample=is_upsample)
        self.output_block = OutputBlock(dims[0], output_dim, init_type=init_type)

    def forward(self, input: torch.Tensor):
        x_list = self.encoder(input)
        x = x_list.pop()
        x = self.decoder(x, x_list)
        return self.output_block(x)
