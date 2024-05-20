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
        filters: list = [64, 128, 256, 512, 1024],
        is_upsample: bool = True,
        init_type=None,
    ) -> None:
        super().__init__()

        # Encoder
        self.e = Encoder(input_dim, filters, init_type)

        # Decoder
        self.d1 = _DecoderBlock(
            filters[4], filters[3], is_upsample=is_upsample, init_type=init_type
        )
        self.d2 = _DecoderBlock(
            filters[3], filters[2], is_upsample=is_upsample, init_type=init_type
        )
        self.d3 = _DecoderBlock(
            filters[2], filters[1], is_upsample=is_upsample, init_type=init_type
        )
        self.d4 = _DecoderBlock(
            filters[1], filters[0], is_upsample=is_upsample, init_type=init_type
        )

        # Output
        self.output_layer = OutputBlock(filters[0], output_dim, init_type=init_type)

    def forward(self, input):
        # Encoder
        x1, x2, x3, x4, x5 = self.e(input)

        # Decoder
        x6 = self.d1(x5, x4)
        x7 = self.d2(x6, x3)
        x8 = self.d3(x7, x2)
        x9 = self.d4(x8, x1)

        # Output
        output = self.output_layer(x9)
        return output
