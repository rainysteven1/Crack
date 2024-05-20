import torch
import torch.nn as nn

from ..modules.attention import AttentionBlock
from ..modules.base import InitModule, OutputBlock
from .common import BasicBlock, ConvBlock, Encoder


class _DecoderBlock(InitModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        skip_dim: int | None = None,
        is_upsample: bool = True,
        is_max_pool: bool = False,
        N_concat: int = 2,
        init_type: str | None = None,
    ):
        super().__init__(init_type)
        self.is_max_pool = is_max_pool
        if not skip_dim:
            skip_dim = output_dim

        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.up_conv = nn.Sequential(
            (
                nn.Upsample(scale_factor=2)
                if is_upsample
                else nn.ConvTranspose2d(
                    input_dim, output_dim, kernel_size=4, stride=2, padding=1
                )
            ),
            BasicBlock(input_dim, output_dim, is_bn=False, init_type=init_type),
        )
        self.attention_block = AttentionBlock(
            F_g=output_dim, F_l=skip_dim, F_int=output_dim // 2
        )
        self.conv_block = (
            ConvBlock(
                input_dim // 2 + (N_concat - 1) * output_dim,
                output_dim,
                is_bn=False,
                init_type=init_type,
            )
            if not is_max_pool
            else ConvBlock(
                input_dim // 2 + output_dim // 2 + (N_concat - 2) * output_dim,
                output_dim,
                is_bn=False,
                init_type=init_type,
            )
        )

    def forward(self, input, *skips):
        x1 = self.up_conv(input)
        x2 = self.attention_block(x1, skips[-1])
        concat_list = (
            [*skips[:-1], x2, x1]
            if not self.is_max_pool
            else [self.max_pool(skips[0]), *skips[1:-1], x2, x1]
        )
        x3 = torch.cat(concat_list, dim=1)
        output = self.conv_block(x3)
        return output


class AttentionUNet2Plus(InitModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        filters: list = [64, 128, 256, 512, 1024],
        is_ds: bool = True,
        init_type: str | None = None,
    ) -> None:
        super().__init__(init_type)
        self.is_ds = is_ds

        # Encoder
        self.e = Encoder(input_dim, filters, init_type)

        # Decoder
        self.d01 = _DecoderBlock(filters[1], filters[0], init_type=init_type)
        self.d11 = _DecoderBlock(
            filters[2], filters[1], is_max_pool=True, N_concat=3, init_type=init_type
        )
        self.d21 = _DecoderBlock(
            filters[3], filters[2], is_max_pool=True, N_concat=3, init_type=init_type
        )
        self.d31 = _DecoderBlock(
            filters[4], filters[3], is_max_pool=True, N_concat=3, init_type=init_type
        )

        self.d02 = _DecoderBlock(
            filters[1], filters[0], N_concat=3, init_type=init_type
        )
        self.d12 = _DecoderBlock(
            filters[2], filters[1], is_max_pool=True, N_concat=4, init_type=init_type
        )
        self.d22 = _DecoderBlock(
            filters[3], filters[2], is_max_pool=True, N_concat=4, init_type=init_type
        )

        self.d03 = _DecoderBlock(
            filters[1], filters[0], N_concat=4, init_type=init_type
        )
        self.d13 = _DecoderBlock(
            filters[2], filters[1], is_max_pool=True, N_concat=5, init_type=init_type
        )

        self.d04 = _DecoderBlock(
            filters[1], filters[0], N_concat=5, init_type=init_type
        )

        # Output
        def final_block():
            return OutputBlock(filters[0], output_dim, init_type=init_type)

        if self.is_ds:
            self.final1 = final_block()
            self.final2 = final_block()
            self.final3 = final_block()
            self.final4 = final_block()
        else:
            self.final = final_block()

    def forward(self, input):
        # Encoder
        x00, x10, x20, x30, x40 = self.e(input)

        # Decoder
        x01 = self.d01(x10, x00)
        x11 = self.d11(x20, x01, x10)
        x21 = self.d21(x30, x11, x20)
        x31 = self.d31(x40, x21, x30)

        x02 = self.d02(x11, x00, x01)
        x12 = self.d12(x21, x02, x10, x11)
        x22 = self.d22(x31, x12, x20, x21)

        x03 = self.d03(x12, x00, x01, x02)
        x13 = self.d13(x22, x03, x10, x11, x12)

        x04 = self.d04(x13, x00, x01, x02, x03)

        # Output
        x1 = self.final1(x01)
        x2 = self.final2(x02)
        x3 = self.final3(x03)
        x4 = self.final4(x04)
        output = [x1, x2, x3, x4]
        if not self.is_ds:
            output = torch.mean(output)
        return output
