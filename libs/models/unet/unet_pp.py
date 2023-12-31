import torch
import torch.nn as nn

from .common import ConvBlock, Encoder
from ..modules.base import OutputBlock, InitModule


class _DecoderBlock(InitModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        is_upsample: bool = True,
        N_concat: int = 2,
        init_type=None,
    ) -> None:
        """
        Args:
            N_concat: 总合并模块数
        """
        super().__init__(init_type)

        if is_upsample:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = ConvBlock(
                input_dim + (N_concat - 1) * output_dim,
                output_dim,
                is_batchNorm=False,
                init_type=init_type,
            )
        else:
            self.up = nn.ConvTranspose2d(
                input_dim, output_dim, kernel_size=4, stride=2, padding=1
            )
            self.conv = ConvBlock(
                input_dim // 2 + (N_concat - 1) * output_dim,
                output_dim,
                is_batchNorm=False,
                init_type=init_type,
            )

        if self.init_type:
            self._initialize_weights()

    def forward(self, input, *skips):
        x1 = self.up(input)
        x = torch.cat([x1, *skips], dim=1)
        return self.conv(x)

    def _initialize_weights(self):
        self.init(self.up)


class UNet2Plus(nn.Module):
    """
    UNet++
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        filters: list = [64, 128, 256, 512, 1024],
        is_ds: bool = True,
        init_type: str | None = "kaiming",
    ) -> None:
        super().__init__()
        assert len(filters) == 5
        self.is_ds = is_ds

        # Encoder
        self.e = Encoder(input_dim, filters, init_type)

        # Decoder
        self.d01 = _DecoderBlock(filters[1], filters[0], init_type=init_type)
        self.d11 = _DecoderBlock(filters[2], filters[1], init_type=init_type)
        self.d21 = _DecoderBlock(filters[3], filters[2], init_type=init_type)
        self.d31 = _DecoderBlock(filters[4], filters[3], init_type=init_type)

        self.d02 = _DecoderBlock(
            filters[1], filters[0], N_concat=3, init_type=init_type
        )
        self.d12 = _DecoderBlock(
            filters[2], filters[1], N_concat=3, init_type=init_type
        )
        self.d22 = _DecoderBlock(
            filters[3], filters[2], N_concat=3, init_type=init_type
        )

        self.d03 = _DecoderBlock(
            filters[1], filters[0], N_concat=4, init_type=init_type
        )
        self.d13 = _DecoderBlock(
            filters[2], filters[1], N_concat=4, init_type=init_type
        )

        self.d04 = _DecoderBlock(
            filters[1], filters[0], N_concat=5, init_type=init_type
        )

        def final_block():
            return OutputBlock(filters[0], output_dim, init_type=init_type)

        # Output
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
        x11 = self.d11(x20, x10)
        x21 = self.d21(x30, x20)
        x31 = self.d31(x40, x30)

        x02 = self.d02(x11, x00, x01)
        x12 = self.d12(x21, x10, x11)
        x22 = self.d22(x31, x20, x21)

        x03 = self.d03(x12, x00, x01, x02)
        x13 = self.d13(x22, x10, x11, x12)

        x04 = self.d04(x13, x00, x01, x02, x03)

        if self.is_ds:
            x1 = self.final1(x01)
            x2 = self.final2(x02)
            x3 = self.final3(x03)
            x4 = self.final4(x04)
            output = [x1, x2, x3, x4]
        else:
            output = self.final(x04)
        return output
