import torch
import torch.nn as nn
from .common import ConvBlock, EncoderBlock
from ..modules import Conv2dSame, InitModule


class DecoderBlock(InitModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        is_upsample: bool = False,
        N_concat: int = 2,
        init_type=None,
    ) -> None:
        """
        Args:
            N_concat: 总合并模块数
        """
        super().__init__(init_type)
        self.up = (
            nn.Upsample(scale_factor=2, mode="bilinear")
            if is_upsample
            else nn.ConvTranspose2d(
                input_dim, output_dim, kernel_size=4, stride=2, padding=1
            )
        )
        self.conv = ConvBlock(
            input_dim // 2 + (N_concat - 1) * output_dim, output_dim, is_batchNorm=False
        )
        if self.init_type:
            self._initialize_weights()

    def forward(self, input, *skips):
        x1 = self.up(input)
        x = torch.cat([x1, *skips], dim=1)
        return self.conv(x)

    def _initialize_weights(self):
        self.init(self.up)


class UNet2Plus(InitModule):
    def __init__(
        self,
        input_dim,
        output_dim,
        filters=[64, 128, 256, 512, 1024],
        init_type="kaiming",
    ) -> None:
        super().__init__(init_type)

        # Encoder
        self.input_layer = ConvBlock(input_dim, filters[0], init_type=init_type)
        self.e1 = EncoderBlock(filters[0], filters[1], init_type=init_type)
        self.e2 = EncoderBlock(filters[1], filters[2], init_type=init_type)
        self.e3 = EncoderBlock(filters[2], filters[3], init_type=init_type)
        self.e4 = EncoderBlock(filters[3], filters[4], init_type=init_type)

        # Decoder
        self.d01 = DecoderBlock(filters[1], filters[0])
        self.d11 = DecoderBlock(filters[2], filters[1])
        self.d21 = DecoderBlock(filters[3], filters[2])
        self.d31 = DecoderBlock(filters[4], filters[3])

        self.d02 = DecoderBlock(filters[1], filters[0], N_concat=3)
        self.d12 = DecoderBlock(filters[2], filters[1], N_concat=3)
        self.d22 = DecoderBlock(filters[3], filters[2], N_concat=3)

        self.d03 = DecoderBlock(filters[1], filters[0], N_concat=4)
        self.d13 = DecoderBlock(filters[2], filters[1], N_concat=4)

        self.d04 = DecoderBlock(filters[1], filters[0], N_concat=5)

        # Output
        self.final_1 = Conv2dSame(filters[0], output_dim, kernel_size=1, padding="same")
        self.final_2 = Conv2dSame(filters[0], output_dim, kernel_size=1, padding="same")
        self.final_3 = Conv2dSame(filters[0], output_dim, kernel_size=1, padding="same")
        self.final_4 = Conv2dSame(filters[0], output_dim, kernel_size=1, padding="same")

        if self.init_type:
            self._initialize_weights()

    def forward(self, input):
        # Encoder
        x00 = self.input_layer(input)
        x10 = self.e1(x00)
        x20 = self.e2(x10)
        x30 = self.e3(x20)
        x40 = self.e4(x30)

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

        x1 = self.final_1(x01)
        x2 = self.final_2(x02)
        x3 = self.final_3(x03)
        x4 = self.final_4(x04)
        x5 = (x1 + x2 + x3 + x4) / 4

        output = torch.sigmoid(x5)
        return output

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, Conv2dSame):
                self.init(module)
