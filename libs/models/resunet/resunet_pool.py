import torch
import torch.nn as nn

from ..modules.base import Conv2dSame, BasicBlock, OutputBlock


class _InputBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            BasicBlock(input_dim, output_dim, kernel_size=1, stride=1, padding="same"),
            BasicBlock(output_dim, output_dim, kernel_size=3, stride=1, padding="same"),
            BasicBlock(output_dim, output_dim, kernel_size=1, stride=1, padding="same"),
        )
        self.skip_block = nn.Sequential(
            Conv2dSame(input_dim, output_dim, kernel_size=1, stride=1, padding="same"),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, input):
        return torch.add(self.conv_block(input), self.skip_block(input))


class _RedisualBlock(_InputBlock):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        is_pool: bool = True,
    ) -> None:
        super().__init__(input_dim, output_dim)
        self.is_pool = is_pool
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, input):
        x = super().forward(input)
        output = x if not self.is_pool else self.max_pool(x)
        return output


class _DecoderBlock(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, skip_dim: int, is_upsample=True
    ) -> None:
        super().__init__()
        self.upsample = (
            nn.Upsample(scale_factor=2)
            if is_upsample
            else nn.ConvTranspose2d(input_dim, input_dim, kernel_size=2, stride=2)
        )
        self.redisual = _RedisualBlock(input_dim + skip_dim, output_dim, is_pool=False)

    def forward(self, input, skip):
        x0 = self.upsample(input)
        x1 = torch.cat((x0, skip), dim=1)
        output = self.redisual(x1)
        return output


class ResUNetPool(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, filters: list = [16, 32, 64, 96, 128]
    ) -> None:
        super().__init__()
        assert len(filters) == 5
        is_upsample = False

        # Encoder
        self.input_layer = _InputBlock(input_dim, filters[0])  # c1
        self.e1 = _RedisualBlock(filters[0], filters[1])  # c2
        self.e2 = _RedisualBlock(filters[1], filters[2])  # c3
        self.e3 = _RedisualBlock(filters[2], filters[3])  # c4

        # Bridge_
        self.b1 = _RedisualBlock(filters[3], filters[4])
        self.b2 = _RedisualBlock(filters[4], filters[4], is_pool=False)

        # Decoder_
        self.d1 = _DecoderBlock(filters[4], filters[3], filters[3], is_upsample=False)
        self.d2 = _DecoderBlock(filters[3], filters[2], filters[2], is_upsample=False)
        self.d3 = _DecoderBlock(filters[2], filters[1], filters[1], is_upsample=False)
        self.d4 = _DecoderBlock(filters[1], filters[0], filters[0], is_upsample=False)

        # Output
        self.ouput_layer = OutputBlock(filters[0], output_dim, is_bn=True)

    def forward(self, input):
        # Encoder
        x1 = self.input_layer(input)
        x2 = self.e1(x1)
        x3 = self.e2(x2)
        x4 = self.e3(x3)

        # Bridge
        x5 = self.b1(x4)
        x6 = self.b2(x5)

        # Decoder
        x7 = self.d1(x6, x4)
        x8 = self.d2(x7, x3)
        x9 = self.d3(x8, x2)
        x10 = self.d4(x9, x1)

        # Output
        output = self.ouput_layer(x10)

        return output
