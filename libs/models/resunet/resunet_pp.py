import torch
import torch.nn as nn

from ..modules import (
    Conv2dSame,
    BasicBlock,
    RedisualBlock,
    SqueezeExciteBlock,
    ASPP_v3,
    AttentionBlock,
    OutputBlock,
)


class _InputBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()

        self.conv_block = nn.Sequential(
            Conv2dSame(input_dim, output_dim, kernel_size=3, padding="same"),
            BasicBlock(output_dim, output_dim, kernel_size=3, padding="same"),
        )
        self.skip_block = nn.Sequential(
            Conv2dSame(input_dim, output_dim, kernel_size=1, padding="same"),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, input):
        return torch.add(self.conv_block(input), self.skip_block(input))


class _EncoderBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            SqueezeExciteBlock(input_dim),
            RedisualBlock(input_dim, output_dim, stride=2),
        )

    def forward(self, input):
        return self.layers(input)


class _DecoderBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, skip_dim: int) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.attention = AttentionBlock(input_dim, input_dim, skip_dim)
        self.redisual = RedisualBlock(input_dim + skip_dim, output_dim)

    def forward(self, input, skip):
        x1 = self.attention(input, skip)
        x2 = self.upsample(x1)
        x3 = torch.cat((x2, skip), dim=1)
        output = self.redisual(x3)
        return output


class ResUNet2Plus(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, filters: list = [16, 32, 64, 128, 256]
    ) -> None:
        super().__init__()
        assert len(filters) == 5

        # Encoder
        self.input_layer = _InputBlock(input_dim, filters[0])
        self.e1 = _EncoderBlock(filters[0], filters[1])
        self.e2 = _EncoderBlock(filters[1], filters[2])
        self.e3 = _EncoderBlock(filters[2], filters[3])

        # Bridge
        self.b1 = ASPP_v3(filters[3], filters[4])

        # Decoder
        self.d1 = _DecoderBlock(filters[4], filters[3], skip_dim=filters[2])
        self.d2 = _DecoderBlock(filters[3], filters[2], skip_dim=filters[1])
        self.d3 = _DecoderBlock(filters[2], filters[1], skip_dim=filters[0])

        # Output
        self.output_layer = nn.Sequential(
            ASPP_v3(filters[1], filters[0]),
            OutputBlock(filters[0], output_dim),
        )

    def forward(self, input):
        # Encoder
        x1 = self.input_layer(input)
        x2 = self.e1(x1)
        x3 = self.e2(x2)
        x4 = self.e3(x3)

        # Bridge
        x5 = self.b1(x4)

        # Decoder
        x6 = self.d1(x5, x3)
        x7 = self.d2(x6, x2)
        x8 = self.d3(x7, x1)

        # Output
        output = self.output_layer(x8)

        return output
