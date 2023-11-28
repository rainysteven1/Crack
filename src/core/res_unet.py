import torch.nn as nn
from torchsummary import summary
from .modules import *


class ResUNet0(nn.Module):
    """
    原版ResUNet
    """

    def __init__(
        self,
        input_dim,
        filters=[64, 128, 256, 512],
    ) -> None:
        super().__init__()

        # Encoder
        self.input_layer = InputBlock0(input_dim, filters[0])
        self.e1 = RedisualBlock0(filters[0], filters[1], stride=2, padding=1)
        self.e2 = RedisualBlock0(filters[1], filters[2], stride=2, padding=1)

        # Bridge
        self.b0 = RedisualBlock0(filters[2], filters[3], stride=2, padding=1)

        # Decoder
        self.d1 = DecoderBlock0(filters[3], filters[2], kernel_size=2, stride=2)
        self.d2 = DecoderBlock0(filters[2], filters[1], kernel_size=2, stride=2)
        self.d3 = DecoderBlock0(filters[1], filters[0], kernel_size=2, stride=2)

        # Output
        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], 1, kernel_size=1, stride=1, padding=0), nn.Sigmoid()
        )

    def forward(self, input):
        # Encoder
        x1 = self.input_layer(input)
        x2 = self.e1(x1)
        x3 = self.e2(x2)

        # Bridge
        x4 = self.b0(x3)

        # Decoder
        x5 = self.d1(x4, x3)
        x6 = self.d2(x5, x2)
        x7 = self.d3(x6, x1)

        # Output
        output = self.output_layer(x7)
        return output


class ResUNet1(nn.Module):
    def __init__(self, input_dim, output_dim, filters=[16, 32, 64, 128, 256]) -> None:
        super().__init__()

        # Encoder
        self.input_layer = InputBlock1(input_dim, filters[0])
        self.e1 = RedisualBlock1(filters[0], filters[1], stride=2)
        self.e2 = RedisualBlock1(filters[1], filters[2], stride=2)
        self.e3 = RedisualBlock1(filters[2], filters[3], stride=2)
        self.e4 = RedisualBlock1(filters[3], filters[4], stride=2)

        # Bridge
        self.b0 = BasicBlock1(filters[4], filters[4], kernel_size=3, stride=1)
        self.b1 = BasicBlock1(filters[4], filters[4], kernel_size=3, stride=1)

        # Decoder
        self.d1 = DecoderBlock1(filters[4], filters[4], skip_dim=filters[3])
        self.d2 = DecoderBlock1(filters[4], filters[3], skip_dim=filters[2])
        self.d3 = DecoderBlock1(filters[3], filters[2], skip_dim=filters[1])
        self.d4 = DecoderBlock1(filters[2], filters[1], skip_dim=filters[0])

        # Output
        self.output_layer = nn.Sequential(
            Conv2dSame(filters[1], output_dim, kernel_size=1, stride=1), nn.Sigmoid()
        )

    def forward(self, input):
        # Encoder
        x1 = self.input_layer(input)
        x2 = self.e1(x1)
        x3 = self.e2(x2)
        x4 = self.e3(x3)
        x5 = self.e4(x4)

        # Bridge
        x6 = self.b0(x5)
        x7 = self.b1(x6)

        # Decoder
        x8 = self.d1(x7, x4)
        x9 = self.d2(x8, x3)
        x10 = self.d3(x9, x2)
        x11 = self.d4(x10, x1)

        # Output
        output = self.output_layer(x11)

        return output


if __name__ == "__main__":
    resunet = ResUNet1(3)
    summary(resunet, (3, 256, 256), 4, "cpu")
