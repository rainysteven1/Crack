import torch
import torch.nn as nn
from torchsummary import summary
from ..modules import Conv2dSame


class BasicBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int,
        stride: int,
        padding: str | int,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            Conv2dSame(
                input_dim,
                output_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
        )

    def forward(self, x):
        return self.layers(x)


class InputBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernal_size: int = 3,
        skip_kernal_size: int = 1,
        stride: int = 1,
        padding: str | int = "same",
        skip_bn: bool = True,
    ) -> None:
        super().__init__()
        self.skip_bn = skip_bn
        self.conv_block = nn.Sequential(
            Conv2dSame(input_dim, output_dim, kernal_size, stride, padding),
            BasicBlock(output_dim, output_dim, kernal_size, stride, padding),
        )
        self.layers = [
            Conv2dSame(
                input_dim,
                output_dim,
                skip_kernal_size,
                stride,
                padding,
            )
        ]
        if skip_bn:
            self.layers.append(nn.BatchNorm2d(output_dim))
        self.skip_block = nn.Sequential(*self.layers)

    def forward(self, input):
        return torch.add(self.conv_block(input), self.skip_block(input))


class RedisualBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int = 3,
        skip_kernel_size: int = 1,
        stride: int = 1,
        padding: str | int = "same",
        is_bridge: bool = False,
    ) -> None:
        self.is_bridge = is_bridge
        super().__init__()
        self.conv_block = nn.Sequential(
            BasicBlock(input_dim, output_dim, kernel_size, stride, padding),
            BasicBlock(output_dim, output_dim, kernel_size, 1, padding),
        )
        self.skip_block = nn.Sequential(
            Conv2dSame(input_dim, output_dim, skip_kernel_size, stride, padding),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):
        return (
            self.conv_block(x)
            if self.is_bridge
            else torch.add(self.conv_block(x), self.skip_block(x))
        )


class DecoderBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        skip_dim: int,
        kernel_size: int = 3,
        skip_kernel_size: int = 1,
        stride: int = 1,
        r_stride: int = 1,
        padding: str | int = "same",
        is_upsample: bool = True,
    ) -> None:
        super().__init__()
        self.upsample = (
            nn.Upsample(scale_factor=2, mode="nearest")
            if is_upsample
            else nn.ConvTranspose2d(
                input_dim, input_dim, kernel_size=kernel_size, stride=stride
            )
        )
        self.redisual = RedisualBlock(
            input_dim + skip_dim,
            output_dim,
            kernel_size,
            skip_kernel_size,
            r_stride,
            padding,
        )

    def forward(self, input, skip):
        x0 = self.upsample(input)
        x1 = torch.cat((x0, skip), dim=1)
        x2 = self.redisual(x1)
        return x2


class ResUNet0(nn.Module):
    """
    原版ResUNet
    """

    def __init__(
        self,
        input_dim: int,
        ouput_dim: int,
        filters: list = [64, 128, 256, 512],
    ) -> None:
        super().__init__()

        # Encoder
        self.input_layer = InputBlock(
            input_dim, filters[0], skip_kernal_size=3, padding=1
        )
        self.e1 = RedisualBlock(
            filters[0], filters[1], skip_kernel_size=3, stride=2, padding=1
        )
        self.e2 = RedisualBlock(
            filters[1], filters[2], skip_kernel_size=3, stride=2, padding=1
        )

        # Bridge
        self.b1 = RedisualBlock(
            filters[2], filters[3], skip_kernel_size=3, stride=2, padding=1
        )

        # Decoder
        self.d1 = DecoderBlock(
            filters[3],
            filters[2],
            filters[2],
            kernel_size=2,
            skip_kernel_size=3,
            stride=2,
            r_stride=1,
            is_upsample=False,
        )
        self.d2 = DecoderBlock(
            filters[2],
            filters[1],
            filters[1],
            kernel_size=2,
            skip_kernel_size=3,
            stride=2,
            r_stride=1,
            is_upsample=False,
        )
        self.d3 = DecoderBlock(
            filters[1],
            filters[0],
            filters[0],
            kernel_size=2,
            skip_kernel_size=3,
            stride=2,
            r_stride=1,
            is_upsample=False,
        )

        # Output
        self.output_layer = nn.Sequential(
            Conv2dSame(filters[0], ouput_dim, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, input):
        # Encoder
        x1 = self.input_layer(input)
        x2 = self.e1(x1)
        x3 = self.e2(x2)

        # Bridge
        x4 = self.b1(x3)

        # Decoder
        x5 = self.d1(x4, x3)
        x6 = self.d2(x5, x2)
        x7 = self.d3(x6, x1)

        # Output
        output = self.output_layer(x7)
        return output


class ResUNet1(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, filters: list = [16, 32, 64, 128, 256]
    ) -> None:
        super().__init__()

        # Encoder
        self.input_layer = InputBlock(input_dim, filters[0])
        self.e1 = RedisualBlock(filters[0], filters[1], stride=2)
        self.e2 = RedisualBlock(filters[1], filters[2], stride=2)
        self.e3 = RedisualBlock(filters[2], filters[3], stride=2)
        self.e4 = RedisualBlock(filters[3], filters[4], stride=2)

        # Bridge
        self.b1 = RedisualBlock(filters[4], filters[4], is_bridge=True)

        # Decoder
        self.d1 = DecoderBlock(filters[4], filters[4], skip_dim=filters[3])
        self.d2 = DecoderBlock(filters[4], filters[3], skip_dim=filters[2])
        self.d3 = DecoderBlock(filters[3], filters[2], skip_dim=filters[1])
        self.d4 = DecoderBlock(filters[2], filters[1], skip_dim=filters[0])

        # Output
        self.output_layer = nn.Sequential(
            Conv2dSame(filters[1], output_dim, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, input):
        # Encoder
        x1 = self.input_layer(input)
        x2 = self.e1(x1)
        x3 = self.e2(x2)
        x4 = self.e3(x3)
        x5 = self.e4(x4)

        # Bridge
        x6 = self.b1(x5)

        # Decoder
        x7 = self.d1(x6, x4)
        x8 = self.d2(x7, x3)
        x9 = self.d3(x8, x2)
        x10 = self.d4(x9, x1)

        # Output
        output = self.output_layer(x10)

        return output


if __name__ == "__main__":
    resunet = ResUNet0(3, 1)
    summary(resunet, (3, 256, 256), 4, "cpu")
