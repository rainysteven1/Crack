import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights
from ..modules import Conv2dSame, ASPP_v3, SqueezeExciteBlock, OutputBlock


class ConvBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            Conv2dSame(input_dim, output_dim, kernel_size=3, padding="same"),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            Conv2dSame(output_dim, output_dim, kernel_size=3, padding="same"),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            SqueezeExciteBlock(output_dim),
        )

    def forward(self, input):
        return self.layers(input)


class DecoderBlock1(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, skip_dim: int) -> None:
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv_block = ConvBlock(input_dim + skip_dim, output_dim)

    def forward(self, input, skip):
        x1 = self.upsample(input)
        x2 = torch.cat((x1, skip), dim=1)
        output = self.conv_block(x2)
        return output


class DecoderBlock2(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, skip1_dim: int, skip2_dim: int
    ) -> None:
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv_block = ConvBlock(input_dim + skip1_dim + skip2_dim, output_dim)

    def forward(self, input, skip1, skip2):
        x1 = self.upsample(input)
        x2 = torch.cat((x1, skip1, skip2), dim=1)
        output = self.conv_block(x2)
        return output


class Encoder1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        network = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)

        self.encoder_block1 = network.features[:4]
        self.encoder_block2 = network.features[4:9]
        self.encoder_block3 = network.features[9:18]
        self.encoder_block4 = network.features[18:27]
        self.encoder_block5 = network.features[27:36]

    def forward(self, input):
        # 64,128,256,512,512
        x1 = self.encoder_block1(input)
        x2 = self.encoder_block2(x1)
        x3 = self.encoder_block3(x2)
        x4 = self.encoder_block4(x3)
        x5 = self.encoder_block5(x4)
        return x5, x4, x3, x2, x1


class Decoder1(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, filters: list) -> None:
        super().__init__()
        assert len(filters) == 4

        self.decoder_block1 = DecoderBlock1(input_dim, filters[2], skip_dim=filters[3])
        self.decoder_block2 = DecoderBlock1(filters[2], filters[1], skip_dim=filters[2])
        self.decoder_block3 = DecoderBlock1(filters[1], filters[0], skip_dim=filters[1])
        self.decoder_block4 = DecoderBlock1(filters[0], output_dim, skip_dim=filters[0])

    def forward(self, input, skip_list):
        assert len(skip_list) == 4
        x1 = self.decoder_block1(input, skip_list[0])
        x2 = self.decoder_block2(x1, skip_list[1])
        x3 = self.decoder_block3(x2, skip_list[2])
        output = self.decoder_block4(x3, skip_list[3])
        return output


class Encoder2(nn.Module):
    def __init__(self, input_dim: int, filters: list) -> None:
        super().__init__()
        assert len(filters) == 4
        self.max_pool = nn.MaxPool2d(kernel_size=2)

        self.conv_block1 = ConvBlock(input_dim, filters[0])
        self.conv_block2 = ConvBlock(filters[0], filters[1])
        self.conv_block3 = ConvBlock(filters[1], filters[2])
        self.conv_block4 = ConvBlock(filters[2], filters[3])

    def forward(self, input):
        x1 = self.conv_block1(input)
        p1 = self.max_pool(x1)
        x2 = self.conv_block2(p1)
        p2 = self.max_pool(x2)
        x3 = self.conv_block3(p2)
        p3 = self.max_pool(x3)
        x4 = self.conv_block4(p3)
        p4 = self.max_pool(x4)
        return p4, x4, x3, x2, x1


class Decoder2(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, filters: list, encoder1_filters: list
    ) -> None:
        super().__init__()
        assert len(filters) == 4
        assert len(encoder1_filters) == 4

        self.decoder_block1 = DecoderBlock2(
            input_dim, filters[3], skip1_dim=encoder1_filters[3], skip2_dim=filters[3]
        )
        self.decoder_block2 = DecoderBlock2(
            filters[3], filters[2], skip1_dim=encoder1_filters[2], skip2_dim=filters[2]
        )
        self.decoder_block3 = DecoderBlock2(
            filters[2], filters[1], skip1_dim=encoder1_filters[1], skip2_dim=filters[1]
        )
        self.decoder_block4 = DecoderBlock2(
            filters[1], output_dim, skip1_dim=encoder1_filters[0], skip2_dim=filters[0]
        )

    def forward(self, input, skip_list1, skip_list2):
        x1 = self.decoder_block1(input, skip_list1[0], skip_list2[0])
        x2 = self.decoder_block2(x1, skip_list1[1], skip_list2[1])
        x3 = self.decoder_block3(x2, skip_list1[2], skip_list2[2])
        output = self.decoder_block4(x3, skip_list1[3], skip_list2[3])
        return output


class DoubleUNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        encoder1_filters: list = [64, 128, 256, 512, 512]
        ASPP1_output_dim = 64
        decoder1_output_dim = 32
        encoder2_filters = [32, 64, 128, 256]
        ASPP2_output_dim = 64
        decoder2_output_dim = 32

        # Network1
        self.e1 = Encoder1()
        self.ASPP1 = ASPP_v3(encoder1_filters[-1], ASPP1_output_dim)
        self.d1 = Decoder1(
            ASPP1_output_dim, decoder1_output_dim, filters=encoder1_filters[:-1]
        )
        self.output_layer1 = OutputBlock(decoder1_output_dim, output_dim)

        # Network2
        self.e2 = Encoder2(input_dim, filters=encoder2_filters)
        self.ASPP2 = ASPP_v3(encoder2_filters[-1], ASPP2_output_dim)
        self.d2 = Decoder2(
            ASPP2_output_dim,
            decoder2_output_dim,
            filters=encoder2_filters,
            encoder1_filters=encoder1_filters[:-1],
        )
        self.output_layer2 = OutputBlock(decoder2_output_dim, output_dim)

    def forward(self, input):
        # Network1
        skip_list1 = self.e1(input)
        x1 = self.ASPP1(skip_list1[0])
        x2 = self.d1(x1, skip_list1[1:])
        output1 = self.output_layer1(x2)

        x3 = torch.multiply(input, output1)

        # Network2
        skip_list2 = self.e2(x3)
        x4 = self.ASPP2(skip_list2[0])
        x5 = self.d2(x4, skip_list1[1:], skip_list2[1:])
        output2 = self.output_layer2(x5)

        return output1, output2
