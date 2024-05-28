from typing import List, Optional

import torch
import torch.nn as nn
from torchvision.models import VGG19_Weights, vgg19

from ..modules.base import OutputBlock, SqueezeExciteBlock
from ..modules.pyramid import ASPP_v3
from .common import ConvBlock

__all__ = ["DoubleUNet"]


class _ConvBlock1(nn.Module):

    def __init__(
        self, input_dim: int, output_dim: int, init_type: Optional[str]
    ) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            ConvBlock(input_dim, output_dim, init_type=init_type),
            SqueezeExciteBlock(output_dim),
        )

    def forward(self, input: torch.Tensor):
        return self.layers(input)


class _DecoderBlock1(nn.Module):

    def __init__(
        self, input_dim: int, output_dim: int, skip_dim: int, init_type: Optional[str]
    ) -> None:
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv_block = _ConvBlock1(input_dim + skip_dim, output_dim, init_type)

    def forward(self, input: torch.Tensor, skip: torch.Tensor):
        x1 = self.upsample(input)
        x2 = torch.cat((x1, skip), dim=1)
        output = self.conv_block(x2)
        return output


class _DecoderBlock2(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        skip1_dim: int,
        skip2_dim: int,
        init_type: Optional[str],
    ) -> None:
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv_block = _ConvBlock1(
            input_dim + skip1_dim + skip2_dim, output_dim, init_type
        )

    def forward(self, input: torch.Tensor, skip1: torch.Tensor, skip2: torch.Tensor):
        x1 = self.upsample(input)
        x2 = torch.cat((x1, skip1, skip2), dim=1)
        output = self.conv_block(x2)
        return output


class _Encoder1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        network = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)

        self.encoder_block1 = network.features[:4]
        self.encoder_block2 = network.features[4:9]
        self.encoder_block3 = network.features[9:18]
        self.encoder_block4 = network.features[18:27]
        self.encoder_block5 = network.features[27:36]

    def forward(self, input: torch.Tensor):
        x1 = self.encoder_block1(input)
        x2 = self.encoder_block2(x1)
        x3 = self.encoder_block3(x2)
        x4 = self.encoder_block4(x3)
        x5 = self.encoder_block5(x4)
        return x5, x4, x3, x2, x1


class _Decoder1(nn.Module):

    def __init__(
        self, input_dim: int, output_dim: int, filters: list, init_type: Optional[str]
    ) -> None:
        super().__init__()
        assert len(filters) == 4

        self.decoder_block1 = _DecoderBlock1(
            input_dim, filters[2], skip_dim=filters[3], init_type=init_type
        )
        self.decoder_block2 = _DecoderBlock1(
            filters[2], filters[1], skip_dim=filters[2], init_type=init_type
        )
        self.decoder_block3 = _DecoderBlock1(
            filters[1], filters[0], skip_dim=filters[1], init_type=init_type
        )
        self.decoder_block4 = _DecoderBlock1(
            filters[0], output_dim, skip_dim=filters[0], init_type=init_type
        )

    def forward(self, input: torch.Tensor, skip_list: List[torch.Tensor]):
        assert len(skip_list) == 4
        x1 = self.decoder_block1(input, skip_list[0])
        x2 = self.decoder_block2(x1, skip_list[1])
        x3 = self.decoder_block3(x2, skip_list[2])
        output = self.decoder_block4(x3, skip_list[3])
        return output


class _Encoder2(nn.Module):
    def __init__(self, input_dim: int, filters: list, init_type: Optional[str]) -> None:
        super().__init__()
        assert len(filters) == 4
        self.max_pool = nn.MaxPool2d(kernel_size=2)

        self.conv_block1 = _ConvBlock1(input_dim, filters[0], init_type)
        self.conv_block2 = _ConvBlock1(filters[0], filters[1], init_type)
        self.conv_block3 = _ConvBlock1(filters[1], filters[2], init_type)
        self.conv_block4 = _ConvBlock1(filters[2], filters[3], init_type)

    def forward(self, input: torch.Tensor):
        x1 = self.conv_block1(input)
        p1 = self.max_pool(x1)
        x2 = self.conv_block2(p1)
        p2 = self.max_pool(x2)
        x3 = self.conv_block3(p2)
        p3 = self.max_pool(x3)
        x4 = self.conv_block4(p3)
        p4 = self.max_pool(x4)
        return p4, x4, x3, x2, x1


class _Decoder2(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        filters: list,
        encoder1_filters: list,
        init_type: Optional[str],
    ) -> None:
        super().__init__()
        assert len(filters) == 4
        assert len(encoder1_filters) == 4

        self.decoder_block1 = _DecoderBlock2(
            input_dim,
            filters[3],
            skip1_dim=encoder1_filters[3],
            skip2_dim=filters[3],
            init_type=init_type,
        )
        self.decoder_block2 = _DecoderBlock2(
            filters[3],
            filters[2],
            skip1_dim=encoder1_filters[2],
            skip2_dim=filters[2],
            init_type=init_type,
        )
        self.decoder_block3 = _DecoderBlock2(
            filters[2],
            filters[1],
            skip1_dim=encoder1_filters[1],
            skip2_dim=filters[1],
            init_type=init_type,
        )
        self.decoder_block4 = _DecoderBlock2(
            filters[1],
            output_dim,
            skip1_dim=encoder1_filters[0],
            skip2_dim=filters[0],
            init_type=init_type,
        )

    def forward(
        self, input: torch.Tensor, skip_list1: torch.Tensor, skip_list2: torch.Tensor
    ):
        x1 = self.decoder_block1(input, skip_list1[0], skip_list2[0])
        x2 = self.decoder_block2(x1, skip_list1[1], skip_list2[1])
        x3 = self.decoder_block3(x2, skip_list1[2], skip_list2[2])
        output = self.decoder_block4(x3, skip_list1[3], skip_list2[3])
        return output


class DoubleUNet(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, init_type: Optional[str]
    ) -> None:
        super().__init__()
        encoder1_filters = [64, 128, 256, 512, 512]
        aspp1_output_dim = 64
        decoder1_output_dim = 32
        encoder2_filters = [32, 64, 128, 256]
        aspp2_output_dim = 64
        decoder2_output_dim = 32
        atrous_rates = [1, 6, 12, 18]

        # Network1
        self.e1 = _Encoder1()
        self.aspp1 = ASPP_v3(
            encoder1_filters[-1], aspp1_output_dim, atrous_rates, init_type
        )
        self.d1 = _Decoder1(
            aspp1_output_dim, decoder1_output_dim, encoder1_filters[:-1], init_type
        )
        self.output_layer1 = OutputBlock(
            decoder1_output_dim, output_dim, init_type=init_type
        )

        # Network2
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.e2 = _Encoder2(input_dim, encoder2_filters, init_type)
        self.aspp2 = ASPP_v3(
            encoder2_filters[-1], aspp2_output_dim, atrous_rates, init_type
        )
        self.d2 = _Decoder2(
            aspp2_output_dim,
            decoder2_output_dim,
            filters=encoder2_filters,
            encoder1_filters=encoder1_filters[:-1],
            init_type=init_type,
        )
        self.output_layer2 = OutputBlock(
            decoder2_output_dim, output_dim, init_type=init_type
        )

        self.combine_output_layer = OutputBlock(
            2 * output_dim, output_dim, init_type=init_type
        )

    def forward(self, input):
        # Network1
        skip_list1 = self.e1(input)
        x1 = self.aspp1(skip_list1[0])
        x2 = self.d1(x1, skip_list1[1:])
        output1 = self.output_layer1(x2)

        x3 = torch.multiply(input, output1)

        # Network2
        skip_list2 = self.e2(x3)
        x4 = self.aspp2(skip_list2[0])
        x5 = self.d2(x4, skip_list1[1:], skip_list2[1:])
        output2 = self.output_layer2(x5)

        combine_output = torch.cat((output1, output2), dim=1)
        output = self.combine_output_layer(combine_output)

        return output
