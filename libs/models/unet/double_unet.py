from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torchvision.models import VGG19_Weights, vgg19

from ..modules.base import BasicBlock, OutputBlock, SqueezeExciteBlock
from ..modules.pyramid import ASPP_v3
from ._base import Decoder, DoubleConv, Encoder

__all__ = ["DoubleUNet"]


class _ConvBlock(nn.Sequential):

    def __init__(
        self, input_dim: int, output_dim: int, init_type: Optional[str] = None
    ) -> None:
        super().__init__(
            DoubleConv(input_dim, output_dim, init_type=init_type),
            SqueezeExciteBlock(output_dim),
        )


class _DecoderBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        index: int,
        skip_dims_list: List[List[int]],
        init_type: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_block = _ConvBlock(
            input_dim + sum(skip_dims_list[index - 1]), output_dim, init_type
        )

    def forward(self, input: torch.Tensor, *skips: torch.Tensor) -> torch.Tensor:
        return self.conv_block(torch.cat((self.upsample(input), *skips), dim=1))


class _EncoderBlock(nn.Sequential):
    def __init__(
        self, input_dim: int, output_dim: int, _: int, init_type: Optional[str] = None
    ) -> None:
        super().__init__(
            nn.MaxPool2d(kernel_size=2),
            _ConvBlock(input_dim, output_dim, init_type),
        )


class _Encoder1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        network = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)

        self.encoder_block1 = network.features[:4]
        self.encoder_block2 = network.features[4:9]
        self.encoder_block3 = network.features[9:18]
        self.encoder_block4 = network.features[18:27]
        self.encoder_block5 = network.features[27:36]

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor]:
        x1 = self.encoder_block1(input)
        x2 = self.encoder_block2(x1)
        x3 = self.encoder_block3(x2)
        x4 = self.encoder_block4(x3)
        x5 = self.encoder_block5(x4)
        return [x1, x2, x3, x4, x5]


def _get_decoder_config(
    decoder_output_dim: int,
    aspp_output_dim: int,
    *encoder_dims_tuple: List[int],
) -> Dict:
    config = dict()
    config["dims"] = (
        [decoder_output_dim] + encoder_dims_tuple[-1][:-1] + [aspp_output_dim]
    )
    config["skip_dims_list"] = list()
    for i in range(len(encoder_dims_tuple[0])):
        skip_dims = [encoder_dims[i] for encoder_dims in encoder_dims_tuple]
        config["skip_dims_list"].append(skip_dims)
    return config


class DoubleUNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        config: DictConfig,
        init_type: Optional[str] = None,
    ) -> None:
        super().__init__()
        encoder1_dims = config.get("encoder1_dims")
        encoder2_dims = config.get("encoder2_dims")
        decoder_output_dim = config.get("decoder_output_dim")
        aspp_output_dim = config.get("aspp_output_dim")
        atrous_rates = config.get("atrous_rates")

        decoder1 = _get_decoder_config(
            decoder_output_dim, aspp_output_dim, *[encoder1_dims]
        )
        decoder2 = _get_decoder_config(
            decoder_output_dim, aspp_output_dim, *[encoder1_dims, encoder2_dims]
        )

        # Network1
        self.e1 = _Encoder1()
        self.aspp1 = ASPP_v3(
            encoder1_dims[-1], aspp_output_dim, atrous_rates, init_type
        )
        self.d1 = Decoder(decoder_block=_DecoderBlock, init_type=init_type, **decoder1)
        self.output_layer1 = OutputBlock(
            decoder_output_dim, output_dim, init_type=init_type
        )

        # Network2
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.e2 = Encoder(
            input_dim, encoder2_dims, _ConvBlock, _EncoderBlock, init_type
        )
        self.aspp2 = ASPP_v3(
            encoder2_dims[-1], aspp_output_dim, atrous_rates, init_type
        )
        self.d2 = Decoder(decoder_block=_DecoderBlock, init_type=init_type, **decoder2)
        self.output_layer2 = OutputBlock(
            decoder_output_dim, output_dim, init_type=init_type
        )

        self.combine_output_layer = OutputBlock(
            2 * output_dim, output_dim, init_type=init_type
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Network1
        x_list1 = self.e1(input)
        x = self.aspp1(x_list1[-1])
        x_list1 = x_list1[:-1]
        x_list1_copy = x_list1.copy()
        x = self.d1(x, x_list1)
        output1 = self.output_layer1(x)
        x = input * output1

        # Network2
        x_list2 = self.e2(x)
        x = self.max_pool(self.aspp2(x_list2[-1]))
        x = self.d2(x, *[x_list1_copy, x_list2])
        output2 = self.output_layer2(x)

        return self.combine_output_layer(torch.cat((output1, output2), dim=1))
