from typing import List, Optional, OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from omegaconf import DictConfig

from ...modules.base import BasicBlock, OutputBlock, SqueezeExciteBlock
from ...modules.dcn import DeformConv2d
from ...transformer.mobilevit import MobileViTBlock
from .._base import Decoder, Encoder

__all__ = ["TransMUNet"]


class _DUC(nn.Sequential):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        ratio: int = 2,
        init_type: Optional[str] = None,
    ) -> None:
        super().__init__(
            BasicBlock(input_dim, output_dim, init_type=init_type),
            nn.PixelShuffle(ratio),
        )


class _DilatedRedisual(nn.Module):

    dilations = [1, 2, 5]

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        middle_dim: Optional[int] = None,
        dilations: Optional[List[int]] = None,
        init_type: Optional[str] = None,
    ) -> None:
        super().__init__()
        middle_dim = middle_dim or output_dim
        dilations = dilations or self.dilations

        self.conv_block = nn.Sequential(
            BasicBlock(
                input_dim,
                middle_dim,
                padding=dilations[0],
                dilation=dilations[0],
                init_type=init_type,
            ),
            BasicBlock(
                middle_dim,
                middle_dim,
                padding=dilations[1],
                dilation=dilations[1],
                init_type=init_type,
            ),
            BasicBlock(
                middle_dim,
                output_dim,
                padding=dilations[2],
                dilation=dilations[2],
                init_type=init_type,
            ),
            SqueezeExciteBlock(output_dim, ratio=16),
        )
        self.skip_block = (
            nn.Identity()
            if input_dim == output_dim
            else BasicBlock(
                input_dim,
                output_dim,
                kernel_size=1,
                padding=0,
                is_bias=False,
                is_relu=False,
                init_type=init_type,
            )
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.conv_block(input) + self.skip_block(input)


class _Boundary(nn.Sequential):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        middle_dim: Optional[int] = None,
        ratio: int = 4,
        init_type: Optional[str] = None,
    ) -> None:
        middle_dim = middle_dim or input_dim

        super().__init__(
            SqueezeExciteBlock(input_dim, ratio=ratio),
            DeformConv2d(input_dim, middle_dim, modulation=True),
            nn.BatchNorm2d(middle_dim),
            nn.ReLU(),
            BasicBlock(
                middle_dim,
                output_dim,
                kernel_size=1,
                padding=0,
                is_bias=False,
                is_bn=False,
                is_relu=False,
                init_type=init_type,
            ),
        )


class _EncoderBlock(nn.Sequential):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        index: int,
        length: int,
        block_configs: List[List[int]],
        init_type: Optional[str] = None,
        **kwargs,
    ) -> None:

        super().__init__(
            nn.MaxPool2d(kernel_size=2, stride=2),
            _DilatedRedisual(input_dim, output_dim, init_type=init_type),
            (
                nn.Identity()
                if index == length - 2
                else MobileViTBlock(
                    output_dim,
                    output_dim,
                    block_config=None if index == length - 2 else block_configs[index],
                    init_type=init_type,
                    **kwargs,
                )
            ),
        )


class _DecoderBlock(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, _: int, init_type: Optional[str] = None
    ) -> None:
        super().__init__()
        self.upsample = _DUC(input_dim, 2 * input_dim, init_type=init_type)
        self.conv_block = _DilatedRedisual(input_dim, output_dim, init_type=init_type)

    def forward(self, input: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.conv_block(torch.cat((self.upsample(input), skip), dim=1))


class TransMUNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        config: DictConfig,
        init_type: Optional[str] = None,
    ) -> None:
        super().__init__()
        dims = config.get("dims")
        length = len(dims)
        trunk_config = config.get("trunk")
        patch_size = trunk_config.pop("patch_size")
        block_configs = trunk_config.pop("block_configs")

        self.encoder = Encoder(
            input_dim,
            dims,
            _DilatedRedisual,
            _EncoderBlock,
            init_type,
            length=length,
            block_configs=block_configs,
            patch_size=patch_size,
            transformer_config=trunk_config,
        )

        self.decoder = Decoder(dims, _DecoderBlock, init_type)

        self.boundary = _Boundary(dims[0], 1, init_type=init_type)
        self.output_block = OutputBlock(dims[0], output_dim, init_type=init_type)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x_list = self.encoder(input)
        x = x_list.pop()
        B_out = repeat(
            self.boundary(x_list[0]), "b 1 h w -> b c h w", c=x_list[0].shape[1]
        )
        x_list[0] = x_list[0] + B_out
        x = self.decoder(x, x_list)
        return F.sigmoid(B_out), self.output_block(x)
