from typing import List, Optional

import torch
import torch.nn as nn

from ...backbone.resnet import RedisualBlock
from ...modules.base import BasicBlock, OutputBlock, SqueezeExciteBlock
from ...modules.pyramid import ASPP_v3

__all__ = ["ResUNet2Plus"]

kwargs = {"skip_kernel_size": 1, "skip_padding": 0, "is_bias": True, "reversed": False}


class _AttentionBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        skip_dim: int,
        init_type: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.conv_block = BasicBlock(
            input_dim, output_dim, reversed=True, init_type=init_type
        )
        self.skip_block = nn.Sequential(
            BasicBlock(skip_dim, output_dim, reversed=True, init_type=init_type),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv_attn = BasicBlock(
            output_dim, output_dim, reversed=True, init_type=init_type
        )

    def forward(self, input: torch.Tensor, skip: torch.Tensor):
        x = self.conv_block(input) + self.skip_block(skip)
        return input * self.conv_attn(x)


class _DecoderBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, skip_dim: int) -> None:
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2)
        self.attention = _AttentionBlock(
            input_dim, input_dim, skip_dim, kwargs.get("init_type")
        )
        self.redisual = RedisualBlock(input_dim + skip_dim, output_dim, **kwargs)

    def forward(self, input: torch.Tensor, skip: torch.Tensor):
        x = torch.cat((self.upsample(self.attention(input, skip)), skip), dim=1)
        return self.redisual(x)


class ResUNet2Plus(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dims: List[int],
        atrous_rates: List[int],
        init_type: Optional[str] = None,
    ) -> None:
        super().__init__()
        length = len(dims)
        kwargs.update({"init_type": init_type})

        self.input_block = RedisualBlock(input_dim, dims[0], is_bn=False, **kwargs)
        kwargs.update({"reversed": True})
        self.encoder_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    SqueezeExciteBlock(dims[i]),
                    RedisualBlock(
                        dims[i],
                        dims[i + 1],
                        stride=2,
                        **kwargs,
                    ),
                )
                for i in range(length - 2)
            ]
            + [
                ASPP_v3(
                    dims[-2],
                    dims[-1],
                    atrous_rates,
                    init_type,
                )
            ]
        )
        self.decoder_blocks = nn.ModuleList(
            _DecoderBlock(
                dims[i],
                dims[i - 1],
                dims[i - 2],
            )
            for i in range(length - 1, 1, -1)
        )
        self.output_block = nn.Sequential(
            ASPP_v3(
                dims[1],
                dims[0],
                atrous_rates,
                init_type,
            ),
            OutputBlock(dims[0], output_dim, init_type=init_type),
        )

    def forward(self, input: torch.Tensor):
        x = self.input_block(input)
        x_list = [x]
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            x_list.append(x)
        x_list = x_list[:-2]
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, x_list.pop())
        return self.output_block(x)
