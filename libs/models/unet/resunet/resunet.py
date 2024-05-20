from typing import Optional

import torch
import torch.nn as nn

from ...modules.base import OutputBlock
from ...modules.resnet import RedisualBlock

kwargs = {"skip_kernel_size": 3, "skip_padding": 1, "is_bias": True, "reversed": False}


class _DecoderBlock(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, skip_dim: int, is_upsample: bool
    ) -> None:
        super().__init__()

        self.upsample = (
            nn.Upsample(scale_factor=2, mode="nearest")
            if is_upsample
            else nn.ConvTranspose2d(input_dim, input_dim, kernel_size=2, stride=2)
        )
        self.redisual = RedisualBlock(input_dim + skip_dim, output_dim, **kwargs)

    def forward(self, input: torch.Tensor, skip: torch.Tensor):
        return self.redisual(torch.cat((self.upsample(input), skip), dim=1))


class ResUNet(nn.Module):
    layer_configurations = [64, 128, 256, 512]

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        init_type: Optional[str] = None,
    ) -> None:
        super().__init__()
        length = len(self.layer_configurations)
        is_upsample = False
        kwargs.update({"init_type": init_type})

        self.input_block = RedisualBlock(
            input_dim, self.layer_configurations[0], is_bn=False, **kwargs
        )
        kwargs.update({"reversed": True})
        self.encoder_blocks = nn.ModuleList(
            [
                RedisualBlock(
                    self.layer_configurations[i],
                    self.layer_configurations[i + 1],
                    stride=2,
                    **kwargs,
                )
                for i in range(length - 1)
            ]
        )
        self.decoder_blocks = nn.ModuleList(
            [
                _DecoderBlock(
                    self.layer_configurations[i],
                    self.layer_configurations[i - 1],
                    self.layer_configurations[i - 1],
                    is_upsample,
                )
                for i in range(length - 1, 0, -1)
            ]
        )
        self.output_block = OutputBlock(
            self.layer_configurations[0], output_dim, init_type=init_type
        )

    def forward(self, input: torch.Tensor):
        x = self.input_block(input)
        x_list = [x]
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            x_list.append(x)
        x_list = x_list[:-1]
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, x_list.pop())
        return self.output_block(x)
