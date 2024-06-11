import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from omegaconf import DictConfig

from ...modules.base import BasicBlock
from ...transformer.vit import VisionTransformer

__all__ = ["TransUNet"]


class _DecoderBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        skip_dim: int,
        init_type: Optional[str],
    ) -> None:
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv_block = nn.Sequential(
            BasicBlock(
                input_dim + skip_dim, output_dim, is_bias=False, init_type=init_type
            ),
            BasicBlock(output_dim, output_dim, is_bias=False, init_type=init_type),
        )

    def forward(self, input: torch.Tensor, skip: Optional[torch.Tensor] = None):
        x = self.upsample(input)
        if skip is not None:
            x = torch.cat((x, skip), dim=1)
        return self.conv_block(x)


class _Decoder(nn.Module):

    head_dim = 512

    def __init__(
        self,
        embedding_dim: int,
        decoder_dims: List[int],
        init_type: Optional[str],
        n_skips: Optional[int] = None,
        skip_dims: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        input_dims = [self.head_dim] + list(decoder_dims[:-1])
        output_dims = decoder_dims
        self.n_skips = n_skips

        if n_skips and skip_dims:
            for i in range(len(skip_dims) - n_skips):
                skip_dims[3 - i] = 0
        else:
            skip_dims = [0] * len(decoder_dims)

        self.conv_block = BasicBlock(
            embedding_dim, self.head_dim, is_bias=False, init_type=init_type
        )
        self.decoder_blocks = nn.ModuleList(
            [
                _DecoderBlock(*list(dims), init_type)
                for dims in zip(input_dims, output_dims, skip_dims)
            ]
        )

    def forward(self, input: torch.Tensor, skips: Optional[torch.Tensor] = None):
        height = weight = int(math.sqrt(input.shape[1]))
        x = rearrange(input, "b (h w) d -> b d h w", h=height, w=weight)
        x = self.conv_block(x)
        for idx, decoder_block in enumerate(self.decoder_blocks):
            skip = None if not skips else skips[idx] if idx < self.n_skips else None
            x = decoder_block(x, skip)
        return x


class _SegmentationHead(nn.Sequential):
    def __init__(self, input_dim: int, output_dim: int, init_type: Optional[str]):
        conv = BasicBlock(
            input_dim, output_dim, is_bn=False, is_relu=False, init_type=init_type
        )
        upsample = nn.Upsample(scale_factor=1, mode="bilinear")

        super().__init__(conv, upsample)


class TransUNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        config: DictConfig,
        init_type: Optional[str],
    ):
        super().__init__()
        decoder_config = config.pop("decoder")

        self.transformer = VisionTransformer(input_dim, config)
        self.decoder = _Decoder(init_type=init_type, **decoder_config)
        self.segmentation_head = _SegmentationHead(
            decoder_config.get("decoder_dims")[-1], output_dim, init_type
        )

    def forward(self, input: torch.Tensor):
        x, features = self.transformer(input)
        x = self.decoder(x, features)
        x = self.segmentation_head(x)
        return F.sigmoid(x)
