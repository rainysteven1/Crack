import math
from typing import List, Optional, OrderedDict

import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import DictConfig

from ...modules.base import BasicBlock
from ...transformer.vit import VisionTransformer

__all__ = ["TransUNet"]


class _PatchProjection(nn.Conv2d):

    root = "conv_proj.{}"

    def __init__(
        self, input_dim: int, embedding_dim: int, patch_size: int, n_patch: int
    ) -> None:
        super().__init__(
            input_dim, embedding_dim, kernel_size=patch_size, stride=patch_size
        )

    def load_from(self, weights: OrderedDict):
        state_dict = {
            key: weights[self.root.format(key)] for key in self.state_dict().keys()
        }
        self.load_state_dict(state_dict)


class _DecoderBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        skip_dim: int = 0,
        init_type: Optional[str] = None,
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
        if skip:
            x = torch.cat((x, skip), dim=1)
        return self.conv_block(x)


class _Decoder(nn.Module):

    head_dim = 512

    def __init__(
        self,
        embedding_dim: int,
        decoder_dims: List[int],
        n_skips: Optional[int],
        skip_dims: Optional[List[int]],
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

        self.conv_block = BasicBlock(embedding_dim, self.head_dim)
        self.decoder_blocks = nn.ModuleList(
            [
                _DecoderBlock(*list(dims))
                for dims in zip(input_dims, output_dims, skip_dims)
            ]
        )

    def forward(self, input: torch.Tensor, skips: Optional[torch.Tensor] = None):
        height = weight = int(math.sqrt(input.shape[1]))
        x = rearrange(input, "b (h w) d -> b d h w", h=height, w=weight)
        x = self.conv_block(x)
        for idx, module in enumerate(self.decoder_blocks):
            skip = None if not skips else skips[idx] if idx < self.n_skips else None
            x = module(x, skip)
        return x


class _SegmentationHead(nn.Sequential):
    def __init__(self, input_dim: int, output_dim: int):
        conv = nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1)
        upsample = nn.Upsample(scale_factor=1, mode="bilinear")

        super().__init__(conv, upsample)


class TransUNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        config: DictConfig,
    ):
        super().__init__()
        print(config)
        decoder_config = config.pop("decoder")

        self.transformer = VisionTransformer(input_dim, config, _PatchProjection)
        self.decoder = _Decoder(**decoder_config)
        self.segmentation_head = _SegmentationHead(
            decoder_config.get("decoder_dims")[-1], output_dim
        )

    def forward(self, input: torch.Tensor):
        x = self.transformer(input)
        if isinstance(x, tuple):
            features = x[1:]
            x = x[0]
        else:
            features = None
        x = self.decoder(x, features)
        x = self.segmentation_head(x)
        return torch.sigmoid(x)
