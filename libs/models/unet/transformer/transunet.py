import math
import numpy as np
import torch
import torch.nn as nn

from ml_collections import ConfigDict
from typing import Optional

from ...modules.base import Conv2dSame, SingleBlock
from ...modules.transformer import *


class _DecoderBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, skip_dim: int = 0) -> None:
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        self.layers = nn.Sequential(
            SingleBlock(input_dim + skip_dim, output_dim),
            SingleBlock(output_dim, output_dim),
        )

    def forward(self, input, skip=None):
        x = self.upsample(input)
        if skip:
            x = torch.cat([x, skip], dim=1)
        output = self.layers(x)
        return output


class _Decoder(nn.Module):
    def __init__(self, config: ConfigDict) -> None:
        super().__init__()
        filters = config.get("decoder_channels")
        head_channel = 512
        in_channels = [head_channel] + list(filters[:-1])
        out_channels = filters
        if config.n_skip != 0:
            skip_channels = config.skip_channels
            for i in range(4 - config.n_skip):
                skip_channels[3 - i] = 0
        else:
            skip_channels = [0, 0, 0, 0]
        self.config = config

        self.conv_block = SingleBlock(config.hidden_size, head_channel)
        self.layers = nn.ModuleList(
            [
                _DecoderBlock(*list(dims))
                for dims in zip(in_channels, out_channels, skip_channels)
            ]
        )

    def forward(self, input, skips=None):
        x = self._reshape_input(input)
        x = self.conv_block(x)
        for idx, module in enumerate(self.layers):
            skip = (
                None if not skips else skips[idx] if idx < self.config.n_skip else None
            )
            x = module(x, skip)
        return x

    def _reshape_input(self, input):
        (n_batch, n_patch, hidden) = input.size()
        height, width = int(math.sqrt(n_patch)), int(math.sqrt(n_patch))
        x = input.permute(0, 2, 1)
        x = x.contiguous().view(n_batch, hidden, height, width)
        return x


class _SegmentationHead(nn.Sequential):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int = 3,
        is_upsample: bool = True,
    ):
        conv = Conv2dSame(
            input_dim, output_dim, kernel_size=kernel_size, padding=kernel_size // 2
        )
        upsample = (
            nn.Upsample(scale_factor=1, mode="bilinear")
            if is_upsample
            else nn.Identity()
        )

        super().__init__(conv, upsample)


class TransUNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        img_size: int = 256,
        train_config: Optional[str] = "ViT-B_16",
        test_config: Optional[str] = None,
    ):
        super().__init__()
        self.config = (
            CONFIGS.get(train_config)()
            if not test_config
            else CONFIGS.get(test_config)(train_config)
        )

        self.encoder = TransModel(input_dim, img_size, self.config)
        self.decoder = _Decoder(self.config)
        self.segmentation_head = _SegmentationHead(
            self.config["decoder_channels"][-1], output_dim
        )

        if not test_config:
            self._load_from()

    def forward(self, input):
        x = self.encoder(input)
        if isinstance(x, tuple):
            features = x[1:]
            x = x[0]
        else:
            features = None
        x = x.squeeze()
        x = self.decoder(x, features)
        x = self.segmentation_head(x)
        output = torch.sigmoid(x)
        return output

    def _load_from(self):
        weights = np.load(self.config.pretrained_path)
        with torch.no_grad():
            self.encoder.load_from(weights)
