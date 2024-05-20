import math
from typing import Optional

import torch
import torch.nn as nn
from ml_collections import ConfigDict

from ..modules.base import BasicBlock, Conv2dSame, OutputBlock
from .setr_config import CONFIGS
from .vit import VisionTransformer


class _Decoder_PUP(nn.Module):
    def __init__(self, output_dim: int, config: ConfigDict):
        super().__init__()
        in_channels = [config.get("hidden_size")] + list(
            config.get("decoder_channels")
        )[:-1]
        out_channels = config.get("decoder_channels")

        self.layers = nn.Sequential(
            *[
                nn.Sequential(
                    BasicBlock(in_channel, out_channel),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                )
                for in_channel, out_channel in zip(in_channels, out_channels)
            ]
        )

        self.output_block = OutputBlock(out_channels[-1], output_dim)

    def forward(self, input: torch.Tensor):
        x = self._reshape_input(input)
        x = self.layers(x)
        output = self.output_block(x)
        return output

    def _reshape_input(self, input):
        (n_batch, n_patch, hidden) = input.size()
        height, width = int(math.sqrt(n_patch)), int(math.sqrt(n_patch))
        x = input.permute(0, 2, 1)
        x = x.contiguous().view(n_batch, hidden, height, width)
        return x


class _Decoder_MLA(nn.Module):
    def __init__(self, output_dim: int, config: ConfigDict):
        super().__init__()
        self.embedding_dim = config.get("hidden_size")

        self.block1_in, _, self.block1_out = self._define_agg_block()
        self.block2_in, self.block2_intmd, self.block2_out = self._define_agg_block()
        self.block3_in, self.block3_intmd, self.block3_out = self._define_agg_block()
        self.block4_in, self.block4_intmd, self.block4_out = self._define_agg_block()

        self.output_block = nn.Sequential(
            Conv2dSame(
                self.embedding_dim, output_dim, kernel_size=1, stride=1, padding="same"
            ),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
            nn.Sigmoid(),
        )

    def forward(self, *inputs):
        x1 = self._reshape_input(inputs[0])
        x1_in = self.block1_in(x1)
        x1 = self.block1_out(x1_in)

        x2 = self._reshape_input(inputs[1])
        x2_in = self.block2_in(x2)
        x2_intmd_in = x1_in + x2_in
        x2_intmd = self.block2_intmd(x2_intmd_in)
        x2 = self.block2_out(x2_intmd)

        x3 = self._reshape_input(inputs[2])
        x3_in = self.block3_in(x3)
        x3_intmd_in = x2_intmd_in + x3_in
        x3_intmd = self.block3_intmd(x3_intmd_in)
        x3 = self.block3_out(x3_intmd)

        x4 = self._reshape_input(inputs[3])
        x4_in = self.block4_in(x4)
        x4_intmd_in = x3_intmd_in + x4_in
        x4_intmd = self.block4_intmd(x4_intmd_in)
        x4 = self.block4_out(x4_intmd)

        x5 = torch.cat([x1, x2, x3, x4], dim=1)
        output = self.output_block(x5)
        return output

    def _define_agg_block(self):
        model_in = Conv2dSame(
            self.embedding_dim, self.embedding_dim // 2, kernel_size=1, padding="same"
        )
        model_intmd = Conv2dSame(
            self.embedding_dim // 2,
            self.embedding_dim // 2,
            kernel_size=3,
            padding="same",
        )
        model_output = nn.Sequential(
            Conv2dSame(
                self.embedding_dim // 2,
                self.embedding_dim // 4,
                kernel_size=3,
                padding="same",
            ),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
        )
        return model_in, model_intmd, model_output

    def _reshape_input(self, input):
        (n_batch, n_patch, hidden) = input.size()
        height, width = int(math.sqrt(n_patch)), int(math.sqrt(n_patch))
        x = input.permute(0, 2, 1)
        x = x.contiguous().view(n_batch, hidden, height, width)
        return x


# TODO auxiliary loss
class SETR(nn.Module):
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
        self.return_intermediate = self.config.get("decoder_classifier") == "MLA"

        self.encoder = VisionTransformer(
            input_dim, img_size, self.config, self.return_intermediate
        )
        self.decoder = (
            _Decoder_MLA(output_dim, self.config)
            if self.return_intermediate
            else _Decoder_PUP(output_dim, self.config)
        )

    def forward(self, input):
        x = self.encoder(input)
        if not isinstance(x, tuple):
            x = x.squeeze()
            output = self.decoder(x)
        else:
            if not self.return_intermediate:
                x = x[0]
                x = x.squeeze()
                output = self.decoder(x)
            else:
                x = x[:4]
                x = [temp.squeeze() for temp in x]
                output = self.decoder(*x)
        return output
