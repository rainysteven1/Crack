from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.base import InitModule, OutputBlock
from ._base import Decoder, DoubleConv, Encoder


class _DecoderBlock(InitModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        _: int,
        ratio: int,
        is_upsample: bool = True,
        init_type: Optional[str] = None,
    ) -> None:
        super().__init__()
        output_dim = output_dim // ratio

        if is_upsample:
            self.upsample = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            )
            self.conv_block = DoubleConv(
                input_dim, output_dim, input_dim // 2, init_type=init_type
            )
        else:
            self.upsample = nn.ConvTranspose2d(
                input_dim, input_dim // 2, kernel_size=2, stride=2
            )
            self.conv_block = DoubleConv(input_dim, output_dim, init_type=init_type)

    def forward(self, input: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(input)
        # input is CHW
        if skip.shape != x.shape:
            diff_y = skip.size()[2] - x.size()[2]
            diff_x = skip.size()[3] - x.size()[3]
            x = F.pad(
                x,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )
        return self.conv_block(torch.cat((x, skip), dim=1))

    def _initialize_weights(self):
        self.init(self.upsample)


class UNet(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dims: List[int],
        is_upsample: bool = False,
        init_type: Optional[str] = None,
    ) -> None:
        super().__init__()
        ratio = 2 if is_upsample else 1
        decoder_config = {"is_upsample": is_upsample, "ratio": ratio}

        self.encoder = Encoder(
            input_dim, dims[:-1] + [dims[-1] // ratio], init_type=init_type
        )
        self.decoder = Decoder(dims, _DecoderBlock, init_type, **decoder_config)
        self.output_block = OutputBlock(dims[0], output_dim, init_type=init_type)

    def forward(self, input: torch.Tensor):
        x_list = self.encoder(input)
        x = x_list.pop()
        x = self.decoder(x, x_list)
        return self.output_block(x)
