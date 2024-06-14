from typing import List, Optional, Union

import torch
import torch.nn as nn

from ..modules.base import OutputBlock
from ._base import DoubleConv, Encoder
from .attention_unet import DecoderBlock as BasicDecoderBlock

__all__ = ["AttentionUNet2Plus"]


class _DecoderBlock(BasicDecoderBlock):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        _: int,
        is_upsample: bool = True,
        is_max_pool: bool = False,
        N_concat: int = 2,
        init_type: Optional[str] = None,
    ):
        super().__init__(input_dim, output_dim, is_upsample, init_type)
        self.is_max_pool = is_max_pool

        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.conv_block = (
            DoubleConv(
                input_dim // 2 + (N_concat - 1) * output_dim,
                output_dim,
                is_bn=False,
                init_type=init_type,
            )
            if not is_max_pool
            else DoubleConv(
                input_dim // 2 + output_dim // 2 + (N_concat - 2) * output_dim,
                output_dim,
                is_bn=False,
                init_type=init_type,
            )
        )

    def forward(self, input: torch.Tensor, *skips: torch.Tensor) -> torch.Tensor:
        x1 = self.up_conv(input)
        x2 = self.attention_block(x1, skips[-1])
        concat_list = (
            [*skips[:-1], x2, x1]
            if not self.is_max_pool
            else [self.max_pool(skips[0]), *skips[1:-1], x2, x1]
        )
        x3 = torch.cat(concat_list, dim=1)
        return self.conv_block(x3)


class AttentionUNet2Plus(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dims: List[int],
        is_upsample: bool = True,
        is_ds: bool = True,
        init_type: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.is_ds = is_ds

        # Encoder
        self.e = Encoder(input_dim, dims, init_type=init_type)

        # Decoder
        self.d01 = _DecoderBlock(dims[1], dims[0], is_upsample, init_type=init_type)
        self.d11 = _DecoderBlock(
            dims[2], dims[1], is_upsample, True, N_concat=3, init_type=init_type
        )
        self.d21 = _DecoderBlock(
            dims[3], dims[2], is_upsample, True, N_concat=3, init_type=init_type
        )
        self.d31 = _DecoderBlock(
            dims[4], dims[3], is_upsample, True, N_concat=3, init_type=init_type
        )

        self.d02 = _DecoderBlock(
            dims[1], dims[0], is_upsample, N_concat=3, init_type=init_type
        )
        self.d12 = _DecoderBlock(
            dims[2], dims[1], is_upsample, True, N_concat=4, init_type=init_type
        )
        self.d22 = _DecoderBlock(
            dims[3], dims[2], is_upsample, True, N_concat=4, init_type=init_type
        )

        self.d03 = _DecoderBlock(
            dims[1], dims[0], is_upsample, N_concat=4, init_type=init_type
        )
        self.d13 = _DecoderBlock(
            dims[2], dims[1], is_upsample, True, N_concat=5, init_type=init_type
        )

        self.d04 = _DecoderBlock(
            dims[1], dims[0], is_upsample, N_concat=5, init_type=init_type
        )

        self.final = (
            OutputBlock(dims[0], output_dim, init_type)
            if not is_ds
            else nn.ModuleList(
                [OutputBlock(dims[0], output_dim, init_type=init_type)] * 4
            )
        )

    def forward(self, input: torch.Tensor) -> Union[List[torch.Tensor], torch.Tensor]:
        # Encoder
        x00, x10, x20, x30, x40 = self.e(input)

        # Decoder
        x01 = self.d01(x10, x00)
        x11 = self.d11(x20, x01, x10)
        x21 = self.d21(x30, x11, x20)
        x31 = self.d31(x40, x21, x30)

        x02 = self.d02(x11, x00, x01)
        x12 = self.d12(x21, x02, x10, x11)
        x22 = self.d22(x31, x12, x20, x21)

        x03 = self.d03(x12, x00, x01, x02)
        x13 = self.d13(x22, x03, x10, x11, x12)

        x04 = self.d04(x13, x00, x01, x02, x03)

        x_list = [x01, x02, x03, x04]

        # Output
        outputs = [module(x) for x, module in zip(x_list, self.final)]
        return outputs if self.is_ds else torch.mean(outputs)
