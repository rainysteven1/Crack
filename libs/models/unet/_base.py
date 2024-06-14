from typing import Callable, List, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig

from ..modules.base import BasicBlock

__all__ = ["DoubleConv", "Decoder", "Encoder"]


class DoubleConv(nn.Sequential):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        middle_dim: Optional[int] = None,
        is_bn: bool = True,
        is_pool: bool = False,
        init_type: Optional[str] = None,
    ) -> None:
        middle_dim = middle_dim or output_dim

        layers = [
            BasicBlock(input_dim, middle_dim, is_bn=is_bn, init_type=init_type),
            BasicBlock(middle_dim, output_dim, is_bn=is_bn, init_type=init_type),
        ]
        if is_pool:
            layers.append(nn.MaxPool2d(kernel_size=2))
        super().__init__(*layers)


class _EncoderBlock(nn.Sequential):
    def __init__(
        self, input_dim: int, output_dim: int, init_type: Optional[str] = None
    ) -> None:
        super().__init__(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(input_dim, output_dim, init_type=init_type),
        )


class Encoder(nn.Module):

    def __init__(
        self,
        input_dim: int,
        dims: List[int],
        stem_block: Callable[..., torch.Tensor] = DoubleConv,
        encoder_block: Callable[..., torch.Tensor] = _EncoderBlock,
        init_type: Optional[str] = None,
        **kwargs
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            [stem_block(input_dim, dims[0], init_type=init_type)]
            + [
                encoder_block(dims[i], dims[i + 1], init_type=init_type, **kwargs)
                for i in range(len(dims) - 1)
            ]
        )

    def forward(self, input: torch.Tensor) -> List[torch.Tensor]:
        x = input
        x_list = list()
        for layer in self.layers:
            x = layer(x)
            x_list.append(x)
        return x_list


class Decoder(nn.Module):
    def __init__(
        self,
        dims: List[int],
        decoder_block: Callable[..., torch.Tensor],
        init_type: Optional[str] = None,
        **kwargs
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            [
                decoder_block(dims[i], dims[i - 1], i, init_type=init_type, **kwargs)
                for i in range(len(dims) - 1, 0, -1)
            ]
        )

    def forward(
        self, input: torch.Tensor, *skips_tuple: List[torch.Tensor]
    ) -> torch.Tensor:
        x = input
        for layer in self.layers:
            skips = [skips.pop() for skips in skips_tuple]
            x = layer(x, *skips)
        return x
