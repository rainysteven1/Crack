from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.base import BasicBlock, OutputBlock

__all__ = ["u2net_B", "u2net_S"]


class _RSU0_DecoderBlock(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        is_upsample: bool,
        init_type: Optional[str],
    ) -> None:
        super().__init__()

        self.upsample = (
            BasicBlock(
                output_dim,
                output_dim,
                padding=2,
                dilation=2,
                init_type=init_type,
            )
            if not is_upsample
            else nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.conv_block = BasicBlock(input_dim, output_dim, init_type=init_type)

    def forward(self, input: torch.Tensor, skip: torch.Tensor):
        x = self.upsample(input)
        return self.conv_block(torch.cat((x, skip), dim=1))


class _RSU0(nn.Module):
    """对应U2Net架构图中的Stage1~Stage3."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        middle_dim: int,
        n_blocks: int,
        init_type: Optional[str],
    ) -> None:
        super().__init__()

        self.input_block = BasicBlock(input_dim, output_dim, init_type=init_type)
        self.encoder_blocks = nn.ModuleList(
            [BasicBlock(output_dim, middle_dim, init_type=init_type)]
            + [
                nn.Sequential(
                    nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
                    BasicBlock(middle_dim, middle_dim, init_type=init_type),
                )
                for _ in range(n_blocks - 1)
            ]
        )
        self.decoder_blocks = nn.ModuleList(
            [
                _RSU0_DecoderBlock(
                    middle_dim * 2,
                    middle_dim if i != n_blocks - 1 else output_dim,
                    is_upsample=i != 0,
                    init_type=init_type,
                )
                for i in range(n_blocks)
            ]
        )

    def forward(self, input: torch.Tensor):
        x = self.input_block(input)
        x_list = [x]
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            x_list.append(x)
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, x_list.pop())
        return x + x_list.pop()


class _RSU1(nn.Module):
    """对应U2Net架构图中的Stage4和Stage5."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        middle_dim: int,
        n_block: int,
        init_type: Optional[str],
    ) -> None:
        super().__init__()

        self.input_block = BasicBlock(input_dim, output_dim, init_type=init_type)
        self.encoder_blocks = nn.ModuleList(
            [
                BasicBlock(
                    output_dim if i == 0 else middle_dim,
                    middle_dim,
                    padding=2**i,
                    dilation=2**i,
                    init_type=init_type,
                )
                for i in range(n_block)
            ]
        )
        self.bridge_block = BasicBlock(
            middle_dim,
            middle_dim,
            padding=2**n_block,
            dilation=2**n_block,
            init_type=init_type,
        )
        self.decoder_blocks = nn.ModuleList(
            [
                BasicBlock(
                    middle_dim * 2,
                    output_dim if i == 0 else middle_dim,
                    padding=2**i,
                    dilation=2**i,
                    init_type=init_type,
                )
                for i in range(n_block - 1, -1, -1)
            ]
        )

    def forward(self, input: torch.Tensor):
        x = self.input_block(input)
        x_list = [x]
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            x_list.append(x)
        x = self.bridge_block(x)
        for decoder_block in self.decoder_blocks:
            x = decoder_block(torch.cat((x, x_list.pop()), dim=1))
        return x + x_list.pop()


class _DecoderBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        middle_dim: int,
        n_block: int,
        is_RSU1: bool,
        init_type: Optional[str],
    ) -> None:
        super().__init__()
        conv_module = _RSU1 if is_RSU1 else _RSU0

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_block = conv_module(
            input_dim, output_dim, middle_dim, n_block, init_type
        )

    def forward(self, input: torch.Tensor, skip: torch.Tensor):
        x = self.upsample(input)
        return self.conv_block(torch.cat((x, skip), dim=1))


class _U2Net(nn.Module):

    n_RUS1 = 2

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dims: list,
        n_blocks: list[int],
        e_RUS0_ratio: int,
        e_RUS1_ratio: int,
        d_RUS0_ratio: int,
        d_RUS1_ratio: int,
        init_type: Optional[str],
    ) -> None:
        assert len(dims) == len(n_blocks)
        assert min(n_blocks) >= 3

        super().__init__()
        new_dims = [dims[0], *dims, dims[-1]]
        length = len(dims)

        self.input_block = _RSU0(
            input_dim, dims[0], dims[0] // 2, n_blocks[0], init_type
        )
        self.encoder_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
                    _RSU0(
                        dims[i],
                        dims[i + 1],
                        dims[i + 1] // e_RUS0_ratio,
                        n_blocks[i + 1],
                        init_type,
                    ),
                )
                for i in range(length - 1)
            ]
            + [
                nn.Sequential(
                    nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
                    _RSU1(
                        dims[-1],
                        dims[-1],
                        dims[-1] // e_RUS1_ratio,
                        n_blocks[-1] - 1,
                        init_type,
                    ),
                )
            ]
            * self.n_RUS1
        )
        self.decoder_blocks = nn.ModuleList(
            [
                _DecoderBlock(
                    new_dims[-2] * 2,
                    new_dims[-2],
                    dims[-1] // d_RUS0_ratio,
                    n_blocks[-1] - 1,
                    True,
                    init_type,
                )
            ]
            * (self.n_RUS1 - 1)
            + [
                _DecoderBlock(
                    dims[i] * 2,
                    new_dims[i],
                    dims[i] // d_RUS1_ratio,
                    n_blocks[i],
                    False,
                    init_type,
                )
                for i in range(length - 1, -1, -1)
            ]
        )

        length = len(new_dims)
        self.slide_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    BasicBlock(
                        new_dims[i],
                        output_dim,
                        is_bn=False,
                        is_relu=False,
                        init_type=init_type,
                    ),
                    nn.Upsample(scale_factor=2**i, mode="bilinear", align_corners=True),
                )
                for i in range(length)
            ]
        )
        self.output_block = OutputBlock(
            length * output_dim, output_dim, init_type=init_type
        )

    def forward(self, input: torch.Tensor):
        x = self.input_block(input)
        x_list = [x]
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            x_list.append(x)
        x = x_list.pop()
        slide_list = [x]
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, x_list.pop())
            slide_list.append(x)
        slide_list = [
            slide_block(slide)
            for slide_block, slide in zip(self.slide_blocks, reversed(slide_list))
        ]
        output = self.output_block(torch.cat(slide_list, dim=1))
        outputs = [F.sigmoid(slide) for slide in slide_list]
        outputs.append(output)
        return outputs


def u2net_B(
    input_dim: int,
    output_dim: int,
    dims: list,
    n_blocks: list[int],
    init_type: Optional[str],
):
    kwargs = {
        "e_RUS0_ratio": 4,
        "e_RUS1_ratio": 2,
        "d_RUS0_ratio": 2,
        "d_RUS1_ratio": 4,
    }
    return _U2Net(input_dim, output_dim, dims, n_blocks, init_type=init_type, **kwargs)


def u2net_S(
    input_dim: int,
    output_dim: int,
    dims: list,
    n_blocks: list[int],
    init_type: Optional[str],
):
    kwargs = {
        "e_RUS0_ratio": 4,
        "e_RUS1_ratio": 4,
        "d_RUS0_ratio": 4,
        "d_RUS1_ratio": 4,
    }
    assert len(set(dims)) == 1
    return _U2Net(input_dim, output_dim, dims, n_blocks, init_type=init_type, **kwargs)
