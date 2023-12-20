import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Conv2dSame

_BOTTLENECK_EXPANSION = 4


class BasicBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int,
        stride: int,
        padding: str | int = "same",
        dilation: int = 1,
        is_activate: bool = True,
    ) -> None:
        super().__init__()

        layer_list = [
            Conv2dSame(
                input_dim,
                output_dim,
                kernel_size,
                stride,
                padding,
                dilation,
                bias=False,
            ),
            nn.BatchNorm2d(output_dim, eps=1e-5, momentum=1e-3),
        ]
        if is_activate:
            layer_list.append(nn.ReLU())
        self.layers = nn.Sequential(*layer_list)

    def forward(self, input):
        return self.layers(input)


class _BottleNeck(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        stride: int,
        dilation: int,
        is_downsample: bool,
    ) -> None:
        super().__init__()
        middle_dim = output_dim // _BOTTLENECK_EXPANSION

        self.conv_block = nn.Sequential(
            BasicBlock(input_dim, middle_dim, 1, stride, 0, dilation, True),
            BasicBlock(middle_dim, middle_dim, 3, 1, dilation, dilation, True),
            BasicBlock(middle_dim, output_dim, 1, 1, 0, 1, False),
        )
        self.skip_block = (
            BasicBlock(input_dim, output_dim, 1, stride, 0, 1, False)
            if is_downsample
            else nn.Identity()
        )

    def forward(self, input):
        x1 = self.conv_block(input)
        x2 = self.skip_block(input)
        output = F.relu(torch.add(x1, x2))
        return output


class _ResLayer(nn.Module):
    """
    Residual layer with multi grids
    """

    def __init__(
        self, n_layers, input_dim, output_dim, stride, dilation, multi_grids=None
    ):
        super(_ResLayer, self).__init__()

        if multi_grids is None:
            multi_grids = [1 for _ in range(n_layers)]
        else:
            assert n_layers == len(multi_grids)

        # Downsampling is only in the first block
        layer_list = [
            _BottleNeck(
                input_dim if i == 0 else output_dim,
                output_dim,
                stride if i == 0 else 1,
                dilation * multi_grids[i],
                is_downsample=i == 0,
            )
            for i in range(n_layers)
        ]
        self.layers = nn.Sequential(*layer_list)

    def forward(self, input):
        return self.layers(input)


class ResNet0(nn.Module):
    """
    ResNet没有全连接层
    """

    def __init__(
        self,
        n_blocks: list,
        n_dims: list,
        layer_params: list[tuple],
        input_block: nn.Module,
    ):
        super().__init__()
        assert len(n_blocks) == len(layer_params) == len(n_dims) - 2

        layer_list = list()
        layer_list.append(input_block)
        for i in range(len(n_blocks)):
            layer_list.append(
                _ResLayer(
                    n_blocks[i],
                    n_dims[i] if i == 0 else n_dims[i + 1],
                    n_dims[i + 2],
                    *layer_params[i],
                )
            )
        self.layers = nn.Sequential(*layer_list)

    def forward(self, input):
        return self.layers(input)
