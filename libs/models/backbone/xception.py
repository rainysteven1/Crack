from typing import List, Optional, OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.base import BasicBlock, IntermediateSequential
from ._utils import fixed_padding

__all__ = ["xception"]

_PRETRAINED_MODELS = {
    "xception": "resources/checkpoints/xception-43020ad28.pth",
}


_REPLACE_DICT = {"8": [False, False, True, True], "16": [False, False, False, True]}

_REDISUAL_BLOCK_CFG = [
    # output_dim, n_layer, stride, starts_with_relu, grow_first, replace_stride_with_dilation_idx
    [128, 2, 2, False, True, 0],  # Entry flow
    [256, 2, 2, True, True, 1],
    [728, 2, 2, True, True, 2],
    [728, 3, 1, True, True, 2],  # Middle flow (repeat 8 times)
    [728, 3, 1, True, True, 2],
    [728, 3, 1, True, True, 2],
    [728, 3, 1, True, True, 2],
    [728, 3, 1, True, True, 2],
    [728, 3, 1, True, True, 2],
    [728, 3, 1, True, True, 2],
    [728, 3, 1, True, True, 2],
    [1024, 2, 2, True, False, 3],  # Exit flow
]


class _SeparableConv2d(nn.Sequential):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        is_bn: bool,
        is_bias: bool,
        init_type: Optional[str],
    ) -> None:
        # depthwise和pointwise的前后顺序不影响结果
        super().__init__(
            *[
                # Depthwise
                BasicBlock(
                    input_dim,
                    input_dim,
                    kernel_size,
                    stride,
                    0,
                    dilation,
                    groups=input_dim,
                    is_bn=is_bn,
                    is_relu=False,
                    is_bias=is_bias,
                    init_type=init_type,
                ),
                # Pointwise
                BasicBlock(
                    input_dim,
                    output_dim,
                    kernel_size=1,
                    padding=0,
                    is_bn=False,
                    is_relu=False,
                    is_bias=is_bias,
                    init_type=init_type,
                ),
            ]
        )
        self.input_padding = fixed_padding(kernel_size, dilation)

    def forward(self, input: torch.Tensor):
        return super().forward(F.pad(input, self.input_padding))


class _BasicBlock(nn.Sequential):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        is_relu: bool,
        is_bn: bool,
        is_bias: bool,
        init_type: Optional[str],
    ) -> None:
        layers = [
            _SeparableConv2d(
                input_dim,
                output_dim,
                kernel_size,
                stride,
                dilation,
                is_bn,
                is_bias,
                init_type,
            ),
            nn.BatchNorm2d(output_dim),
        ]
        if is_relu:
            layers.insert(0, nn.ReLU())
        super().__init__(*layers)


class _RedisualBlock(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_layer: int,
        stride: int,
        start_with_relu: bool,
        grow_first: bool,
        dilation: int,
        is_bn: bool,
        is_bias: bool,
        init_type: Optional[str],
    ) -> None:
        super().__init__()
        kwargs = {
            "is_bn": is_bn,
            "is_bias": is_bias,
            "init_type": init_type,
        }
        middle_dim = output_dim if grow_first else input_dim

        blocks = [
            _BasicBlock(
                middle_dim,
                middle_dim,
                3,
                1,
                dilation,
                is_relu=True if i != 0 else grow_first or start_with_relu,
                **kwargs,
            )
            for i in range(n_layer - 1)
        ]
        if grow_first:
            blocks.insert(
                0,
                _BasicBlock(
                    input_dim,
                    middle_dim,
                    3,
                    1,
                    dilation,
                    is_relu=start_with_relu,
                    **kwargs,
                ),
            )
        else:
            blocks.append(
                _BasicBlock(
                    middle_dim,
                    output_dim,
                    3,
                    1,
                    dilation,
                    is_relu=True,
                    **kwargs,
                )
            )
        if stride != 1:
            blocks.append(nn.MaxPool2d(3, stride, 1))
        self.conv_block = nn.Sequential(*blocks)

        self.skip_block = (
            nn.Identity()
            if input_dim == output_dim and stride == 1
            else BasicBlock(
                input_dim,
                output_dim,
                kernel_size=1,
                stride=stride,
                padding=0,
                is_relu=False,
                is_bias=kwargs["is_bias"],
                init_type=init_type,
            )
        )

    def forward(self, input: torch.Tensor):
        return self.conv_block(input) + self.skip_block(input)


class _Xception(IntermediateSequential):
    """Xception optimized for the ImageNet dataset
    Reference: https://github.com/tstandley/Xception-PyTorch/blob/master/xception.py
    """

    first_output_dim = 32
    last_input_dim = 1536
    last_output_dim = 2048

    def __init__(
        self,
        input_dim: int,
        output_stride: Optional[int],
        redisual_block_cfg: Optional[List],
        init_type: Optional[str],
        return_intermediate: bool,
    ) -> None:

        self.replace_stride_with_dilation = (
            _REPLACE_DICT[str(output_stride)]
            if output_stride
            else [False, False, False, False]
        )
        self.redisual_block_cfg = (
            redisual_block_cfg if redisual_block_cfg else _REDISUAL_BLOCK_CFG
        )
        self.kwargs = {"is_bn": False, "is_bias": False, "init_type": init_type}

        self.middle_dim = 64
        self.dilation = 1

        input_block = nn.Sequential(
            BasicBlock(
                input_dim,
                self.first_output_dim,
                stride=2,
                padding=1,
                is_bias=self.kwargs["is_bias"],
                init_type=init_type,
            ),
            BasicBlock(
                self.first_output_dim,
                self.middle_dim,
                padding=1,
                is_relu=False,
                is_bias=self.kwargs["is_bias"],
                init_type=init_type,
            ),
        )
        stages = self._make_stages()
        output_block = nn.Sequential(
            _BasicBlock(
                self.middle_dim,
                self.last_input_dim,
                3,
                1,
                self.dilation,
                is_relu=False,
                **self.kwargs,
            ),
            _BasicBlock(
                self.last_input_dim,
                self.last_output_dim,
                3,
                1,
                self.dilation,
                is_relu=True,
                **self.kwargs,
            ),
            nn.ReLU(),
        )

        super().__init__(
            *[
                input_block,
                *stages,
                output_block,
            ],
            return_intermediate=return_intermediate,
        )

    def _make_stages(self):
        stages = list()
        for cfg in self.redisual_block_cfg:
            if self.replace_stride_with_dilation[cfg[-1]]:
                self.dilation *= cfg[2]
                cfg[2] = 1
            stages.append(
                _RedisualBlock(
                    self.middle_dim, *cfg[:-1], dilation=self.dilation, **self.kwargs
                )
            )
            self.middle_dim = cfg[0]
        return stages

    def load_from(self, weights: OrderedDict):
        model_dict = dict()
        bn_skip = "num_batches_tracked"
        last_index = len(self.redisual_block_cfg) + 1
        for key in self.state_dict().keys():
            weights_key = None
            str_list = key.split(".")
            n_block = int(str_list[0])
            if n_block == 0:
                block_type = "conv" if str_list[-2] == "0" else "bn"
                prefix = f"{block_type}{str(int(str_list[1]) + 1)}"
                if str_list[-1] != bn_skip:
                    weights_key = f"{prefix}.{str_list[-1]}"
            elif n_block == last_index:
                block_type = "conv" if "layers" in str_list else "bn"
                prefix = f"{block_type}{str(int(str_list[1]) + 3)}"
                if block_type == "conv":
                    weights_key = "{}.{}.weight".format(
                        prefix, "conv1" if str_list[3] == "0" else "pointwise"
                    )
                else:
                    if str_list[-1] != bn_skip:
                        weights_key = f"{prefix}.{str_list[-1]}"
            else:
                cfg = self.redisual_block_cfg[n_block - 1]
                prefix = f"block{n_block}"
                if "conv_block" in str_list:
                    prefix = f"{prefix}.rep"
                    # SeparableConv2d
                    if "layers" in str_list:
                        weights_key = "{}.{}.{}.weight".format(
                            prefix,
                            3 * int(str_list[2]) + (0 if not cfg[3] else 1),
                            "conv1" if str_list[4] == "0" else "pointwise",
                        )
                    else:
                        if str_list[-1] != bn_skip:
                            weights_key = "{}.{}.{}".format(
                                prefix,
                                3 * int(str_list[2]) + (1 if not cfg[3] else 2),
                                str_list[-1],
                            )
                else:
                    if str_list[-1] != bn_skip:
                        weights_key = "{}.{}.{}".format(
                            prefix,
                            "skip" if str_list[-2] == 0 else "skipbn",
                            str_list[-1],
                        )
            if weights_key:
                model_dict[key] = weights[weights_key]
        self.state_dict().update(model_dict)
        self.load_state_dict(self.state_dict())


def xception(input_dim: int, pretrained: bool, **kwargs):
    """
    Constructs a Xception architecture from
    `"Xception: Deep Learning with Depthwise Separable Convolutions" <https://arxiv.org/abs/1610.02357>`_.

    Args:
        input_dim  (int):  Number of input channels.
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = _Xception(input_dim, **kwargs)
    if pretrained:
        model.load_from(torch.load(_PRETRAINED_MODELS.get("xception")))
    return model
