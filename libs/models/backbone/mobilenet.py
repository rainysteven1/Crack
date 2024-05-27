from typing import Dict, Optional, OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.base import BasicBlock, IntermediateSequential

__all__ = ["mobilenetv2"]

PRETRAINED_MODELS = {
    "mobilenetv2": "resources/checkpoints/mobilenet_v2-b0353104.pth",
}

STRIDES = [1, 2]

INVERTED_REDISUAL_CFG = [
    # t, c, n, s
    [1, 16, 1, 1],
    [6, 24, 2, 2],
    [6, 32, 3, 2],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1],
]


def _fixed_padding(kernel_size: int, dilation: int):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    return (pad_beg, pad_end, pad_beg, pad_end)


class _InvertedResidual(nn.Sequential):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        stride: int,
        dilation: int,
        ratio: int,
        is_redisual: bool,
        init_type: Optional[str],
    ) -> None:
        assert stride in STRIDES
        middle_dim = int(round(input_dim * ratio))
        self.stride = stride
        self.is_redisual = is_redisual

        conv_list = [
            # Depthwise layer
            BasicBlock(
                middle_dim,
                middle_dim,
                stride=stride,
                padding=0,
                dilation=dilation,
                groups=middle_dim,
                is_bias=False,
                relu_type=nn.ReLU6,
                init_type=init_type,
            ),
            # Projection Layer
            BasicBlock(
                middle_dim,
                output_dim,
                kernel_size=1,
                padding=0,
                is_relu=False,
                is_bias=False,
                init_type=init_type,
            ),
        ]
        if ratio != 1:
            # Expansion Layer
            conv_list.insert(
                0,
                BasicBlock(
                    input_dim,
                    middle_dim,
                    kernel_size=1,
                    padding=0,
                    is_bias=False,
                    relu_type=nn.ReLU6,
                    init_type=init_type,
                ),
            )

        super().__init__(*conv_list)
        self.input_padding = _fixed_padding(3, dilation)

    def forward(self, input: torch.Tensor):
        x_pad = F.pad(input, self.input_padding)
        return (
            super().forward(x_pad)
            if not self.is_redisual
            else input + super().forward(x_pad)
        )


class _MobileNetV2(IntermediateSequential):

    first_output_dim = 32

    def __init__(
        self,
        input_dim: int,
        output_stride: int,
        width_mult: float,
        inverted_redisual_cfg: Optional[Dict] = None,
        init_type: Optional[str] = None,
        return_intermediate: bool = False,
    ) -> None:
        self.output_stride = output_stride
        self.width_mult = width_mult
        self.inverted_redisual_cfg = (
            inverted_redisual_cfg if inverted_redisual_cfg else INVERTED_REDISUAL_CFG
        )
        self.init_type = init_type

        first_output_dim = int(self.first_output_dim * width_mult)
        self.current_stride = 2
        self.middle_dim = first_output_dim

        super().__init__(
            *[
                BasicBlock(
                    input_dim,
                    self.middle_dim,
                    stride=self.current_stride,
                    is_bias=False,
                    relu_type=nn.ReLU6,
                    init_type=init_type,
                ),
                *self._make_stages(),
            ],
            return_intermediate=return_intermediate,
        )

    def _make_stages(self):
        rate = 1
        dilation = 1
        stages = list()
        # stages
        for t, c, n, s in self.inverted_redisual_cfg:
            # use Dilated Convolution
            rate = dilation
            if self.current_stride == self.output_stride:
                stride = 1
                dilation *= s
            else:
                stride = s
                self.current_stride *= s
            output_dim = int(c * self.width_mult)
            stage = list()
            # layers
            for i in range(n):
                print(dilation)
                stage.append(
                    _InvertedResidual(
                        self.middle_dim,
                        output_dim,
                        stride=1 if i != 0 else stride,
                        dilation=rate if i == 0 else dilation,
                        ratio=t,
                        is_redisual=stride == 1 and self.middle_dim == output_dim,
                        init_type=self.init_type,
                    )
                )
                self.middle_dim = output_dim
            stages.append(nn.Sequential(*stage))
        return stages

    def load_from(self, weights: OrderedDict):
        model_dict = {}
        state_dict = self.state_dict()
        prefix_0 = "features.0{}"
        prefix_conv = "features.{}.conv.{}"
        for key in state_dict.keys():
            if key.startswith("0"):
                model_dict[key] = weights[prefix_0.format(key.split("layers")[-1])]
            else:
                str_list = key.split(".")
                n_stage = int(str_list[0]) - 1
                n_feature = (
                    1
                    + sum([item[-2] for item in self.inverted_redisual_cfg[:n_stage]])
                    + int(str_list[1])
                )
                flag = 1 if self.inverted_redisual_cfg[n_stage][0] == 1 else 2
                temp = int(str_list[2])
                index = -2 if temp < flag else -1
                temp = temp if temp < flag else temp + int(str_list[-2])
                model_dict[key] = weights[
                    prefix_conv.format(
                        n_feature, ".".join([str(temp)] + str_list[index:])
                    )
                ]
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def get_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                yield m.weight

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


def mobilenetv2(input_dim: int, pretrained: bool, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        input_dim  (int):  Number of input channels.
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = _MobileNetV2(input_dim, **kwargs)
    if pretrained:
        model.load_from(torch.load(PRETRAINED_MODELS.get("mobilenetv2")))
    return model
