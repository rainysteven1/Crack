from typing import Dict, List, Optional, OrderedDict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.base import BasicBlock, IntermediateSequential, SqueezeExciteBlock
from ._utils import fixed_padding

__all__ = ["mobilenetv2", "mobilenetv3", "mobilenetv3_large", "mobilenetv3_small"]

PRETRAINED_MODELS = {
    "mobilenetv2": "resources/checkpoints/mobilenet_v2-b0353104.pth",
    "mobilenetv3_large": "resources/checkpoints/mobilenet_v3_large-8738ca79.pth",
    "mobilenetv3_small": "resources/checkpoints/mobilenet_v3_small-047dcff4.pth",
}

KERNEL_SIZES = [3, 5]
STRIDES = [1, 2]

MOBILENET_V2_CFG = [
    # t, c, n, s
    [1, 16, 1, 1],
    [6, 24, 2, 2],
    [6, 32, 3, 2],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1],
]

MOBILENET_V3_CFG_LARGE = [
    # k, exp, c, se, nl, s,
    [3, 16, 16, False, "RE", 1],
    [3, 64, 24, False, "RE", 2],
    [3, 72, 24, False, "RE", 1],
    [5, 72, 40, True, "RE", 2],
    [5, 120, 40, True, "RE", 1],
    [5, 120, 40, True, "RE", 1],
    [3, 240, 80, False, "HS", 2],
    [3, 200, 80, False, "HS", 1],
    [3, 184, 80, False, "HS", 1],
    [3, 184, 80, False, "HS", 1],
    [3, 480, 112, True, "HS", 1],
    [3, 672, 112, True, "HS", 1],
    [5, 672, 160, True, "HS", 2],
    [5, 960, 160, True, "HS", 1],
    [5, 960, 160, True, "HS", 1],
]

MOBILENET_V3_CFG_SMALL = [
    # k, exp, c, se, nl, s,
    [3, 16, 16, True, "RE", 2],
    [3, 72, 24, False, "RE", 2],
    [3, 88, 24, False, "RE", 1],
    [5, 96, 40, True, "HS", 2],
    [5, 240, 40, True, "HS", 1],
    [5, 240, 40, True, "HS", 1],
    [5, 120, 48, True, "HS", 1],
    [5, 144, 48, True, "HS", 1],
    [5, 288, 96, True, "HS", 2],
    [5, 576, 96, True, "HS", 1],
    [5, 576, 96, True, "HS", 1],
]


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """This function is taken from the original tf repo.

    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class _InvertedResidual(nn.Sequential):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        middle_dim: Optional[int],
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        ratio: Optional[int],
        is_redisual: bool,
        is_se: bool,
        relu_type: nn.Module,
        init_type: Optional[str],
    ) -> None:
        assert kernel_size in KERNEL_SIZES
        assert stride in STRIDES

        middle_dim = middle_dim or int(round(input_dim * ratio))
        self.is_redisual = is_redisual
        self.padding = padding
        self.is_cn = stride > 1

        conv_list = [
            # Depthwise layer
            BasicBlock(
                middle_dim,
                middle_dim,
                kernel_size,
                stride,
                padding,
                dilation,
                middle_dim,
                is_bias=False,
                relu_type=relu_type,
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
        if not ratio or ratio != 1:
            # Expansion Layer
            conv_list.insert(
                0,
                BasicBlock(
                    input_dim,
                    middle_dim,
                    kernel_size=1,
                    padding=0,
                    is_bias=False,
                    relu_type=relu_type,
                    init_type=init_type,
                ),
            )
        if is_se:
            se_dim = _make_divisible(middle_dim // 4, 8)
            conv_list.insert(
                -1,
                SqueezeExciteBlock(
                    middle_dim, se_dim, 4, nn.ReLU, nn.Hardswish, is_bias=True
                ),
            )

        super().__init__(*conv_list)
        self.input_padding = fixed_padding(3, dilation)

    def forward(self, input: torch.Tensor):
        x_pad = input if self.padding else F.pad(input, self.input_padding)
        return (
            super().forward(x_pad)
            if not self.is_redisual
            else input + super().forward(x_pad)
        )


class _MoblieNetHead(IntermediateSequential):

    def __init__(
        self,
        input_dim: int,
        output_stride: Optional[int],
        width_mult: float,
        inverted_redisual_cfg: Optional[List],
        init_type: Optional[str],
        return_intermediate: bool,
        round_nearest: int = 8,
    ) -> None:
        assert isinstance(inverted_redisual_cfg, list)
        self.output_stride = output_stride
        self.width_mult = width_mult
        self.inverted_redisual_cfg = inverted_redisual_cfg
        self.init_type = init_type
        self.is_dilation = output_stride is not None
        self.return_intermediate = return_intermediate
        self.round_nearest = round_nearest

        first_output_dim = _make_divisible(
            self.first_output_dim * width_mult, round_nearest
        )
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
        pass

    def load_from(self, weights: OrderedDict):
        pass

    def get_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                yield m.weight

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class _MobileNetV2(_MoblieNetHead):
    """
    Reference: https://github.com/VainF/DeepLabV3Plus-Pytorch/blob/master/network/backbone/mobilenetv2.py
    """

    first_output_dim = 32
    first_block_relu_type = nn.ReLU6

    def __init__(
        self,
        input_dim: int,
        output_stride: Optional[int],
        width_mult: float,
        inverted_redisual_cfg: Optional[List],
        init_type: Optional[str],
        return_intermediate: bool,
    ) -> None:
        inverted_redisual_cfg = (
            inverted_redisual_cfg if inverted_redisual_cfg else MOBILENET_V2_CFG
        )

        super().__init__(
            input_dim,
            output_stride,
            width_mult,
            inverted_redisual_cfg,
            init_type,
            return_intermediate,
        )

    def _make_stages(self):
        rate = 1
        stages = list()
        # stages
        for t, c, n, s in self.inverted_redisual_cfg:
            # use Dilated Convolution
            if not self.is_dilation:
                stride = s
                dilation = 1
            else:
                if self.current_stride == self.output_stride:
                    stride = 1
                    dilation = rate
                    rate *= stride
                else:
                    stride = s
                    dilation = 1
                    self.current_stride *= s
            output_dim = _make_divisible(c * self.width_mult, self.round_nearest)
            stage = list()
            # layers
            for i in range(n):
                stage.append(
                    _InvertedResidual(
                        self.middle_dim,
                        output_dim,
                        None,
                        kernel_size=3,
                        stride=1 if i != 0 else stride,
                        padding=0,
                        dilation=dilation,
                        ratio=t,
                        is_redisual=stride == 1 and self.middle_dim == output_dim,
                        is_se=False,
                        relu_type=nn.ReLU6,
                        init_type=self.init_type,
                    )
                )
                self.middle_dim = output_dim
            stages.append(nn.Sequential(*stage))
        return stages

    def load_from(self, weights: OrderedDict):
        model_dict = dict()
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


class _MobileNetV3(_MoblieNetHead):

    first_output_dim = 16
    first_block_relu_type = nn.Hardswish

    def __init__(
        self,
        input_dim: int,
        output_stride: Optional[int],
        width_mult: float,
        inverted_redisual_cfg: Optional[Union[str, List]],
        init_type: Optional[str],
        return_intermediate: bool,
    ) -> None:
        self.mode = None
        if not inverted_redisual_cfg:
            inverted_redisual_cfg = MOBILENET_V3_CFG_LARGE
        elif isinstance(inverted_redisual_cfg, Dict):
            inverted_redisual_cfg = inverted_redisual_cfg
        elif isinstance(inverted_redisual_cfg, str):
            self.mode = inverted_redisual_cfg
            inverted_redisual_cfg = (
                MOBILENET_V3_CFG_LARGE
                if inverted_redisual_cfg == "large"
                else MOBILENET_V3_CFG_SMALL
            )

        super().__init__(
            input_dim,
            output_stride,
            width_mult,
            inverted_redisual_cfg,
            init_type,
            return_intermediate,
        )

    def _make_stages(self):
        rate = 1
        stages = list()
        # stages
        for idx, (k, exp, c, se, nl, s) in enumerate(self.inverted_redisual_cfg):
            # use Dilated Convolution
            if not self.is_dilation:
                stride = s
                dilation = 1
            else:
                if self.current_stride == self.output_stride:
                    stride = 1
                    dilation = rate
                    rate *= stride
                else:
                    stride = s
                    dilation = 1
                    self.current_stride *= s
            output_dim = _make_divisible(c * self.width_mult, self.round_nearest)
            stages.append(
                _InvertedResidual(
                    self.middle_dim,
                    output_dim,
                    exp,
                    kernel_size=k,
                    stride=stride,
                    padding=(k - 1) // 2,
                    dilation=dilation,
                    ratio=1 if idx == 0 else None,
                    is_redisual=s == 1 and self.middle_dim == output_dim,
                    is_se=se,
                    relu_type=nn.ReLU6 if nl == "RE" else nn.Hardswish,
                    init_type=self.init_type,
                )
            )
            self.middle_dim = output_dim
        return stages

    def load_from(self, weights: OrderedDict):
        se_list = [
            idx + 1 for idx, cfg in enumerate(self.inverted_redisual_cfg) if cfg[3]
        ]
        model_dict = dict()
        prefix_0 = "features.0{}"
        prefix_block = "features.{}.block.{}"
        for key in self.state_dict().keys():
            str_list = key.split(".")
            n_block = int(str_list[0])
            if n_block == 0:
                model_dict[key] = weights[prefix_0.format(key.split("layers")[-1])]
            else:
                n_layer = int(str_list[1])
                suffix = str_list[-2:]
                se_layer = 2
                if self.mode == "small" and n_block == 1:
                    se_layer = 1
                if n_block in se_list and n_layer == se_layer:
                    suffix = ["fc1" if str_list[-2] == "1" else "fc2", str_list[-1]]
                model_dict[key] = weights[
                    prefix_block.format(n_block, ".".join([str(n_layer)] + suffix))
                ]
        self.state_dict().update(model_dict)
        self.load_state_dict(self.state_dict())


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


def mobilenetv3(input_dim: int, **kwargs):
    """
    Constructs a MobileNetV3 architecture from
    `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_.

    Args:
        input_dim  (int):  Number of input channels.
    """
    return _MobileNetV3(input_dim, **kwargs)


def mobilenetv3_large(input_dim: int, pretrained: bool, **kwargs):
    """
    Constructs a MobileNetV3 architecture from
    `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_.

    Args:
        input_dim  (int):  Number of input channels.
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    assert kwargs["inverted_redisual_cfg"] == "large"
    model = mobilenetv3(input_dim, **kwargs)
    if pretrained:
        model.load_from(torch.load(PRETRAINED_MODELS.get("mobilenetv3_large")))
    return model


def mobilenetv3_small(input_dim: int, pretrained: bool, **kwargs):
    """
    Constructs a MobileNetV3 architecture from
    `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_.

    Args:
        input_dim  (int):  Number of input channels.
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    assert kwargs["inverted_redisual_cfg"] == "small"
    model = mobilenetv3(input_dim, **kwargs)
    if pretrained:
        model.load_from(torch.load(PRETRAINED_MODELS.get("mobilenetv3_small")))
    return model
