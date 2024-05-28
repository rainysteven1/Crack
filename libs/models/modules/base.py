import math
from typing import Optional, Union

import torch
import torch.nn as nn

from ...utils.init import InitModule


class SqueezeExciteBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        middle_dim: Optional[int] = None,
        ratio: int = 8,
        activation: nn.Module = nn.ReLU,
        scale_activation: nn.Module = nn.Sigmoid,
        is_bias: bool = False,
    ) -> None:
        super().__init__()
        self.is_linear = middle_dim is None
        middle_dim = middle_dim or math.ceil(input_dim / ratio)

        if self.is_linear:
            self.pool = nn.AdaptiveAvgPool2d(1)
            layers = [
                nn.Linear(input_dim, middle_dim, bias=is_bias),
                nn.Linear(middle_dim, input_dim, bias=is_bias),
            ]
        else:
            self.pool = None
            layers = [
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(input_dim, middle_dim, 1, bias=is_bias),
                nn.Conv2d(middle_dim, input_dim, 1, bias=is_bias),
            ]

        layers.insert(-1, activation())
        layers.append(scale_activation())
        self.layers = nn.Sequential(*layers)

    def forward(self, input: torch.Tensor):
        if not self.is_linear:
            return input * self.layers(input)
        n, c, _, _ = input.shape
        x = self.pool(input).view(n, c)
        x = self.layers(x).view(n, c, 1, 1)
        return input * x


class BasicBlock(InitModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Union[int, str] = 1,
        dilation: int = 1,
        groups: int = 1,
        is_bn: bool = True,
        is_relu: bool = True,
        is_bias: bool = True,
        reversed: bool = False,
        norm_type: nn.Module = nn.BatchNorm2d,
        relu_type: nn.Module = nn.ReLU,
        init_type: Optional[str] = None,
    ):
        super().__init__(init_type)
        layer_list = [
            nn.Conv2d(
                input_dim,
                output_dim,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                is_bias,
            ),
        ]
        if reversed:
            layer_list.insert(0, norm_type(input_dim))
            layer_list.insert(1, relu_type())
        else:
            if is_relu:
                layer_list.append(relu_type())
            if is_bn:
                layer_list.insert(1, norm_type(output_dim))
        self.layers = nn.Sequential(*layer_list)

        if self.init:
            self._initialize_weights()

    def forward(self, input: torch.Tensor):
        return self.layers(input)

    def _initialize_weights(self):
        self.layers.apply(lambda m: self.init(m))


class IntermediateSequential(nn.Sequential):
    def __init__(self, *args, return_intermediate: bool = False):
        super().__init__(*args)
        self.return_intermediate = return_intermediate

    def forward(self, input: torch.Tensor):
        if not self.return_intermediate:
            return super().forward(input)

        intermediate_outputs = list()
        x = input
        for module in self.children():
            x = module(x)
            intermediate_outputs.append(x)
        return intermediate_outputs


class OutputBlock(nn.Sequential):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int = 1,
        padding: int = 0,
        is_bn: bool = False,
        is_bias: bool = True,
        init_type: Optional[str] = None,
    ) -> None:
        super().__init__(
            BasicBlock(
                input_dim,
                output_dim,
                kernel_size,
                padding=padding,
                is_bn=is_bn,
                is_relu=False,
                is_bias=is_bias,
                init_type=init_type,
            ),
            nn.Sigmoid(),
        )
