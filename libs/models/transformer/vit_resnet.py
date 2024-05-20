import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.utils import np2th


class StdConv2d(nn.Conv2d):
    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


def conv3x3(input_dim, output_dim, stride=1, groups=1, bias=False):
    return StdConv2d(
        input_dim,
        output_dim,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=bias,
        groups=groups,
    )


def conv1x1(input_dim, output_dim, stride=1, bias=False):
    return StdConv2d(
        input_dim,
        output_dim,
        kernel_size=1,
        stride=stride,
        padding=0,
        bias=bias,
    )


class _PreActBottleNeck(nn.Module):
    """Pre-activation (v2) bottleneck block."""

    def __init__(self, input_dim, output_dim=None, middle_dim=None, stride=1):
        super().__init__()
        output_dim = output_dim or input_dim
        middle_dim = middle_dim or output_dim // 4

        self.block1 = nn.Sequential(
            conv1x1(input_dim, middle_dim, bias=False),
            nn.GroupNorm(32, middle_dim, eps=1e-6),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            conv3x3(middle_dim, middle_dim, stride, bias=False),
            nn.GroupNorm(32, middle_dim, eps=1e-6),
            nn.ReLU(),
        )
        self.block3 = nn.Sequential(
            conv1x1(middle_dim, output_dim, bias=False),
            nn.GroupNorm(32, output_dim, eps=1e-6),
        )

        if stride != 1 or input_dim != output_dim:
            # Projection also with pre-activation according to paper.
            self.downsample = nn.Sequential(
                conv1x1(input_dim, output_dim, stride, bias=False),
                nn.GroupNorm(output_dim, output_dim),
            )

    def forward(self, input):
        redisual = input

        if hasattr(self, "downsample"):
            redisual = self.downsample(input)

        x = self.block1(input)
        x = self.block2(x)
        x = self.block3(x)
        output = torch.relu(redisual + x)
        return output

    def load_from(self, weights, n_block, n_unit):
        def load_block(num: int):
            block = getattr(self, f"block{num}")
            block.get_submodule("0").weight.copy_(
                np2th(
                    weights[os.path.join(n_block, n_unit, f"conv{num}/kernel")],
                    conv=True,
                )
            )
            block.get_submodule("1").weight.copy_(
                np2th(weights[os.path.join(n_block, n_unit, f"gn{num}/scale")]).view(-1)
            )
            block.get_submodule("1").bias.copy_(
                np2th(weights[os.path.join(n_block, n_unit, f"gn{num}/bias")]).view(-1)
            )

        load_block(1)
        load_block(2)
        load_block(3)

        if hasattr(self, "downsample"):
            self.downsample.get_submodule("0").weight.copy_(
                np2th(
                    weights[os.path.join(n_block, n_unit, "conv_proj/kernel")],
                    conv=True,
                )
            )
            self.downsample.get_submodule("1").weight.copy_(
                np2th(weights[os.path.join(n_block, n_unit, "gn_proj/scale")]).view(-1)
            )
            self.downsample.get_submodule("1").bias.copy_(
                np2th(weights[os.path.join(n_block, n_unit, "gn_proj/bias")]).view(-1)
            )


class ResNetV2(nn.Module):
    def __init__(self, block_units: list, width_factor: int):
        """Implementation of Pre-activation (v2) ResNet mode."""
        super().__init__()
        width = 64 * width_factor
        self.width = width

        def get_block_dict(factor: int, num: int):
            return OrderedDict(
                [
                    (
                        "unit1",
                        _PreActBottleNeck(
                            input_dim=width * (1 if num == 0 else 2) * factor,
                            output_dim=width * 4 * factor,
                            middle_dim=width * factor,
                            stride=1 if num == 0 else 2,
                        ),
                    )
                ]
                + [
                    (
                        f"unit{i:d}",
                        _PreActBottleNeck(
                            input_dim=width * 4 * factor,
                            output_dim=width * 4 * factor,
                            middle_dim=width * factor,
                        ),
                    )
                    for i in range(2, block_units[num] + 1)
                ]
            )

        self.root = nn.Sequential(
            StdConv2d(3, width, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(32, width, eps=1e-6),
            nn.ReLU(),
        )
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.body = nn.Sequential(
            OrderedDict(
                [
                    (f"block{i+1}", nn.Sequential(get_block_dict(2**i, i)))
                    for i in range(len(block_units))
                ]
            )
        )

    def forward(self, input: torch.Tensor):
        features = list()
        batch, _, height, _ = input.size()
        x = self.root(input)
        features.append(x)
        x = self.pool(x)
        for i in range(len(self.body) - 1):
            x = self.body[i](x)
            right_size = int(height / 4 / (i + 1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert pad < 3 and pad > 0, "x {} should {}".format(
                    x.size(), right_size
                )
                feature = torch.zeros(
                    (batch, x.size()[1], right_size, right_size), device=x.device
                )
                feature[:, :, 0 : x.size()[2], 0 : x.size()[3]] = x[:]
            else:
                feature = x
            features.append(feature)
        output = self.body[-1](x)
        return output, *features[::-1]
