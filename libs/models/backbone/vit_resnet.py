from typing import Dict, List, Optional, OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

__all__ = ["ResNet"]

_STAGE_CFG = {
    "n_layers": [3, 4, 9],
    "strides": [1, 2, 2],
}


class _StdConv2d(nn.Conv2d):
    def forward(self, input: torch.Tensor):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(
            input, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


def _conv1x1(
    input_dim: int,
    output_dim: int,
    n_groups: int = 32,
    stride: int = 1,
    is_relu: bool = True,
):
    layers = [
        _StdConv2d(
            input_dim, output_dim, kernel_size=1, stride=stride, padding=0, bias=False
        ),
        nn.GroupNorm(n_groups, output_dim, eps=1e-6),
    ]
    if is_relu:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def _conv3x3(input_dim: int, output_dim: int, stride: int = 1):
    return nn.Sequential(
        _StdConv2d(
            input_dim,
            output_dim,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        ),
        nn.GroupNorm(32, output_dim, eps=1e-6),
        nn.ReLU(),
    )


class _BottleNeck(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: Optional[int],
        middle_dim: Optional[int],
        stride: int = 1,
    ) -> None:
        super().__init__()
        output_dim = output_dim or input_dim
        middle_dim = middle_dim or output_dim // 4

        self.conv_block = nn.Sequential(
            _conv1x1(input_dim, middle_dim),
            _conv3x3(middle_dim, middle_dim, stride),
            _conv1x1(middle_dim, output_dim, is_relu=False),
        )
        self.skip_block = (
            nn.Identity()
            if stride == 1 and input_dim == output_dim
            else _conv1x1(input_dim, output_dim, output_dim, stride, is_relu=False)
        )

    def forward(self, input: torch.Tensor):
        return F.relu(self.conv_block(input) + self.skip_block(input))


class _Stem(nn.Sequential):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__(
            _StdConv2d(
                input_dim, output_dim, kernel_size=7, stride=2, padding=3, bias=False
            ),
            nn.GroupNorm(32, output_dim, eps=1e-6),
            nn.ReLU(),
        )


class ResNet(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    dim = 64

    def __init__(
        self,
        input_dim: int,
        ratio: float,
        stage_cfg: Optional[Dict[str, List[int]]],
    ) -> None:
        super().__init__()
        self.width = int(self.dim * ratio)
        self.stage_cfg = stage_cfg if stage_cfg else _STAGE_CFG
        self.n_stages = len(self.stage_cfg["n_layers"])

        self.input_block = _Stem(input_dim, self.width)
        self.stages = nn.ModuleList(self._make_stages())

    def _make_stages(self) -> List[nn.Sequential]:
        stages = list()
        for idx in range(self.n_stages):
            stride = self.stage_cfg["strides"][idx]
            n_layers = self.stage_cfg["n_layers"][idx]
            layers = [
                _BottleNeck(
                    self.width * 2 ** (idx if idx == 0 else idx + 1),
                    self.width * 2 ** (idx + 2),
                    self.width * 2**idx,
                    stride,
                )
            ] + [
                _BottleNeck(
                    self.width * 2 ** (idx + 2),
                    self.width * 2 ** (idx + 2),
                    self.width * 2**idx,
                )
            ] * (
                n_layers - 1
            )
            stages.append(nn.Sequential(*layers))
        return stages

    def forward(self, input: torch.Tensor):
        features = list()
        x = self.input_block(input)
        features.append(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        for i, stage in enumerate(self.stages[:-1]):
            x = stage(x)
            right_size = input.shape[2] // (4 * (i + 1))
            if x.shape[2] != right_size:
                pad = right_size - x.shape[2]
                assert 0 < pad < 3, f"x {x.shape} should be {right_size}"
                x = F.pad(x, (0, pad, 0, pad))
            features.append(x)

        x = self.stages[-1](x)
        return x, features[::-1]

    def load_from(self, weights: OrderedDict, weights_keys: DictConfig):
        state_dict = dict()
        for key in self.state_dict().keys():
            str_list = key.split(".")
            if "input_block" in str_list:
                weights_key = weights_keys["input_block"].format(
                    "conv" if str_list[-2] == "0" else "norm", str_list[-1]
                )
            elif "stages" in str_list:
                root = weights_keys["stages"]["root"].format(
                    str_list[1], str_list[2], str_list[-1]
                )
                if "conv_block" in str_list:
                    num_conv = int(str_list[-3])
                    weights_key = root.format(
                        ("conv" if str_list[-2] == "0" else "norm") + str(num_conv + 1)
                    )
                elif "skip_block" in str_list:
                    root = root.format(weights_keys["stages"]["skip_block"])
                    weights_key = root.format("conv" if str_list[-2] == "0" else "norm")
            state_dict[key] = weights[weights_key]
        self.load_state_dict(state_dict)
