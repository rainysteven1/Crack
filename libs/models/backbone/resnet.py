import copy
from typing import Dict, List, Optional, OrderedDict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.attention import BoT_MHSA
from ..modules.base import BasicBlock, IntermediateSequential

__all__ = ["resnet34", "resnet50", "resnet101"]


PRETRAINED_MODELS = {
    "resnet34": "resources/checkpoints/resnet34-333f7ec4.pth",
    "resnet50": "resources/checkpoints/resnet50-19c8e357.pth",
    "resnet101": "resources/checkpoints/resnet101-5d3b4d8f.pth",
}

STAGE_CFG = {
    "n_layers": [3, 4, 6, 3],
    "strides": [1, 2, 2, 2],
    "dilations": [1, 1, 1, 1],
}


class RedisualBlock(nn.Module):

    expansion = 1

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        stride: int = 1,
        skip_kernel_size: int = 1,
        skip_padding: int = 0,
        is_identity: bool = False,
        is_bn: bool = True,
        is_bias: bool = False,
        reversed: bool = False,
        init_type: Optional[str] = None,
        zero_init: bool = False,
    ) -> None:
        super().__init__()

        self.conv_block = nn.Sequential(
            BasicBlock(
                input_dim,
                output_dim,
                stride=stride,
                is_bias=is_bias,
                reversed=reversed,
                init_type=init_type,
            ),
            BasicBlock(
                output_dim,
                output_dim,
                is_bn=is_bn,
                is_relu=False,
                is_bias=is_bias,
                reversed=reversed,
                init_type="constant" if zero_init else init_type,
            ),
        )
        self.skip_block = (
            nn.Identity()
            if is_identity
            else BasicBlock(
                input_dim,
                output_dim * self.expansion,
                kernel_size=skip_kernel_size,
                stride=stride,
                padding=skip_padding,
                is_bn=is_bn,
                is_relu=False,
                is_bias=is_bias,
                init_type=init_type,
            )
        )

    def forward(self, input: torch.Tensor):
        return F.relu(self.conv_block(input) + self.skip_block(input))


class BottleNeck(RedisualBlock):

    expansion = 4

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        middle_dim: int = None,
        stride: int = 1,
        dilation: int = 1,
        is_identity: bool = False,
        reversed: bool = False,
        MHSA_CFG: Optional[Dict] = None,
        init_type: Optional[str] = None,
        zero_init: bool = False,
    ) -> None:
        super().__init__(input_dim, output_dim, stride, 1, 0, is_identity)
        middle_dim = middle_dim or output_dim
        is_bias = not reversed

        if not MHSA_CFG:
            conv = BasicBlock(
                middle_dim,
                middle_dim,
                stride=stride,
                padding=dilation,
                dilation=dilation,
                is_bias=is_bias,
                reversed=reversed,
                init_type=init_type,
            )
        else:
            module_list = [BoT_MHSA(middle_dim, **MHSA_CFG)]
            if not is_identity:
                module_list.append(nn.AvgPool2d(kernel_size=2, stride=2))
            conv = nn.Sequential(*module_list)
        self.conv_block = nn.Sequential(
            BasicBlock(
                input_dim,
                middle_dim,
                kernel_size=1,
                padding=0,
                is_bias=is_bias,
                reversed=reversed,
                init_type=init_type,
            ),
            conv,
            # Linear bottlenecks：Elwise + with No ReLU at the end of a bottleneck
            BasicBlock(
                middle_dim,
                output_dim * self.expansion,
                kernel_size=1,
                padding=0,
                is_relu=False,
                is_bias=is_bias,
                reversed=reversed,
                init_type="constant" if zero_init else init_type,
            ),
        )


class _Stem(nn.Sequential):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__(
            BasicBlock(
                input_dim,
                output_dim,
                kernel_size=7,
                stride=2,
                padding=3,
                is_bias=False,
            ),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )


class _ResNet(IntermediateSequential):
    """
    Reference: https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
    """

    dims = [64, 128, 256, 512]

    def __init__(
        self,
        block: RedisualBlock | BottleNeck,
        input_dim: int,
        stage_cfg: Dict[str, List[int]],
        init_type: Optional[str] = None,
        zero_init: bool = False,
        multi_grids: Optional[List[int]] = None,
        MHSA_cfg: Optional[dict] = None,
        return_intermediate: bool = False,
    ) -> None:

        self.block_type = block
        self.stage_cfg = stage_cfg
        self.n_stages = len(self.stage_cfg["n_layers"])
        self.temp_dim = self.dims[0]
        self.init_type = init_type
        self.zero_init = zero_init
        self.multi_grids = multi_grids
        self.MHSA_cfg = MHSA_cfg

        super().__init__(
            *[
                _Stem(input_dim, self.temp_dim),
                *self._make_stages(),
            ],
            return_intermediate=return_intermediate,
        )

    def _make_stages(self) -> List[nn.Sequential]:
        stages = list()
        for idx in range(self.n_stages):
            layers = list()
            n_layers = self.stage_cfg["n_layers"][idx]
            strides = [self.stage_cfg["strides"][idx]] + [1] * (n_layers - 1)

            multi_grids = (
                self.multi_grids
                if self.multi_grids and idx == self.n_stages - 1
                else [1 for _ in range(n_layers)]
            )

            # layers
            for stride_idx, stride in enumerate(strides):
                args = [self.temp_dim, self.dims[idx]]
                kwargs = {
                    "stride": stride,
                    "is_identity": stride_idx != 0,
                    "init_type": self.init_type,
                    "zero_init": self.zero_init and stride_idx == 0,
                }
                if self.block_type is BottleNeck:
                    kwargs["dilation"] = (
                        self.stage_cfg["dilations"][idx] * multi_grids[stride_idx]
                    )
                    if self.MHSA_cfg and idx == self.n_stages - 1:
                        kwargs["MHSA_cfg"] = self.MHSA_cfg
                        if stride_idx == 0:
                            self.MHSA_cfg["img_size"] = (
                                self.MHSA_cfg["img_size"] // self.block_type.expansion
                            )
                layers.append(self.block_type(*args, **kwargs))
                self.temp_dim = self.dims[idx] * self.block_type.expansion

            stages.append(nn.Sequential(*layers))
        return stages

    def load_from(self, weights: OrderedDict):
        modules = list(self.children())
        input_block = modules[0]
        conv_idx = "conv1"
        bn_idx = "bn1"
        _load_single_block_weight(
            weights, conv_idx, bn_idx, list(input_block.children())[0]
        )
        stages = modules[1:]
        for stage_idx, stage in enumerate(stages):
            blocks = list(stage.children())
            for block_idx, block in enumerate(blocks):
                prefix = f"layer{stage_idx+1}.{block_idx}"
                conv_block = list(block.named_children())[0][1]
                skip_block = list(block.named_children())[1][1]
                # skip_block
                if block_idx == 0:
                    conv_idx = f"{prefix}.downsample.0"
                    bn_idx = f"{prefix}.downsample.1"
                    if stage_idx > 0 or isinstance(self, BottleNeck):
                        _load_single_block_weight(weights, conv_idx, bn_idx, skip_block)
                # conv_block
                for idx, single_block in enumerate(list(conv_block.children())):
                    if self.block_type is RedisualBlock or (
                        self.block_type is BottleNeck
                        and (not self.MHSA_cfg or idx != 1)
                    ):
                        conv_idx = f"{prefix}.conv{idx+1}"
                        bn_idx = f"{prefix}.bn{idx+1}"
                        _load_single_block_weight(
                            weights, conv_idx, bn_idx, single_block
                        )

    def get_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                yield m.weight

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


def _load_single_block_weight(
    weights: OrderedDict, conv_idx: str, bn_idx: str, single_block: BasicBlock
):
    single_block_children = list(list(single_block.children())[0].children())
    conv = single_block_children[0]
    bn = single_block_children[1]
    conv.weight.data = weights.get(f"{conv_idx}.weight")
    bn_state_dict = {
        "weight": weights.get(f"{bn_idx}.weight"),
        "bias": weights.get(f"{bn_idx}.bias"),
        "running_mean": weights.get(f"{bn_idx}.running_mean"),
        "running_var": weights.get(f"{bn_idx}.running_var"),
    }
    bn.load_state_dict(bn_state_dict)


def _resnet(
    arch: str,
    block: Union[BottleNeck, RedisualBlock],
    input_dim: int,
    stage_cfg: Dict[str, List[int]],
    pretrained: bool,
    **kwargs,
):
    model = _ResNet(block, input_dim, stage_cfg, **kwargs)
    if pretrained:
        model.load_from(torch.load(PRETRAINED_MODELS.get(arch)))
    return model


def _get_stage_cfg(n_layers: List[int], **kwargs):
    stage_cfg = copy.deepcopy(STAGE_CFG)
    stage_cfg["n_layers"] = n_layers
    if kwargs["strides"]:
        stage_cfg["strides"] = kwargs.pop("strides")
    if kwargs["dilations"]:
        stage_cfg["dilations"] = kwargs.pop("dilations")
    return stage_cfg, kwargs


def resnet34(input_dim: int, pretrained: bool, **kwargs):
    return _resnet(
        "resnet34", RedisualBlock, input_dim, STAGE_CFG, pretrained, **kwargs
    )


def resnet50(input_dim: int, pretrained: bool, **kwargs):
    stage_cfg, kwargs = _get_stage_cfg([3, 4, 6, 3], **kwargs)
    return _resnet("resnet50", BottleNeck, input_dim, stage_cfg, pretrained, **kwargs)


def resnet101(input_dim: int, pretrained: bool, **kwargs):
    stage_cfg, kwargs = _get_stage_cfg([3, 4, 23, 3], **kwargs)
    return _resnet("resnet101", BottleNeck, input_dim, stage_cfg, pretrained, **kwargs)
