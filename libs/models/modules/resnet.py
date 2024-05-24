from collections import OrderedDict
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import BoT_MHSA
from .base import BasicBlock

PRETRAINED_MODELS = {
    "resnet34": "resources/checkpoints/resnet34-333f7ec4.pth",
    "resnet50": "resources/checkpoints/resnet50-19c8e357.pth",
    "resnet101": "resources/checkpoints/resnet101-5d3b4d8f.pth",
}


class RedisualBlock(nn.Module):

    expansion = 1

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        stride: int = 1,
        skip_kernel_size: int = 1,
        skip_padding: str | int = "same",
        is_identity: bool = False,
        is_bn: bool = True,
        is_bias: bool = False,
        reversed: bool = False,
        init_type: Optional[str] = None,
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
                init_type=init_type,
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
        skip_padding: str | int = "same",
        is_identity: bool = False,
        is_bias: bool = False,
        reversed: bool = False,
        img_size: int = 256,
        is_MHSA: bool = False,
        n_heads: int = 4,
        init_type: Optional[str] = None,
    ) -> None:
        super().__init__(input_dim, output_dim, stride, 1, skip_padding, is_identity)
        middle_dim = middle_dim or output_dim

        if not is_MHSA:
            conv = BasicBlock(
                middle_dim,
                middle_dim,
                stride=stride,
                is_bias=is_bias,
                reversed=reversed,
                init_type=init_type,
            )
        else:
            module_list = [BoT_MHSA(middle_dim, img_size, n_heads)]
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
            BasicBlock(
                middle_dim,
                output_dim * self.expansion,
                kernel_size=1,
                padding=0,
                is_relu=False,
                is_bias=is_bias,
                reversed=reversed,
                init_type=init_type,
            ),
        )


class _InputBlock(nn.Sequential):
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


class _ResNet(nn.Sequential):
    """
    reference: https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
    """

    layer_configurations = [(64, 1), (128, 2), (256, 2), (512, 2)]

    def __init__(
        self,
        input_dim: int,
        num_blocks: list[int],
        block: RedisualBlock | BottleNeck,
        img_size: int,
        is_MHSA: bool,
        n_heads: int,
    ) -> None:
        assert len(num_blocks) == len(self.layer_configurations)

        self.num_blocks = num_blocks
        self.block_type = block
        self.middle_dim = self.layer_configurations[0][0]
        self.img_size = img_size
        self.is_MHSA = is_MHSA
        self.n_heads = n_heads

        super().__init__(
            *[
                _InputBlock(input_dim, self.middle_dim),
                *self._make_stages(),
            ]
        )

    def _make_stages(self) -> List[nn.Sequential]:
        stages = list()
        for idx, (configuration, num_block) in enumerate(
            zip(self.layer_configurations, self.num_blocks)
        ):
            layers = list()
            strides = [configuration[1]] + [1] * (num_block - 1)
            for stride_idx, stride in enumerate(strides):
                args = [self.middle_dim, configuration[0]]
                kwargs = {
                    "stride": stride,
                    "skip_padding": 0,
                    "is_identity": stride_idx != 0,
                }
                if (
                    self.block_type is BottleNeck
                    and idx == len(self.layer_configurations) - 1
                ):
                    kwargs["is_MHSA"] = self.is_MHSA
                    kwargs["n_heads"] = self.n_heads
                    kwargs["img_size"] = self.img_size
                    if stride_idx == 0:
                        self.img_size = self.img_size // 4
                layers.append(self.block_type(*args, **kwargs))
                self.middle_dim = configuration[0] * self.block_type.expansion
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
                        self.block_type is BottleNeck and (not self.is_MHSA or idx != 1)
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
    bn.weight.data = weights.get(f"{bn_idx}.weight")


def _resnet(
    input_dim: int,
    num_blocks: list[int],
    block: RedisualBlock | BottleNeck,
    resnet_type: str,
    pretrained: bool,
    img_size: int = 256,
    is_MHSA: bool = False,
    n_heads: int = 4,
):
    model = _ResNet(input_dim, num_blocks, block, img_size, is_MHSA, n_heads)
    if pretrained:
        model.load_from(torch.load(PRETRAINED_MODELS.get(resnet_type)))
    return model


def resnet34(input_dim: int, pretrained: bool):
    return _resnet(input_dim, [3, 4, 6, 3], RedisualBlock, "resnet34", pretrained)


def resnet50(
    input_dim: int, img_size: int, is_MHSA: bool, n_heads: int, pretrained: bool
):
    return _resnet(
        input_dim,
        [3, 4, 6, 3],
        BottleNeck,
        "resnet50",
        pretrained,
        img_size,
        is_MHSA,
        n_heads,
    )


def resnet101(input_dim: int, pretrained: bool):
    return _resnet(input_dim, [3, 4, 23, 3], BottleNeck, "resnet101", pretrained)
