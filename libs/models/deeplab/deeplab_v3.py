from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbone import resnet
from ..modules.base import BasicBlock
from ..modules.pyramid import ASPP_v3


class _DeepLabV3(nn.Module):
    """DeepLab v3: Dilated ResNet with multi-grid + improved ASPP"""

    def __init__(
        self,
        output_dim: int,
        middle_dim: int,
        backbone: nn.Sequential,
        atrous_rates: List[int],
        init_type: Optional[str],
    ):
        super().__init__()

        self.backbone = backbone
        aspp_input_dim = self.backbone.dims[-1] * self.backbone.block_type.expansion
        self.ASPP = ASPP_v3(
            aspp_input_dim, middle_dim, atrous_rates, init_type=init_type
        )

        concat_dim = middle_dim * (len(atrous_rates) + 2)
        self.fc1 = BasicBlock(
            concat_dim,
            middle_dim,
            kernel_size=1,
            padding=0,
            is_bias=False,
            init_type=init_type,
        )
        self.fc2 = BasicBlock(
            middle_dim,
            output_dim,
            kernel_size=1,
            padding=0,
            is_bn=False,
            is_relu=False,
            is_bias=False,
            init_type=init_type,
        )

    def forward(self, input: torch.Tensor):
        x = self.ASPP(self.backbone(input))
        x = self.fc2(self.fc1(x))
        return F.sigmoid(
            F.interpolate(
                x, size=input.size()[-2:], mode="bilinear", align_corners=False
            )
        )

    def get_params(self, keywords: Dict) -> list[Dict]:
        params = [
            {
                "params": self.backbone.get_params(),
                "lr": keywords["lr"],
                "weight_decay": keywords["weight_decay"],
            },
            {
                "params": self.ASPP.get_weight(),
                "lr": 10 * keywords["lr"],
                "weight_decay": keywords["weight_decay"],
            },
        ]
        if self.ASPP.is_bias:
            params.append(
                {
                    "params": self.ASPP.get_bias(),
                    "lr": 20 * keywords["lr"],
                    "weight_decay": 0.0,
                },
            )
        return params

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def freeze_backbone_bn(self):
        self.backbone.freeze_bn()


class _DeepLabV3Plus(_DeepLabV3):
    """DeepLab v3+: Dilated ResNet with multi-grid + improved ASPP + decoder."""

    def __init__(
        self,
        output_dim: int,
        middle_dim: int,
        backbone: nn.Sequential,
        atrous_rates: List[int],
        init_type: Optional[str],
    ):
        super().__init__(output_dim, middle_dim, backbone, atrous_rates, init_type)
        reduce_dim = 48

        self.reduce = BasicBlock(
            middle_dim,
            reduce_dim,
            kernel_size=1,
            padding=0,
            is_bias=False,
            init_type=init_type,
        )
        self.fc2 = nn.Sequential(
            BasicBlock(
                middle_dim + reduce_dim,
                middle_dim,
                is_bias=False,
                init_type=init_type,
            ),
            BasicBlock(middle_dim, middle_dim, is_bias=False, init_type=init_type),
            self.fc2,
        )

    def forward(self, input: torch.Tensor):
        x_list = self.backbone(input)
        x_ = self.reduce(x_list[1])
        x = F.interpolate(
            self.fc1(self.ASPP(x_list[-1])),
            size=x_.size()[-2:],
            mode="bilinear",
            align_corners=True,
        )
        x = F.interpolate(
            self.fc2(torch.cat((x, x_), dim=1)),
            size=input.size()[-2:],
            mode="bilinear",
            align_corners=True,
        )
        return F.sigmoid(x)


def _segm_resnet(
    arch: str,
    input_dim: int,
    output_dim: int,
    backbone: str,
    output_stride: int,
    pretrained: bool,
    init_type: Optional[str],
):
    if arch == "DeepLabV3":
        model = _DeepLabV3
        return_intermediate = False
    elif arch == "DeepLabV3+":
        model = _DeepLabV3Plus
        return_intermediate = True

    if output_stride == 8:
        backbone_cfg = {
            "strides": [1, 2, 1, 1],
            "dilations": [1, 1, 2, 2],
            "multi_grids": [1, 2, 1],
        }
        atrous_rates = [12, 24, 36]
    elif output_stride == 16:
        backbone_cfg = {
            "strides": [1, 2, 2, 1],
            "dilations": [1, 1, 1, 2],
            "multi_grids": [1, 2, 4],
        }
        atrous_rates = [6, 12, 18]

    backbone_cfg["return_intermediate"] = return_intermediate
    backbone_cfg["init_type"] = init_type
    backbone = resnet.__dict__[backbone](
        input_dim=input_dim, pretrained=pretrained, **backbone_cfg
    )
    return model(output_dim, 256, backbone, atrous_rates, init_type)


def _load_model(
    arch: str,
    input_dim: int,
    output_dim: int,
    backbone: str,
    output_stride: int,
    pretrained: bool,
    init_type: Optional[str],
):
    if backbone.startswith("resnet"):
        model = _segm_resnet(
            arch, input_dim, output_dim, backbone, output_stride, pretrained, init_type
        )
    return model


def deeplabv3_resnet50(
    input_dim: int,
    output_dim: int,
    output_stride: int,
    pretrained: bool,
    init_type: Optional[str],
):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        input_dim (int): number of input channels.
        output_dim (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
        init_type (str): initialization type.
    """
    return _load_model(
        "DeepLabV3",
        input_dim,
        output_dim,
        "resnet50",
        output_stride,
        pretrained,
        init_type,
    )


def deeplabv3_resnet101(
    input_dim: int,
    output_dim: int,
    output_stride: int,
    pretrained: bool,
    init_type: Optional[str],
):
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.

    Args:
        input_dim (int): number of input channels.
        output_dim (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
        init_type (str): initialization type.
    """
    return _load_model(
        "DeepLabV3",
        input_dim,
        output_dim,
        "resnet101",
        output_stride,
        pretrained,
        init_type,
    )


def deeplabv3_plus_resnet50(
    input_dim: int,
    output_dim: int,
    output_stride: int,
    pretrained: bool,
    init_type: Optional[str],
):
    """Constructs a DeepLabV3+ model with a ResNet-50 backbone.

    Args:
        input_dim (int): number of input channels.
        output_dim (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
        init_type (str): initialization type.
    """
    return _load_model(
        "DeepLabV3+",
        input_dim,
        output_dim,
        "resnet50",
        output_stride,
        pretrained,
        init_type,
    )


def deeplabv3_plus_resnet101(
    input_dim: int,
    output_dim: int,
    output_stride: int,
    pretrained: bool,
    init_type: Optional[str],
):
    """Constructs a DeepLabV3+ model with a ResNet-101 backbone.

    Args:
        input_dim (int): number of input channels.
        output_dim (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
        init_type (str): initialization type.
    """
    return _load_model(
        "DeepLabV3+",
        input_dim,
        output_dim,
        "resnet101",
        output_stride,
        pretrained,
        init_type,
    )
