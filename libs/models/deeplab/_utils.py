from typing import Optional

import torch.nn as nn

from ..backbone import mobilenet, resnet, xception
from ._deeplabv1 import DeepLabV1
from ._deeplabv2 import DeepLabV2
from ._deeplabv3 import DeepLabV3, DeepLabV3Plus

_ARCH_DICT = {
    "DeepLabV1": DeepLabV1,
    "DeepLabV2": DeepLabV2,
    "DeepLabV3": DeepLabV3,
    "DeepLabV3+": DeepLabV3Plus,
}


def _segm_mobilenet(
    arch: str,
    input_dim: int,
    output_dim: int,
    backbone_name: str,
    pretrained: bool,
    output_stride: Optional[int],
    init_type: Optional[str],
):
    model = _ARCH_DICT[arch]
    backbone_cfg = {
        "output_stride": output_stride,
        "width_mult": 1.0,
        "inverted_redisual_cfg": None,
        "init_type": init_type,
        "return_intermediate": False,
    }
    model_cfg = {
        "middle_dim": None,
        "atrous_rates": [12, 24, 36] if output_stride == 8 else [6, 12, 18],
        "init_type": init_type,
    }

    if arch == "DeepLabV3+":
        backbone_cfg["return_intermediate"] = True
        if backbone_name == "mobilenetv2":
            model_cfg["return_intermediate_index"] = 2
        elif backbone_name.startswith("mobilenetv3"):
            model_cfg["return_intermediate_index"] = 4

    if backbone_name == "mobilenetv2":
        dim_index = 1
    else:
        dim_index = 2
        backbone_cfg["inverted_redisual_cfg"] = backbone_name.split("_")[-1]

    backbone: nn.Module = mobilenet.__dict__[backbone_name](
        input_dim, pretrained, **backbone_cfg
    )
    if arch == "DeepLabV3+":
        model_cfg["middle_dim"] = backbone.inverted_redisual_cfg[
            model_cfg["return_intermediate_index"] - 1
        ][dim_index]
    aspp_input_dim = backbone.inverted_redisual_cfg[-1][dim_index]
    return model(aspp_input_dim, output_dim, backbone=backbone, **model_cfg)


def _segm_resnet(
    arch: str,
    input_dim: int,
    output_dim: int,
    backbone: str,
    pretrained: bool,
    output_stride: Optional[int],
    init_type: Optional[str],
):
    return_intermediate = False
    model = _ARCH_DICT[arch]
    middle_dim = 256
    model_cfg = dict()

    if arch == "DeepLabV3+":
        return_intermediate = True
        model_cfg = {"return_intermediate_index": 1}
    elif arch == "DeepLabV2":
        middle_dim = None

    if output_stride is None:
        backbone_cfg = {
            "strides": [1, 2, 2, 2],
            "dilations": [1, 1, 1, 1],
        }
        atrous_rates = [6, 12, 18, 24]
    elif output_stride == 8:
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
    backbone = resnet.__dict__[backbone](input_dim, pretrained, **backbone_cfg)
    aspp_input_dim = backbone.dims[-1] * backbone.block_type.expansion
    return model(
        aspp_input_dim,
        output_dim,
        middle_dim,
        backbone,
        atrous_rates,
        init_type,
        **model_cfg
    )


def _segm_xception(
    arch: str,
    input_dim: int,
    output_dim: int,
    backbone_name: str,
    pretrained: bool,
    output_stride: Optional[int],
    init_type: Optional[str],
):
    model = _ARCH_DICT[arch]
    backbone_cfg = {
        "output_stride": output_stride,
        "redisual_block_cfg": None,
        "init_type": init_type,
        "return_intermediate": False,
    }
    model_cfg = {
        "middle_dim": None,
        "atrous_rates": [12, 24, 36] if output_stride == 8 else [6, 12, 18],
        "init_type": init_type,
    }

    if arch == "DeepLabV3+":
        backbone_cfg["return_intermediate"] = True

    backbone: nn.Module = xception.__dict__[backbone_name](
        input_dim, pretrained, **backbone_cfg
    )
    dim_index = 0
    if arch == "DeepLabV3+":
        model_cfg["return_intermediate_index"] = 1
        model_cfg["middle_dim"] = backbone.redisual_block_cfg[
            model_cfg["return_intermediate_index"] - 1
        ][dim_index]
    aspp_input_dim = backbone.last_output_dim
    return model(aspp_input_dim, output_dim, backbone=backbone, **model_cfg)


def _load_model(
    arch: str,
    input_dim: int,
    output_dim: int,
    backbone: str,
    pretrained: bool,
    output_stride: Optional[int],
    init_type: Optional[str],
):
    if backbone.startswith("mobilenet"):
        model = _segm_mobilenet
    elif backbone.startswith("resnet"):
        model = _segm_resnet
    elif backbone == "xception":
        model = _segm_xception
    return model(
        arch,
        input_dim,
        output_dim,
        backbone,
        pretrained,
        output_stride,
        init_type,
    )


# DeepLabV1
def deeplabv1(
    input_dim: int, output_dim: int, pretrained: bool, init_type: Optional[str]
):
    return _ARCH_DICT["DeepLabV1"](input_dim, output_dim, pretrained, init_type)


# DeepLabV2


def deeplabv2_mobilenetv2(
    input_dim: int,
    output_dim: int,
    pretrained: bool,
    output_stride: Optional[int],
    init_type: Optional[str],
):
    """Constructs a DeepLabV2 model with a MobileNetV2 backbone.

    Args:
        input_dim (int): number of input channels.
        output_dim (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
        init_type (str): initialization type.
    """
    return _load_model(
        "DeepLabV2",
        input_dim,
        output_dim,
        "mobilenetv2",
        pretrained,
        output_stride,
        init_type,
    )


def deeplabv2_resnet101(
    input_dim: int,
    output_dim: int,
    pretrained: bool,
    output_stride: Optional[int],
    init_type: Optional[str],
):
    """Constructs a DeepLabV2 model with a ResNet-101 backbone.

    Args:
        input_dim (int): number of input channels.
        output_dim (int): number of classes.
        pretrained_backbone (bool): If True, use the pretrained backbone.
        init_type (str): initialization type.
    """
    return _load_model(
        "DeepLabV2",
        input_dim,
        output_dim,
        "resnet101",
        pretrained,
        output_stride,
        init_type,
    )


def deeplabv2_xception(
    input_dim: int,
    output_dim: int,
    pretrained: bool,
    output_stride: Optional[int],
    init_type: Optional[str],
):
    """Constructs a DeepLabV2 model with a Xception backbone.

    Args:
        input_dim (int): number of input channels.
        output_dim (int): number of classes.
        pretrained_backbone (bool): If True, use the pretrained backbone.
        init_type (str): initialization type.
    """
    return _load_model(
        "DeepLabV2",
        input_dim,
        output_dim,
        "xception",
        pretrained,
        output_stride,
        init_type,
    )


# DeepLabV3


def deeplabv3_mobilenetv2(
    input_dim: int,
    output_dim: int,
    pretrained: bool,
    output_stride: int,
    init_type: Optional[str],
):
    """Constructs a DeepLabV3 model with a MobileNetV2 backbone.

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
        "mobilenetv2",
        pretrained,
        output_stride,
        init_type,
    )


def deeplabv3_mobilenetv3_large(
    input_dim: int,
    output_dim: int,
    pretrained: bool,
    output_stride: int,
    init_type: Optional[str],
):
    """Constructs a DeepLabV3 model with a MobileNetV3_Large backbone.

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
        "mobilenetv3_large",
        pretrained,
        output_stride,
        init_type,
    )


def deeplabv3_mobilenetv3_small(
    input_dim: int,
    output_dim: int,
    pretrained: bool,
    output_stride: int,
    init_type: Optional[str],
):
    """Constructs a DeepLabV3 model with a MobileNetV3_Small backbone.

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
        "mobilenetv3_small",
        pretrained,
        output_stride,
        init_type,
    )


def deeplabv3_resnet50(
    input_dim: int,
    output_dim: int,
    pretrained: bool,
    output_stride: Optional[int],
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
        pretrained,
        output_stride,
        init_type,
    )


def deeplabv3_resnet101(
    input_dim: int,
    output_dim: int,
    pretrained: bool,
    output_stride: Optional[int],
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
        pretrained,
        output_stride,
        init_type,
    )


def deeplabv3_xception(
    input_dim: int,
    output_dim: int,
    pretrained: bool,
    output_stride: int,
    init_type: Optional[str],
):
    """Constructs a DeepLabV3 model with a Xception backbone.

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
        "xception",
        pretrained,
        output_stride,
        init_type,
    )


# DeepLabV3+


def deeplabv3_plus_mobilenetv2(
    input_dim: int,
    output_dim: int,
    pretrained: bool,
    output_stride: int,
    init_type: Optional[str],
):
    """Constructs a DeepLabV3+ model with a MobileNetV2 backbone.

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
        "mobilenetv2",
        pretrained,
        output_stride,
        init_type,
    )


def deeplabv3_plus_mobilenetv3_large(
    input_dim: int,
    output_dim: int,
    pretrained: bool,
    output_stride: int,
    init_type: Optional[str],
):
    """Constructs a DeepLabV3+ model with a MobileNetV3_Large backbone.

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
        "mobilenetv3_large",
        pretrained,
        output_stride,
        init_type,
    )


def deeplabv3_plus_mobilenetv3_small(
    input_dim: int,
    output_dim: int,
    pretrained: bool,
    output_stride: int,
    init_type: Optional[str],
):
    """Constructs a DeepLabV3+ model with a MobileNetV3_Small backbone.

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
        "mobilenetv3_small",
        pretrained,
        output_stride,
        init_type,
    )


def deeplabv3_plus_resnet50(
    input_dim: int,
    output_dim: int,
    pretrained: bool,
    output_stride: Optional[int],
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
        pretrained,
        output_stride,
        init_type,
    )


def deeplabv3_plus_resnet101(
    input_dim: int,
    output_dim: int,
    pretrained: bool,
    output_stride: Optional[int],
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
        pretrained,
        output_stride,
        init_type,
    )


def deeplabv3_plus_xception(
    input_dim: int,
    output_dim: int,
    pretrained: bool,
    output_stride: int,
    init_type: Optional[str],
):
    """Constructs a DeepLabV3+ model with a Xception backbone.

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
        "xception",
        pretrained,
        output_stride,
        init_type,
    )
