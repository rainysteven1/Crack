from typing import Optional

from ..backbone import resnet
from ._deeplabv1 import DeepLabV1
from ._deeplabv2 import DeepLabV2
from ._deeplabv3 import DeepLabV3, DeepLabV3Plus

_ARCH_DICT = {
    "DeepLabV1": DeepLabV1,
    "DeepLabV2": DeepLabV2,
    "DeepLabV3": DeepLabV3,
    "DeepLabV3+": DeepLabV3Plus,
}


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

    if arch == "DeepLabV3+":
        return_intermediate = True
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
    backbone = resnet.__dict__[backbone](
        input_dim=input_dim, pretrained=pretrained, **backbone_cfg
    )
    return model(output_dim, middle_dim, backbone, atrous_rates, init_type)


def _load_model(
    arch: str,
    input_dim: int,
    output_dim: int,
    backbone: str,
    pretrained: bool,
    output_stride: Optional[int],
    init_type: Optional[str],
):
    if backbone.startswith("resnet"):
        model = _segm_resnet(
            arch, input_dim, output_dim, backbone, pretrained, output_stride, init_type
        )
    return model


# DeepLabV1
def deeplabv1(
    input_dim: int, output_dim: int, pretrained: bool, init_type: Optional[str]
):
    return _ARCH_DICT["DeepLabV1"](input_dim, output_dim, pretrained, init_type)


# DeepLabV2
def deeplabv2_resnet101(
    input_dim: int,
    output_dim: int,
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
        "DeepLabV2", input_dim, output_dim, "resnet101", pretrained, None, init_type
    )


# DeepLabV3
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
        pretrained,
        output_stride,
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
        pretrained,
        output_stride,
        init_type,
    )


# DeepLabV3+
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
        pretrained,
        output_stride,
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
        pretrained,
        output_stride,
        init_type,
    )
