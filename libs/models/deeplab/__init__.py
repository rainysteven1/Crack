from .common import MSC
from .deeplabv2 import DeepLabV2


def DeepLabV2_ResNet101_MSC(
    input_dim: int,
    output_dim: int,
    atrous_rates: list,
    is_custom_resnet: bool = True,
    n_blocks: list = None,
):
    return MSC(
        DeepLabV2(input_dim, output_dim, atrous_rates, is_custom_resnet, n_blocks),
        scales=[0.5, 0.75],
    )
