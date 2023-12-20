import torch.nn as nn

from torchvision.models import resnet101, ResNet101_Weights

from ..modules.pyramid import ASPP_v2
from ..modules.resnet import BasicBlock, ResNet0


class _InputBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()

        self.layers = nn.Sequential(
            BasicBlock(input_dim, output_dim, 7, 2, 3, 1),
            nn.MaxPool2d(3, 2, 1, ceil_mode=True),
        )

    def forward(self, input):
        return self.layers(input)


class DeepLabV2(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        atrous_rates: list,
        data_attributes: list,
        is_custom_resnet: bool = True,
        n_blocks: list = None,
    ) -> None:
        super().__init__()
        n_dims = [64 * 2**p for p in range(6)]

        layer_list = list()
        input_block = _InputBlock(input_dim, n_dims[0])
        if is_custom_resnet:
            layer_params = [(1, 1), (2, 1), (1, 2), (1, 4)]
            layer_list.append(ResNet0(n_blocks, n_dims, layer_params, input_block))
        else:
            layer_list.append(input_block)
            resnet = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
            resnet_layers = [resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4]
            layer_list.extend(resnet_layers)

        layer_list.extend(
            [
                ASPP_v2(n_dims[-1], output_dim, atrous_rates),
                nn.Upsample(size=data_attributes, mode="bilinear", align_corners=True),
                nn.Sigmoid(),
            ]
        )
        self.layers = nn.Sequential(*layer_list)

    def forward(self, input):
        return self.layers(input)
