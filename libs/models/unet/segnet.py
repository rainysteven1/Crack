import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import VGG16_Weights, vgg16

from ..modules.base import BasicBlock, OutputBlock


class _EncoderBlock(nn.Sequential):
    def __init__(self, input_dim: int, output_dim: int, n_block: int) -> None:
        super().__init__(
            *(
                [BasicBlock(input_dim, output_dim)]
                + [BasicBlock(output_dim, output_dim) for _ in range(n_block - 1)]
            )
        )


class _DecoderBlock(nn.Sequential):
    def __init__(self, input_dim: int, output_dim: int, n_block: int) -> None:
        super().__init__(
            *(
                [BasicBlock(input_dim, input_dim) for _ in range(n_block - 1)]
                + [BasicBlock(input_dim, output_dim)]
            )
        )


class SegNet(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, filters: list = [64, 128, 256, 512, 512]
    ) -> None:
        super().__init__()
        encoder_filters = [input_dim] + filters
        decoder_filters = list(reversed(filters))

        self.n_blocks = [2, 2, 3, 3, 3]
        self.n_blocks_reverse = list(reversed(self.n_blocks))
        self.vgg16 = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

        # Encoder
        self.encoder_list = nn.ModuleList(
            [
                _EncoderBlock(
                    encoder_filters[i], encoder_filters[i + 1], self.n_blocks[i]
                )
                for i in range(len(self.n_blocks))
            ]
        )

        # Decoder
        self.decoder_list = nn.ModuleList(
            [
                _DecoderBlock(
                    decoder_filters[i], decoder_filters[i + 1], self.n_blocks_reverse[i]
                )
                for i in range(len(self.n_blocks_reverse) - 1)
            ]
        )

        # Output
        self.output_layer = nn.Sequential(
            BasicBlock(decoder_filters[-1], decoder_filters[-1]),
            OutputBlock(decoder_filters[-1], output_dim, kernel_size=3),
        )

        self._init_weights()

    def forward(self, input):
        x = input
        indices = list()
        for module in self.encoder_list:
            x, idx = F.max_pool2d(
                module(x), kernel_size=2, stride=2, return_indices=True
            )
            indices.append(idx)
        indices = list(reversed(indices))
        for module, idx in zip(self.decoder_list, indices[:-1]):
            x = module(F.max_unpool2d(x, idx, kernel_size=2, stride=2))
        output = self.output_layer(
            F.max_unpool2d(x, indices[-1], kernel_size=2, stride=2)
        )
        return output

    def _init_weights(self):
        def _init(idx: int, num: str, vgg_idx: int):
            self.encoder_list[idx].get_submodule(num).layers.get_submodule(
                "0"
            ).weight.data = self.vgg16.features[vgg_idx].weight.data
            self.encoder_list[idx].get_submodule(num).layers.get_submodule(
                "0"
            ).bias.data = self.vgg16.features[vgg_idx].bias.data

        params_dict = dict.fromkeys(range(len(self.encoder_list)))
        params_dict[0] = [("0", 0), ("1", 2)]
        params_dict[1] = [("0", 5), ("1", 7)]
        params_dict[2] = [("0", 10), ("1", 12), ("2", 14)]
        params_dict[3] = [("0", 17), ("1", 19), ("2", 21)]
        params_dict[4] = [("0", 24), ("1", 26), ("2", 28)]

        for key, values in params_dict.items():
            for value in values:
                _init(key, value[0], value[1])
