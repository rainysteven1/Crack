import torch
import torch.nn as nn
from .common import SingleBlock, Encoder
from ..modules import OutputBlock


class DecoderBlock(nn.Module):
    def __init__(
        self,
        cat_dim: int,
        up_dim: int,
        filters: list,
        arr: list,
        init_type: str | None = None,
    ):
        super().__init__()
        assert len(filters) == len(arr)
        zero_idx = list(filter(lambda x: x[1] == 0, enumerate(arr)))[0][0]

        self.layers = nn.ModuleList()
        for i in range(0, zero_idx):
            self.layers.append(
                nn.Sequential(
                    nn.MaxPool2d(arr[i], arr[i], ceil_mode=True),
                    SingleBlock(
                        filters[i], cat_dim, padding="same", init_type=init_type
                    ),
                )
            )
        self.layers.append(SingleBlock(filters[zero_idx], cat_dim, padding=1))
        for i in range(zero_idx + 1, len(arr)):
            self.layers.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=arr[i], mode="bilinear"),
                    SingleBlock(
                        filters[i], cat_dim, padding="same", init_type=init_type
                    ),
                ),
            )

        self.output_layer = SingleBlock(
            up_dim, up_dim, padding="same", init_type=init_type
        )

    def forward(self, *inputs):
        x_list = list()
        for module, input in zip(self.layers, inputs):
            x_list.append(module(input))
        output = self.output_layer(torch.cat(x_list, dim=1))
        return output


class UNet3Plus(nn.Module):
    """
    UNet+++
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        filters: list = [32, 64, 128, 256, 512],
        init_type: str | None = "kaiming",
    ) -> None:
        super().__init__()
        assert len(filters) == 5
        cat_dim = filters[0]
        cat_num = len(filters)
        up_dim = cat_dim * cat_num
        matrix = [[8, 4, 2, 0, 2], [4, 2, 0, 2, 4], [2, 0, 2, 4, 8], [0, 2, 4, 8, 16]]
        self.filters = filters
        self.d_length = len(self.filters) - 1

        # Encoder
        self.e = Encoder(input_dim, filters, init_type)

        # Decoder
        self.d = nn.ModuleList()
        for i in range(self.d_length):
            self.d.append(
                DecoderBlock(
                    cat_dim, up_dim, self.filters, arr=matrix[i], init_type=init_type
                )
            )
            self.filters[self.d_length - 1 - i] = up_dim

        # Output
        self.output_layer = OutputBlock(
            up_dim, output_dim, kernel_size=3, init_type=init_type
        )

    def forward(self, input):
        # Encoder
        x_list = self.e(input)

        # Decoder
        for i in range(self.d_length):
            x_list[self.d_length - 1 - i] = self.d[i](*x_list)

        output = self.output_layer(x_list[0])
        return output
