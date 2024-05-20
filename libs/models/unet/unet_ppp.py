import numpy as np
import torch
import torch.nn as nn

from ..modules.base import BasicBlock, Conv2dSame, InitModule, OutputBlock
from .common import Encoder


class _DecoderBlock(nn.Module):
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
                    BasicBlock(
                        filters[i], cat_dim, padding="same", init_type=init_type
                    ),
                )
            )
        self.layers.append(BasicBlock(filters[zero_idx], cat_dim, padding=1))
        for i in range(zero_idx + 1, len(arr)):
            self.layers.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=arr[i], mode="bilinear"),
                    BasicBlock(
                        filters[i], cat_dim, padding="same", init_type=init_type
                    ),
                ),
            )

        self.output_layer = BasicBlock(
            up_dim, up_dim, padding="same", init_type=init_type
        )

    def forward(self, *inputs):
        x_list = list()
        for module, input in zip(self.layers, inputs):
            x_list.append(module(input))
        output = self.output_layer(torch.cat(x_list, dim=1))
        return output


class _BasicModule(InitModule):
    def __init__(
        self,
        input_dim: int,
        filters: list,
        init_type: str | None = "kaiming",
    ) -> None:
        super().__init__(init_type)
        assert len(filters) == 5
        self.cat_dim = filters[0]
        self.cat_num = len(filters)
        self.up_dim = self.cat_dim * self.cat_num
        self.matrix = [
            [8, 4, 2, 0, 2],
            [4, 2, 0, 2, 4],
            [2, 0, 2, 4, 8],
            [0, 2, 4, 8, 16],
        ]
        self.filters = filters
        self.d_length = len(self.filters) - 1

        # Encoder
        self.e = Encoder(input_dim, filters, init_type)

        # Decoder
        self.d = nn.ModuleList()
        for i in range(self.d_length):
            self.d.append(
                _DecoderBlock(
                    self.cat_dim,
                    self.up_dim,
                    self.filters,
                    arr=self.matrix[i],
                    init_type=init_type,
                )
            )
            self.filters[self.d_length - 1 - i] = self.up_dim

    def forward(self, input):
        # Encoder
        self.x_list = self.e(input)

        # Decoder
        for i in range(self.d_length):
            self.x_list[self.d_length - 1 - i] = self.d[i](*self.x_list)


class UNet3Plus(_BasicModule):
    """UNet3+"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        filters: list = [32, 64, 128, 256, 512],
        init_type: str | None = "kaiming",
    ) -> None:
        super().__init__(input_dim, filters, init_type)

        # Output
        self.output_layer = OutputBlock(
            self.up_dim, output_dim, kernel_size=3, init_type=init_type
        )

    def forward(self, input):
        super().forward(input)

        # Output
        output = self.output_layer(self.x_list[0])
        return output


class UNet3PlusDeepSup(_BasicModule):
    """UNet3+ with deep supervision."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        filters: list = [32, 64, 128, 256, 512],
        init_type: str | None = "kaiming",
    ) -> None:
        super().__init__(input_dim, filters, init_type)

        # Output
        self.up_scale_factors = self.matrix[-1][1:]
        self.output_layers = nn.ModuleList()
        for i in range(len(self.up_scale_factors)):
            self.output_layers.append(
                nn.Sequential(
                    Conv2dSame(
                        self.filters[i + 1], output_dim, kernel_size=3, padding="same"
                    ),
                    nn.Upsample(scale_factor=self.up_scale_factors[i], mode="bilinear"),
                    nn.Sigmoid(),
                )
            )
        self.output_layers.insert(
            0, Conv2dSame(self.up_dim, output_dim, kernel_size=3, padding="same")
        )

        self._initialize_weights()

    def forward(self, input):
        super().forward(input)

        # Output
        outputs = [module(x) for module, x in zip(self.output_layers, self.x_list)]
        return outputs

    def _initialize_weights(self):
        self.init(self.output_layers[0])
        self.output_layers[1:].apply(lambda s: s.apply(lambda m: self.init(m)))


class UNet3PlusDeepSupCGM(_BasicModule):
    """UNet3+ with deep supervision and class-guided module."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        filters: list = [32, 64, 128, 256, 512],
        init_type: str | None = "kaiming",
    ) -> None:
        super().__init__(input_dim, filters, init_type)

        # Output
        self.up_scale_factors = [1]
        self.up_scale_factors.extend(self.matrix[-1][1:])
        self.output_layers = nn.ModuleList()
        for i in range(len(self.up_scale_factors)):
            self.output_layers.append(
                nn.Sequential(
                    Conv2dSame(
                        self.filters[i], output_dim, kernel_size=3, padding="same"
                    ),
                    nn.Upsample(scale_factor=self.up_scale_factors[i], mode="bilinear"),
                )
            )

        self.cls = nn.Sequential(
            nn.Dropout(p=0.5),
            Conv2dSame(self.filters[4], 2, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid(),
        )

        self._initialize_weights()

    def dot_product(self, seg, cls):
        B, N, H, W = seg.size()
        seg = seg.view(B, N, H * W)
        final = torch.einsum("ijk,ij->ijk", [seg, cls])
        final = final.view(B, N, H, W)
        return final

    def forward(self, input):
        # Encoder
        self.x_list = self.e(input)

        # Classification
        cls_branch = self.cls(self.x_list[-1]).squeeze(3).squeeze(2)  # (B,N,1,1)->(B,N)
        cls_branch_max = cls_branch.argmax(dim=1)
        cls_branch_max = cls_branch_max[:, np.newaxis].float()

        # Decoder
        for i in range(self.d_length):
            self.x_list[self.d_length - 1 - i] = self.d[i](*self.x_list)

        outputs = [
            torch.sigmoid(self.dot_product(module(x), cls_branch_max))
            for module, x in zip(self.output_layers, self.x_list)
        ]
        return outputs

    def _initialize_weights(self):
        self.output_layers.apply(lambda s: s.apply(lambda m: self.init(m)))
        self.cls.apply(lambda x: self.init(x))
