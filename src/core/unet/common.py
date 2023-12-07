import torch.nn as nn
from ..modules import Conv2dSame, InitModule


class ConvBlock(InitModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        middle_dim: int = None,
        kernel_size: int = 3,
        stride: int = 1,
        padding: str | int = 1,
        init_type=None,
        is_batchNorm=True,
    ) -> None:
        super().__init__(init_type)
        if not middle_dim:
            middle_dim = output_dim
        if not is_batchNorm:
            self.layers = nn.Sequential(
                Conv2dSame(
                    input_dim,
                    middle_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
                nn.ReLU(),
                Conv2dSame(
                    middle_dim,
                    output_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
                nn.ReLU(),
            )
        else:
            self.layers = nn.Sequential(
                Conv2dSame(
                    input_dim,
                    middle_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
                nn.BatchNorm2d(middle_dim),
                nn.ReLU(),
                Conv2dSame(
                    middle_dim,
                    output_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
                nn.BatchNorm2d(output_dim),
                nn.ReLU(),
            )

        if self.init_type:
            self._initialize_weights()

    def forward(self, x):
        return self.layers(x)

    def _initialize_weights(self):
        self.layers.apply(lambda m: self.init(m))


class EncoderBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, init_type=None) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            ConvBlock(input_dim, output_dim, init_type=init_type),
        )

    def forward(self, x):
        return self.layers(x)
