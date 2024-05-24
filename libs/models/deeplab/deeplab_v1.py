from typing import Optional

import torch.nn as nn

from ..modules.base import BasicBlock, OutputBlock


class _BasicBlock(nn.Sequential):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        padding: int,
        dilation: int,
        n_blocks: int,
        max_pool_stride: int,
    ) -> None:
        super().__init__(
            *(
                [
                    BasicBlock(
                        input_dim,
                        output_dim,
                        padding=padding,
                        dilation=dilation,
                        is_bn=False,
                    )
                ]
                + [
                    BasicBlock(
                        output_dim,
                        output_dim,
                        padding=padding,
                        dilation=dilation,
                        is_bn=False,
                    )
                ]
                * (n_blocks - 1)
            ),
            nn.MaxPool2d(kernel_size=3, stride=max_pool_stride, padding=1),
        )


# TODO 1. CRF
class DeepLabV1(nn.Sequential):
    """
    Bottleneck Transformers Multi-Head Self-Attention module
    reference: https://github.com/wangleihitcs/DeepLab-V1-PyTorch/blob/master/nets/vgg.py
    """

    def __init__(
        self, input_dim: int, output_dim: int, init_type: Optional[str]
    ) -> None:
        super().__init__(
            _BasicBlock(
                input_dim, 64, padding=1, dilation=1, n_blocks=2, max_pool_stride=2
            ),
            _BasicBlock(64, 128, padding=1, dilation=1, n_blocks=2, max_pool_stride=2),
            _BasicBlock(128, 256, padding=1, dilation=1, n_blocks=3, max_pool_stride=2),
            _BasicBlock(256, 512, padding=1, dilation=1, n_blocks=3, max_pool_stride=1),
            _BasicBlock(512, 512, padding=2, dilation=2, n_blocks=3, max_pool_stride=1),
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicBlock(512, 1024, padding=12, dilation=12, is_bn=False),
            nn.Dropout2d(0.5),
            BasicBlock(1024, 1024, kernel_size=1, padding=0, is_bn=False),
            nn.Dropout2d(0.5),
            OutputBlock(1024, output_dim, init_type=init_type),
        )
