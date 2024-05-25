from typing import List, Optional

import torch
import torch.nn as nn


class DeepLabV3Plus(nn.Module):
    """DeepLab v3+: Dilated ResNet with multi-grid + improved ASPP + decoder."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        pretrained: bool,
        strides: Optional[List[int]],
        dilations: Optional[List[int]],
        multi_grids: Optional[List[int]],
        atrous_rates: List[int],
        init_type: Optional[str],
    ):
        super().__init__()
