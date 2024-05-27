from typing import List, Optional

import torch
import torch.nn.functional as F

from ..modules.base import IntermediateSequential
from ..modules.pyramid import ASPP_v2
from ._base import DeepLabHead


class DeepLabV2(DeepLabHead):
    """DeepLabV2: Dilated ResNet + ASPP"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        middle_dim: Optional[int],
        backbone: IntermediateSequential,
        atrous_rates: List[int],
        init_type: Optional[str],
    ) -> None:
        super().__init__(
            input_dim,
            output_dim,
            middle_dim,
            backbone,
            ASPP_v2,
            atrous_rates,
            init_type,
        )

    def forward(self, input: torch.Tensor):
        x = super().forward(input)
        return F.sigmoid(
            F.interpolate(
                x, size=input.size()[-2:], mode="bilinear", align_corners=True
            )
        )
