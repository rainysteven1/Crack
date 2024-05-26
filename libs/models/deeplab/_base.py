from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

from ..modules.base import IntermediateSequential
from ..modules.pyramid import ASPP_v2, ASPP_v3


class DeepLabHead(nn.Module):

    def __init__(
        self,
        output_dim: int,
        middle_dim: Optional[int],
        backbone: IntermediateSequential,
        ASPP_arch: Union[ASPP_v2, ASPP_v3],
        atrous_rates: List[int],
        init_type: Optional[str],
    ) -> None:
        super().__init__()
        aspp_input_dim = backbone.dims[-1] * backbone.block_type.expansion
        self.middle_dim = middle_dim or output_dim

        self.backbone = backbone
        self.ASPP = ASPP_arch(aspp_input_dim, self.middle_dim, atrous_rates, init_type)

    def forward(self, input: torch.Tensor):
        return self.ASPP(self.backbone(input))

    def get_params(self, keywords: Dict) -> list[Dict]:
        params = [
            {
                "params": self.backbone.get_params(),
                "lr": keywords["lr"],
                "weight_decay": keywords["weight_decay"],
            },
            {
                "params": self.ASPP.get_weight(),
                "lr": 10 * keywords["lr"],
                "weight_decay": keywords["weight_decay"],
            },
        ]
        if self.ASPP.is_bias:
            params.append(
                {
                    "params": self.ASPP.get_bias(),
                    "lr": 20 * keywords["lr"],
                    "weight_decay": 0.0,
                },
            )
        return params

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def freeze_backbone_bn(self):
        self.backbone.freeze_bn()
