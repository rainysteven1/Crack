import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_SCALES = [0.5, 0.75]


class MSC(nn.Module):
    """
    Multi-scale inputs
    """

    def __init__(self, base: nn.Module, scales: list = None) -> None:
        super().__init__()

        self.base = base
        self.scales = scales if scales else DEFAULT_SCALES

    def forward(self, input):
        # Original
        logits = self.base(input)
        _, _, height, width = logits.shape

        # Scaled
        logits_pyramid = [
            self.base(
                F.interpolate(
                    input, scale_factor=s, mode="bilinear", align_corners=False
                )
            )
            for s in self.scales
        ]

        # Pixel-wise max
        logits_all = [logits] + [
            F.interpolate(l, size=(height, width), mode="bilinear", align_corners=False)
            for l in logits_pyramid
        ]
        logits_max = torch.max(torch.stack(logits_all), dim=0)[0]

        return logits_max
