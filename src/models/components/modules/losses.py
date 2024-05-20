import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5):
        super().__init__()

        self.smooth = smooth

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        N = y_true.size(0)
        pred = y_pred.view(N, -1)
        target = y_true.view(N, -1)

        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)

        return 1 - ((2 * intersection + self.smooth) / (union + self.smooth)).sum() / N


class BCEDiceLoss(DiceLoss):
    def __init__(self, smooth: float = 1e-5):
        super().__init__(smooth)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        return super().forward(y_pred, y_true) + nn.BCELoss()(y_pred, y_true)
