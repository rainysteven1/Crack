import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        intersection = torch.sum(y_pred * y_true)
        dice_coef = (2.0 * intersection + self.smooth) / (
            torch.sum(y_pred) + torch.sum(y_true) + self.smooth
        )
        return 1.0 - dice_coef


class BCEDiceLoss(DiceLoss):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, y_pred, y_true):
        bce_loss = nn.BCELoss()(y_pred, y_true).float()
        dice_loss = super().forward(y_pred, y_true)
        return bce_loss + dice_loss
