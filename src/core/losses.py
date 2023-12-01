import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        y_pred_f = torch.flatten(y_pred)
        y_true_f = torch.flatten(y_true)
        intersection = torch.sum(y_pred_f * y_true_f)
        dice_coef = (2.0 * intersection + self.smooth) / (
            torch.sum(y_pred_f) + torch.sum(y_true_f) + self.smooth
        )
        return 1.0 - dice_coef
