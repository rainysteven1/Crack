

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import MS_SSIM


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        self.inputs = y_pred.view(-1)
        self.targets = y_true.view(-1)
        intersection = (self.inputs * self.targets).sum()
        dice_coef = (2.0 * intersection + self.smooth) / (
            self.inputs.sum() + self.targets.sum() + self.smooth
        )
        return 1.0 - dice_coef


class BCEDiceLoss(DiceLoss):
    def __init__(self, smooth=1.0) -> None:
        super().__init__(smooth)

    def forward(self, y_pred, y_true):
        dice_loss = super().forward(y_pred, y_true)
        bce_loss = F.binary_cross_entropy(self.inputs, self.targets, reduction="mean")
        return bce_loss + dice_loss


class IoULoss(nn.Module):
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        inputs = y_pred.view(-1)
        targets = y_true.view(-1)
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection
        IoU = (intersection + self.smooth) / (union + self.smooth)
        return 1 - IoU


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred, y_true):
        inputs = y_pred.view(-1)
        targets = y_true.view(-1)
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction="mean")
        bce_exp = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1.0 - bce_exp) ** self.gamma * bce_loss
        return focal_loss


class HybridLoss1(nn.Module):
    """
    Focal + IoU + MS_SSIM
    """

    def __init__(self, alpha=0.5, gamma=2, smooth=1) -> None:
        super().__init__()

        self.Focal = FocalLoss(alpha, gamma)
        self.IoU = IoULoss(smooth)
        self.MS_SSIM = MS_SSIM(channel=1)

    def forward(self, y_pred, y_true):
        return (
            self.Focal(y_pred, y_true)
            + self.IoU(y_pred, y_true)
            + 1.0
            - self.MS_SSIM(y_pred, y_true)
        )
