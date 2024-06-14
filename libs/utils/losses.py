from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import MS_SSIM

_DEFAULT_CLASS_WEIGHTS = [0.04, 0.96]

__all__ = [
    "DiceLoss",
    "DiceBCELoss",
    "FocalLoss",
    "IoULoss",
    "HybridLoss",
    "StructureLoss",
]


def _dice_loss(y_pred: torch.Tensor, y_true: torch.Tensor, smooth: float = 1e-5):
    intersection = (y_pred * y_true).sum()
    return 1 - (2.0 * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        return _dice_loss(y_pred, y_true, self.smooth)


class DiceBCELoss(nn.Module):

    def __init__(
        self,
        loss_weights: List[float] = [1, 1],
        class_weights: List[float] = _DEFAULT_CLASS_WEIGHTS,
        smooth: float = 1e-5,
    ):
        super().__init__()
        self.loss_weights = loss_weights
        self.class_weights = class_weights
        self.smooth = smooth

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        weights = torch.zeros_like(y_true)
        weights = torch.fill_(weights, 0.04)
        weights[y_true > 0] = 0.96

        BCE = F.binary_cross_entropy(y_pred, y_true, reduction="mean", weight=weights)
        return self.loss_weights[0] * BCE + self.loss_weights[1] * _dice_loss(
            y_pred, y_true, self.smooth
        )


class IoULoss(nn.Module):
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        intersection = (y_pred * y_true).sum()
        total = (y_pred + y_true).sum()
        union = total - intersection
        IoU = (intersection + self.smooth) / (union + self.smooth)
        return 1 - IoU


class FocalLoss(nn.Module):
    def __init__(
        self,
        class_weights: List[float] = _DEFAULT_CLASS_WEIGHTS,
        alpha: float = 0.5,
        gamma: float = 2,
    ):
        super().__init__()
        self.class_weights = class_weights
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        weights = torch.zeros_like(y_true)
        weights = torch.fill_(weights, self.class_weights[0])
        weights[y_true > 0] = self.class_weights[1]

        bce_loss = F.binary_cross_entropy(y_pred, y_true, reduction="mean")
        bce_exp = torch.exp(-bce_loss)
        return self.alpha * (1.0 - bce_exp) ** self.gamma * bce_loss


class HybridLoss(nn.Module):
    """Focal + IoU + MS_SSIM."""

    def __init__(
        self, alpha: float = 0.5, gamma: float = 2, smooth: float = 1e-5
    ) -> None:
        super().__init__()
        self.Focal = FocalLoss(alpha, gamma)
        self.IoU = IoULoss(smooth)
        self.MS_SSIM = MS_SSIM(channel=1)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        return (
            self.Focal(y_pred, y_true)
            + self.IoU(y_pred, y_true)
            + 1.0
            - self.MS_SSIM(y_pred, y_true)
        )


class StructureLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        eit = 1 + 5 * torch.abs(
            F.avg_pool2d(y_true, kernel_size=31, stride=1, padding=15) - y_true
        )
        bce = F.binary_cross_entropy(y_pred, y_true, reduction="none")
        bce = (eit * bce).sum(dim=(2, 3)) / eit.sum(dim=(2, 3))

        inter = ((y_pred * y_true) * eit).sum(dim=(2, 3))
        union = ((y_pred + y_true) * eit).sum(dim=(2, 3))
        IoU = 1 - (inter + 1) / (union - inter + 1)
        return (bce + IoU).mean()
