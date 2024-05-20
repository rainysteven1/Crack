import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import MS_SSIM


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


class IoULoss(nn.Module):
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        intersection = (y_pred * y_true).sum()
        total = (y_pred + y_true).sum()
        union = total - intersection
        IoU = (intersection + self.smooth) / (union + self.smooth)
        return 1 - IoU


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred, y_true):
        bce_loss = F.binary_cross_entropy(y_pred, y_true, reduction="mean")
        bce_exp = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1.0 - bce_exp) ** self.gamma * bce_loss
        return focal_loss


class HybridLoss1(nn.Module):
    """Focal + IoU + MS_SSIM."""

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


class StructureLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, y_pred, y_true):
        eit = 1 + 5 * torch.abs(
            F.avg_pool2d(y_true, kernel_size=31, stride=1, padding=15) - y_true
        )
        bce = F.binary_cross_entropy(y_pred, y_true, reduction="none")
        bce = (eit * bce).sum(dim=(2, 3)) / eit.sum(dim=(2, 3))

        inter = ((y_pred * y_true) * eit).sum(dim=(2, 3))
        union = ((y_pred + y_true) * eit).sum(dim=(2, 3))
        IoU = 1 - (inter + 1) / (union - inter + 1)
        return (bce + IoU).mean()
