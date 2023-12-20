import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
from .losses import *

OPTIMIZER_DICT = {
    "Adam": optim.Adam,
    "AdamW": optim.AdamW,
    "NAdam": optim.NAdam,
    "SGD": optim.SGD,
}

LOSS_DICT = {
    "BCELoss": nn.BCELoss,
    "DiceLoss": DiceLoss,
    "BCEDiceLoss": BCEDiceLoss,
    "FocalLoss": FocalLoss,
    "IoULoss": IoULoss,
    "HybridLoss1": HybridLoss1,
}


SCHEDULER_DICT = {
    "LambdaLR": scheduler.LambdaLR,
    "CosineAnnealingLR": scheduler.CosineAnnealingLR,
    "ReduceLROnPlateau": scheduler.ReduceLROnPlateau,
}
