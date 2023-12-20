import torch


def IoU(y_pred, y_true, thresh=0.5):
    inputs = torch.ge(y_pred.view(-1), thresh).float()
    targets = torch.ge(y_true.view(-1), thresh).float()

    union = torch.sum(torch.maximum(inputs, targets)) + torch.finfo().eps
    intersection = torch.sum(torch.minimum(inputs, targets)) + torch.finfo().eps
    return intersection / union
