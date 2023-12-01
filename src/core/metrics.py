import torch


def IOU(y_pred, y_true):
    y_pred_f = torch.flatten(y_pred)
    y_true_f = torch.flatten(y_true)

    thresh = 0.5
    y_pred_f = torch.ge(y_pred, thresh).float()
    y_true_f = torch.ge(y_true, thresh).float()

    union = torch.sum(torch.maximum(y_pred_f, y_true_f)) + torch.finfo().eps
    intersection = torch.sum(torch.minimum(y_pred_f, y_true_f)) + torch.finfo().eps
    return intersection / union
