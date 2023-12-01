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


# https://github.com/pytorch/examples/blob/master/imagenet/main.py
class MetricTracker(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def get_avg(self):
        return self.avg
