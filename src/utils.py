import io, math, sys
import torch

from logging import Logger
from torchsummary import summary

from src import DEVICE
from .core import MODEL_DICT


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


def log_model_summary(
    logger: Logger,
    model: torch.nn.Module,
    batch_size: int,
    input_dim: int,
    batch_height: int,
    batch_width: int,
    device: str,
):
    output = io.StringIO()
    sys.stdout = output
    summary(
        model,
        (input_dim, batch_height, batch_width),
        batch_size,
        device,
    )
    sys.stdout = sys.__stdout__
    summary_output = output.getvalue()
    logger.info("Model:\n{}".format(summary_output))


def build_model(category: str):
    return MODEL_DICT.get(category)(input_dim=3, output_dim=1).to(DEVICE)


def lr_schedule1(epoch):
    scale_factor = 1
    if epoch > 150:
        scale_factor *= 2 ** (-1)
    elif epoch > 80:
        scale_factor *= 2 ** (-1)
    elif epoch > 50:
        scale_factor *= 2 ** (-1)
    elif epoch > 30:
        scale_factor *= 2 ** (-1)
    return scale_factor


# https://arxiv.org/pdf/1812.01187.pdf
def lr_schedule2(epochs):
    return (
        lambda epoch: (((1 + math.cos(epoch * math.pi / epochs)) / 2) ** 1.0) * 0.95
        + 0.05
    )
