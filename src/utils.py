import io, math, sys
import torch
import torch.nn as nn

from logging import Logger
from torch.optim.lr_scheduler import LambdaLR
from torchsummary import summary

from src import DEVICE
from src import MODEL_DICT, LOSS_DICT, OPTIMIZER_DICT, SCHEDULER_DICT

INPUT_DIM = 3


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
    batch_height: int,
    batch_width: int,
    device: str,
):
    output = io.StringIO()
    sys.stdout = output
    summary(
        model,
        (INPUT_DIM, batch_height, batch_width),
        batch_size,
        device,
    )
    sys.stdout = sys.__stdout__
    summary_output = output.getvalue()
    logger.info("Model:\n{}".format(summary_output))


def build_model(category: str):
    return MODEL_DICT.get(category)(INPUT_DIM, output_dim=1).to(DEVICE)


def get_criterion(category: str):
    return LOSS_DICT.get(category)()


def get_optimizer(optimizer_settings: dict, model: nn.Module):
    optimizer_name = optimizer_settings["name"]
    del optimizer_settings["name"]
    optimizer = OPTIMIZER_DICT.get(optimizer_name)(
        model.parameters(), **optimizer_settings
    )
    return optimizer


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
def lr_schedule2(epochs, y1=0.05, y2=1):
    return (
        lambda x: (((1 - math.cos(x * math.pi / epochs)) / 2) ** 1.0) * (y2 - y1) + y1
    )


def get_scheduler(scheduler_settings: dict, optimizer, epochs: int = 100):
    if "name" in scheduler_settings:
        scheduler_class = SCHEDULER_DICT.get(scheduler_settings["name"])
        del scheduler_settings["name"]
        if scheduler_class == LambdaLR:
            lr_lambda_dict = {
                "lr_schedule1": lr_schedule1,
                "lr_schedule2": lr_schedule2(epochs)
                if "lrf" not in scheduler_settings
                else lr_schedule2(epochs, 1, scheduler_settings["lrf"]),
            }
            lr_lambda = lr_lambda_dict.get(scheduler_settings["lr_lambda"])
            scheduler = LambdaLR(optimizer, lr_lambda)
        else:
            scheduler = scheduler_class(optimizer, **scheduler_settings)
    return scheduler
