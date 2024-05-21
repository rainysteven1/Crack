import math


def lr_lambda1(epoch: int):
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
def lr_lambda2(epoch: int, epochs: int, y1: float = 0.05, y2: float = 1.0):
    return (((1 - math.cos(epoch * math.pi / epochs)) / 2) ** 1.0) * (y2 - y1) + y1


def lr_lambda3(epoch: int, epochs: int):
    return (1 - epoch / epochs) ** 0.9
