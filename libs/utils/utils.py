import numpy as np
import torch


def np2th(weights: np.ndarray, conv: bool = False):
    """Possibly convert HWIO to OIHW."""
    return torch.from_numpy(weights if not conv else weights.transpose([3, 2, 0, 1]))
