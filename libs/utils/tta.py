import torch
import torch.nn as nn


def horizontal_flip(image: torch.Tensor):
    image = image[:, :, :, ::-1]
    return image


def vertical_flip(image: torch.Tensor):
    image = image[:, :, ::-1, :]
    return image


def tta_model(model: nn.Module, image: torch.Tensor):
    n_image = image
    h_image = horizontal_flip(image)
    v_image = vertical_flip(image)

    n_mask = model(n_image)
    h_mask = model(h_image)
    v_mask = model(v_image)

    n_mask = n_mask if not isinstance(n_mask, list) else n_mask[-1]
    h_mask = horizontal_flip(h_mask if not isinstance(h_mask, list) else h_mask[-1])
    v_mask = vertical_flip(v_mask if not isinstance(v_mask, list) else v_mask[-1])
    return (n_mask + h_mask + v_mask) / 3.0
