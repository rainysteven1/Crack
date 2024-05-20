from typing import Optional

import torch.nn as nn
import torch.nn.init as init


def _init_batchNorm(module):
    init.normal_(module.weight.data, 1.0, 0.02)
    init.constant_(module.bias.data, 0.0)


class _InitWeights_He:
    def __init__(self, neg_slope=0):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.BatchNorm2d):
            _init_batchNorm(module)
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
            module.weight = init.kaiming_normal_(
                module.weight, a=self.neg_slope, mode="fan_in"
            )
            if module.bias is not None:
                module.bias = init.constant_(module.bias, 0)


class _InitWeights_XavierUniform:
    def __init__(self, gain=1):
        self.gain = gain

    def __call__(self, module):
        if isinstance(module, nn.BatchNorm2d):
            _init_batchNorm(module)
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
            module.weight = init.xavier_uniform_(module.weight, self.gain)
            if module.bias is not None:
                module.bias = init.constant_(module.bias, 0)


class InitModule(nn.Module):
    def __init__(self, init_type: Optional[str] = None):
        super().__init__()

        self.init = None
        self.init_type = init_type

        if self.init_type:
            if self.init_type == "kaiming":
                self.init = _InitWeights_He()
            elif self.init_type == "xavier":
                self.init = _InitWeights_XavierUniform()

    def _initialize_weights(self):
        pass
