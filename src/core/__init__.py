from .unet import *
from .resunet import *

MODEL_DICT = {
    "unet": UNet,
    "unet_pp": UNet2Plus,
    "double_unet": DoubleUNet,
    "resunet0": ResUNet0,
    "resunet1": ResUNet1,
    "resunet_pool": ResUNetPool,
    "resunet_pp": ResUNet2Plus,
}
