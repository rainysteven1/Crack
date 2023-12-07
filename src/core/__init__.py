from .unet import *
from .resunet import *

MODEL_DICT = {
    "unet": UNet,
    "double_unet": DoubleUNet,
    "resunet0": ResUNet0,
    "resunet1": ResUNet1,
    "resunet_pool": ResUNetPool,
    "resunet_plusplus": ResUNetPlusPlus,
}
