from .unet import *
from .resunet import *
from .resunet_pp import *

MODEL_DICT = {
    "unet": UNet,
    "double_unet": DoubleUNet,
    "resunet0": ResUNet0,
    "resunet1": ResUNet1,
    "resunet_pool": ResUNetPool,
    "resunet++": ResUNetPlusPlus,
}
