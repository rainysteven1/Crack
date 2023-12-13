from .unet import *
from .resunet import *

MODEL_DICT = {
    "attention_unet": AttentionUNet,
    "attention_unet_pp": AttentionUNet2Plus,
    "double_unet": DoubleUNet,
    "unet": UNet,
    "unet_pp": UNet2Plus,
    "unet_ppp": UNet3Plus,
    "resunet0": ResUNet0,
    "resunet1": ResUNet1,
    "resunet_pool": ResUNetPool,
    "resunet_pp": ResUNet2Plus,
}
