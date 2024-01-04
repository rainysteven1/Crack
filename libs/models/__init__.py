from .unet import *
from .resunet import *
from .deeplab import *

MODEL_DICT = {
    "attention_unet": AttentionUNet,
    "attention_unet_pp": AttentionUNet2Plus,
    "deeplabv2": DeepLabV2,
    "double_unet": DoubleUNet,
    "unet": UNet,
    "unet_pp": UNet2Plus,
    "unet_ppp": UNet3PlusDeepSupCGM,
    "resunet0": ResUNet0,
    "resunet1": ResUNet1,
    "resunet_pool": ResUNetPool,
    "resunet_pp": ResUNet2Plus,
    "transunet": TransUNet,
}
