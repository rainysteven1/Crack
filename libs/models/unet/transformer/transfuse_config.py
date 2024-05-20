import ml_collections

from ...transformer.vit_config import get_b16_config as get_config


def get_s16_config():
    config = get_config()

    config.hidden_size = config.patch_size**2 * 3 // 2

    config.transformer.pretrained_path = "resources/checkpoints/ViT-S_16.npz"
    config.transformer.mlp_dim = config.hidden_size * config.transformer.mlp_ratio
    config.transformer.n_heads = 6
    config.transformer.n_layers = 8

    config.resnet = ml_collections.ConfigDict()
    config.resnet.pretrained_path = "resources/checkpoints/resnet34-333f7ec4.pth"
    config.resnet.dims = [256, 128, 64]
    return config


def get_b16_config():
    config = get_config()

    config.resnet = ml_collections.ConfigDict()
    config.resnet.pretrained_path = "resources/checkpoints/resnet50-19c8e357.pth"
    config.resnet.dims = [1024, 512, 256]
    return config


def get_testing(config_category: str):
    config = CONFIGS.get(config_category)()

    config.classifier = "token"
    config.transformer.pretrained_path = None
    config.resnet.pretrained_path = None
    return config


CONFIGS = {
    "ViT-S_16": get_s16_config,
    "ViT-B_16": get_b16_config,
    "testing": get_testing,
}
