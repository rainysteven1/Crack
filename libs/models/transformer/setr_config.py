import ml_collections

from .vit_config import get_b16_config as get_config


def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = get_config()

    config.hidden_size = 1024

    config.transformer.mlp_dim = config.hidden_size * 4
    config.transformer.n_heads = 16
    config.transformer.n_layers = 24
    config.transformer.attn_dropout_rate = 0.1

    config.decoder_classifier = "PUP"
    config.decoder_channels = (256, 256, 256, 256)
    return config


def get_r50_b16_config():
    """Returns the Resnet50 + ViT-B/16 configuration."""
    config = get_b16_config()

    config.transformer.pretrained_path = "resources/checkpoints/R50+ViT-B_16.npz"

    config.patches.grid = (16, 16)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1

    config.skip_channels = [512, 256, 64, 16]
    return config


CONFIGS = {
    "ViT-B_16": get_b16_config,
    "R50-VIT-B_16": get_r50_b16_config,
}
