import ml_collections


def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()

    config.patch_size = 16
    config.patches = ml_collections.ConfigDict(
        {"size": (config.patch_size, config.patch_size)}
    )
    config.hidden_size = config.patch_size**2 * 3
    config.classifier = "seg"

    config.transformer = ml_collections.ConfigDict()
    config.transformer.pretrained_path = "resources/checkpoints/ViT-B_16.npz"
    config.transformer.mlp_ratio = 4
    config.transformer.mlp_dim = config.hidden_size * config.transformer.mlp_ratio
    config.transformer.n_heads = 12
    config.transformer.n_layers = 12
    config.transformer.attn_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1

    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 1
    config.n_skip = 0
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


def get_b32_config():
    """Returns the ViT-B/32 configuration."""
    config = get_b16_config()

    config.patch_size = 32
    config.patches = ml_collections.ConfigDict(
        {"size": (config.patch_size, config.patch_size)}
    )

    config.transformer.pretrained_path = "resources/checkpoints/ViT-B_32.npz"
    return config


def get_l16_config():
    """Returns the ViT-L/16 configuration."""
    config = ml_collections.ConfigDict()

    config.patch_size = 16
    config.patches = ml_collections.ConfigDict(
        {"size": (config.patch_size, config.patch_size)}
    )
    config.hidden_size = 1024
    config.classifier = "seg"

    config.transformer = ml_collections.ConfigDict()
    config.transformer.pretrained_path = "resources/checkpoints/ViT-L_16.npz"
    config.transformer.mlp_ratio = 4
    config.transformer.mlp_dim = config.hidden_size * config.transformer.mlp_ratio
    config.transformer.n_heads = 16
    config.transformer.n_layers = 24
    config.transformer.attn_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1

    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 1
    return config


def get_l32_config():
    """Returns the ViT-L/32 configuration."""
    config = get_l16_config()

    config.patch_size = 32
    config.patches = ml_collections.ConfigDict(
        {"size": (config.patch_size, config.patch_size)}
    )
    config.transformer.pretrained_path = "resources/checkpoints/ViT-L_32.npz"
    return config


def get_testing(config_category: str):
    config = CONFIGS.get(config_category)()

    config.classifier = "token"

    config.transformer.pretrained_path = None
    return config


CONFIGS = {
    "ViT-B_16": get_b16_config,
    "R50-ViT-B_16": get_r50_b16_config,
    "ViT-B_32": get_b32_config,
    "ViT-L_16": get_l16_config,
    "ViT-L_32": get_l32_config,
    "testing": get_testing,
}
