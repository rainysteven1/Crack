from ..modules.transformer import transformer_config


def get_b16_config():
    """
    Returns the ViT-B/16 configuration
    """
    config = transformer_config.get_b16_config()

    config.hidden_size = 1024

    config.transformer.mlp_dim = config.hidden_size * 4
    config.transformer.n_heads = 16
    config.transformer.n_layers = 24
    config.transformer.attn_dropout_rate = 0.1
    config.transformer.dropout_rate = 0.1

    config.decoder_classifier = "mla"
    config.decoder_channels = (512, 256, 128, 64)
    return config


CONFIGS = {
    "ViT-B_16": get_b16_config,
}
