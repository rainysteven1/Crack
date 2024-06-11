from typing import Optional, OrderedDict

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from omegaconf import DictConfig
from scipy import ndimage

from ...utils.utils import np2th
from ..backbone.vit_resnet import ResNet
from ..modules.attention import MHSA
from ..modules.base import IntermediateSequential

__all__ = ["VisionTransformer"]


class _PatchProjection(nn.Sequential):
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        patch_size: int,
        n_patches: int,
        is_hybrid: bool,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                input_dim,
                embedding_dim,
                kernel_size=1 if is_hybrid else patch_size,
                stride=1 if is_hybrid else patch_size,
            ),
            Rearrange("b c h w -> b (h w) c", h=n_patches, w=n_patches),
        )

    def load_from(self, weights: OrderedDict, weights_key: str):
        state_dict = {
            key: weights[weights_key.format(key.split(".")[-1])]
            for key in self.state_dict().keys()
        }
        self.load_state_dict(state_dict)


class _PatchEmbedding(nn.Module):

    base_size = 16
    ratio = 16

    def __init__(
        self,
        input_dim: int,
        projection: Optional[nn.Module],
        img_size: int,
        patch_size: int,
        embedding_dim: int,
        is_cls: bool,
        dropout: Optional[float],
        backbone_config: Optional[DictConfig] = None,
    ) -> None:
        super().__init__()
        projection = projection if projection else _PatchProjection
        self.is_cls = is_cls
        self.is_hybrid = backbone_config is not None

        if self.is_hybrid:
            grid_size = backbone_config.pop("grid_size")
            patch_size = img_size // self.base_size // grid_size
            patch_size = patch_size * self.base_size
        n_patches = img_size // patch_size

        # Linear Projection
        if self.is_hybrid:
            hybrid_backbone = backbone_config.pop("hybrid_backbone")
            self.backbone = hybrid_backbone(input_dim, **backbone_config)
            input_dim = self.backbone.width * self.ratio
        self.projection = projection(
            input_dim, embedding_dim, patch_size, n_patches, self.is_hybrid
        )

        if is_cls:
            self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
            self.pos_embedding = nn.Parameter(
                torch.randn(1, 1 + n_patches**2, embedding_dim)
            )
        else:
            self.pos_embedding = nn.Parameter(
                torch.randn(1, n_patches**2, embedding_dim)
            )

        self.dropout = nn.Identity() if not dropout else nn.Dropout(dropout)

    def forward(self, input: torch.Tensor):
        x = input
        if self.is_hybrid:
            x, features = self.backbone(x)
        else:
            features = None
        x = self.projection(x)
        if self.is_cls:
            cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=input.shape[0])
            x = torch.cat((cls_tokens, x), dim=1)
        return self.dropout(x + self.pos_embedding), features

    def load_from(self, weights: OrderedDict, _: str, weights_keys: DictConfig):
        if self.is_hybrid:
            self.backbone.load_from(weights, weights_keys["backbone"])

        if hasattr(self.projection, "load_from"):
            getattr(self.projection, "load_from")(weights, weights_keys["projection"])

        with torch.no_grad():
            # cls_token
            if self.is_cls:
                self.cls_token.copy_(weights[weights_keys["cls_token"]])
            # pos_embedding
            pos_embed = weights[weights_keys["pos_embedding"]]
            if pos_embed.shape == self.pos_embedding.shape:
                self.pos_embedding.copy_(pos_embed)
            elif pos_embed.shape[1] - 1 == self.pos_embedding.shape[1]:
                pos_embed = pos_embed[:, 1:]
                self.pos_embedding.copy_(pos_embed)
            else:
                pos_embed_grid = pos_embed[:, 1:]
                ntok_new = self.pos_embedding.shape[1] - (1 if self.is_cls else 0)
                gs_old = int(np.sqrt(pos_embed_grid.shape[1]))
                gs_new = int(np.sqrt(ntok_new))
                pos_embed_grid = rearrange(
                    pos_embed_grid, "1 (h w) d -> h w d", h=gs_old, w=gs_old
                )
                pos_embed_grid = ndimage.zoom(
                    pos_embed_grid, (gs_new / gs_old, gs_new / gs_old, 1), order=1
                )
                pos_embed_grid = rearrange(
                    pos_embed_grid, "h w d -> 1 (h w) d", h=gs_new, w=gs_new
                )
                if self.is_cls:
                    self.pos_embedding[:, :1].copy_(pos_embed[:, :1])
                    self.pos_embedding[:, 1:].copy_(np2th(pos_embed_grid))
                else:
                    self.pos_embedding.copy_(np2th(pos_embed_grid))


class _MultiHeadAttention(nn.Sequential):
    def __init__(
        self,
        embedding_dim: int,
        n_heads: int,
        qkv_bias: bool,
        attn_dropout: float,
        proj_dropout: float,
    ) -> None:
        super().__init__(
            MHSA(embedding_dim, n_heads, qkv_bias, attn_dropout),
            nn.Linear(embedding_dim, embedding_dim),
            nn.Dropout(proj_dropout),
        )


class _MLPBlock(nn.Sequential):
    def __init__(self, embedding_dim: int, mlp_dim: int, dropout: float) -> None:
        super().__init__(
            nn.Linear(embedding_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embedding_dim),
            nn.Dropout(dropout),
        )


class _PreNorm(nn.Sequential):
    def __init__(self, embedding_dim: int, fn: nn.Module):
        super().__init__()

        self.norm = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.fn = fn

    def forward(self, input: torch.tensor, **kwargs):
        x = self.norm(input)
        return self.fn(x, **kwargs) + input


class _TransformerEncoderBlock(nn.Sequential):
    def __init__(
        self,
        embedding_dim: int,
        mlp_ratio: float,
        mlp_dropout: float,
        **kwargs,
    ) -> None:

        super().__init__(
            _PreNorm(
                embedding_dim,
                _MultiHeadAttention(embedding_dim, **kwargs),
            ),
            _PreNorm(
                embedding_dim,
                _MLPBlock(embedding_dim, int(embedding_dim * mlp_ratio), mlp_dropout),
            ),
        )


class _TransformerEncoder(nn.Module):
    def __init__(
        self,
        block_config: DictConfig,
        n_blocks: int,
        return_intermediate: bool,
        is_norm: bool,
    ):
        super().__init__()
        embedding_dim = block_config.get("embedding_dim")
        self.is_norm = is_norm

        self.transformer_encoder_blocks = IntermediateSequential(
            *[_TransformerEncoderBlock(**block_config) for _ in range(n_blocks)],
            return_intermediate=return_intermediate,
        )
        self.norm = (
            nn.Identity() if not is_norm else nn.LayerNorm(embedding_dim, eps=1e-6)
        )

    def forward(self, input: torch.Tensor):
        x = self.transformer_encoder_blocks(input)
        if isinstance(x, list):
            x.append(self.norm(x[-1]))
            return x
        return self.norm(x)

    def load_from(self, weights: OrderedDict, source: str, weights_keys: DictConfig):
        state_dict = dict()

        if self.is_norm:
            norm_prefix = weights_keys["norm"]
            for key in self.norm.state_dict().keys():
                state_dict[f"norm.{key}"] = weights[f"{norm_prefix}.{key}"]

        for key in self.transformer_encoder_blocks.state_dict().keys():
            new_key = f"transformer_encoder_blocks.{key}"
            str_list = key.split(".")
            layer_idx = int(str_list[1])
            block = weights_keys["block"]
            root = block["root"].format(str_list[0], str_list[-1])
            if "norm" in str_list:
                state_dict[new_key] = weights[
                    root.format(block["norm"].format(str(layer_idx + 1)))
                ]
            elif "qkv" in str_list:
                weights_key = root.format(block["qkv"])
                if source == "torchvision":
                    weights_key = "_".join(weights_key.rsplit(".", 1))
                state_dict[new_key] = weights[weights_key]
            elif "fn" in str_list:
                if layer_idx == 0:
                    state_dict[new_key] = weights[root.format(block["fn"]["proj"])]
                elif layer_idx == 1:
                    state_dict[new_key] = weights[
                        root.format(
                            block["fn"]["mlp"].format(1 if str_list[-2] == "0" else 2)
                        )
                    ]
        self.load_state_dict(state_dict)


class VisionTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        config: DictConfig,
        patch_projection: Optional[nn.Module] = None,
    ):
        super().__init__()
        pretrained: Optional[DictConfig] = config.pop("pretrained", None)

        self.patch_embedding = _PatchEmbedding(
            input_dim, patch_projection, **config.get("patch_embedding")
        )
        self.encoder = _TransformerEncoder(
            config.get("encoder_block"), **config.get("encoder")
        )

        if pretrained:
            path = pretrained.pop("path")
            self._load_from(torch.load(path), **pretrained)

    def forward(self, input: torch.Tensor):
        embedding_output, features = self.patch_embedding(input)
        output = self.encoder(embedding_output)
        return output, features

    def _load_from(self, weights: OrderedDict, source: str, weights_keys: DictConfig):
        if "model" in weights.keys():
            weights = weights.get("model")
        self.patch_embedding.load_from(weights, source, weights_keys["patch_embedding"])
        self.encoder.load_from(weights, source, weights_keys["encoder"])
