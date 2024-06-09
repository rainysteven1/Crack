from typing import Optional, OrderedDict

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from omegaconf import DictConfig
from scipy import ndimage

from ...utils.utils import np2th
from ..modules.attention import MHSA
from ..modules.base import IntermediateSequential

__all__ = ["VisionTransformer"]


class _PatchEmbedding(nn.Module):

    def __init__(
        self,
        input_dim: int,
        projection: nn.Module,
        img_size: int,
        patch_size: int,
        embedding_dim: int,
        is_cls: bool,
        dropout: Optional[float],
    ) -> None:
        super().__init__()
        n_patch = img_size // patch_size
        self.is_cls = is_cls

        # Linear Projection
        self.projection = projection(input_dim, embedding_dim, patch_size, n_patch)
        if is_cls:
            self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
            self.pos_embedding = nn.Parameter(
                torch.randn(1, 1 + n_patch**2, embedding_dim)
            )
        else:
            self.pos_embedding = nn.Parameter(torch.randn(1, n_patch**2, embedding_dim))

        self.dropout = nn.Identity() if not dropout else nn.Dropout(dropout)

    def forward(self, input: torch.Tensor):
        n_batch = input.shape[0]
        x = self.projection(input)
        if len(x.shape) == 4:
            x = rearrange(x, "b n h w -> b (h w) n")
        if self.is_cls:
            cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=n_batch)
            x = torch.cat((cls_tokens, x), dim=1)
        return self.dropout(x + self.pos_embedding)

    def load_from(self, weights: OrderedDict, _: str, weights_keys: DictConfig):
        if hasattr(self.projection, "load_from"):
            getattr(self.projection, "load_from")(weights)

        with torch.no_grad():
            # cls_token
            if self.is_cls:
                self.cls_token.copy_(weights[weights_keys["cls_token"]])
            # pos_embedding
            pos_embed = weights[weights_keys["pos_embedding"]]
            if pos_embed.size() == self.pos_embedding.size():
                self.pos_embedding.copy_(pos_embed)
            elif pos_embed.size()[1] - 1 == self.pos_embedding.size()[1]:
                pos_embed = pos_embed[:, 1:]
                self.pos_embedding.copy_(pos_embed)
            else:
                ntok_new = self.pos_embedding.size(1) - 1
                pos_embed_grid = pos_embed[0, 1:]
                gs_old = int(np.sqrt(len(pos_embed_grid)))
                gs_new = int(np.sqrt(ntok_new))
                pos_embed_grid = pos_embed_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                pos_embed_grid = ndimage.zoom(pos_embed_grid, zoom, order=1)  # th2np
                pos_embed_grid = pos_embed_grid.reshape(1, gs_new * gs_new, -1)
                if hasattr(self, "cls_token"):
                    self.pos_embedding[:, :1].copy_(pos_embed[:, :1])
                    self.pos_embedding[:, 1:].copy_(np2th(pos_embed_grid))


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


class _PreNorm(nn.Sequential):
    def __init__(self, embedding_dim: int, fn: nn.Module):
        super().__init__()

        self.norm = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.fn = fn

    def forward(self, input: torch.tensor, **kwargs):
        x = self.norm(input)
        return self.fn(x, **kwargs) + input


class _MLPBlock(nn.Sequential):
    def __init__(self, embedding_dim: int, mlp_dim: int, dropout: float) -> None:
        super().__init__(
            nn.Linear(embedding_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embedding_dim),
            nn.Dropout(dropout),
        )


class _TransformerEncoderBlock(nn.Sequential):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
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
                _MLPBlock(embedding_dim, mlp_dim, mlp_dropout),
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
            norm_prefix = weights_keys["encoder_norm"]
            for key in self.norm.state_dict().keys():
                state_dict[f"norm.{key}"] = weights[f"{norm_prefix}.{key}"]

        for key in self.transformer_encoder_blocks.state_dict().keys():
            new_key = f"transformer_encoder_blocks.{key}"
            str_list = key.split(".")
            layer_idx = int(str_list[1])
            block_dict = weights_keys["encoder_block"]
            root = block_dict["root"].format(str_list[0], str_list[-1])
            if "norm" in str_list:
                state_dict[new_key] = weights[
                    root.format(block_dict["norm"].format(str(layer_idx + 1)))
                ]
            elif "qkv" in str_list:
                weights_key = root.format(block_dict["qkv"])
                if source == "torchvision":
                    weights_key = "_".join(weights_key.rsplit(".", 1))
                state_dict[new_key] = weights[weights_key]
            elif "fn" in str_list:
                if layer_idx == 0:
                    state_dict[new_key] = weights[root.format(block_dict["fn"]["proj"])]
                elif layer_idx == 1:
                    state_dict[new_key] = weights[
                        root.format(
                            block_dict["fn"]["mlp"].format(
                                1 if str_list[-2] == "0" else 2
                            )
                        )
                    ]
        self.load_state_dict(state_dict)


class VisionTransformer(nn.Sequential):
    def __init__(
        self,
        input_dim: int,
        config: DictConfig,
        patch_projection: nn.Module,
    ):
        pretrained: Optional[DictConfig] = config.pop("pretrained", None)

        super().__init__(
            _PatchEmbedding(
                input_dim, patch_projection, **config.get("patch_embedding")
            ),
            _TransformerEncoder(config.get("encoder_block"), **config.get("encoder")),
        )

        if pretrained:
            path = pretrained.pop("path")
            self._load_from(torch.load(path), **pretrained)

    def _load_from(self, weights: OrderedDict, source: str, weights_keys: DictConfig):
        for _, module in self.named_children():
            module.load_from(weights, source, weights_keys)
