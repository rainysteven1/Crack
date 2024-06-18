from typing import Callable, List, Optional, OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from omegaconf import DictConfig

from ..backbone.mobilenet import InvertedResidual
from ..modules.base import BasicBlock
from ._base import MLPBlock, PreNorm

__all__ = ["MobileViT", "MobileViTBlock"]


class _MultiHeadAttention(nn.Module):

    def __init__(
        self,
        input_dim: int,
        n_heads: int,
        head_dim: Optional[int],
        qkv_bias: bool = True,
        attn_dropout: Optional[float] = None,
        proj_dropout: Optional[float] = None,
    ):
        super().__init__()
        head_dim = head_dim if head_dim else input_dim // n_heads
        inner_dim = n_heads * head_dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.scale = input_dim**-0.5

        self.query = nn.Linear(input_dim, inner_dim, bias=qkv_bias)
        self.key = nn.Linear(input_dim, inner_dim, bias=qkv_bias)
        self.value = nn.Linear(input_dim, inner_dim, bias=qkv_bias)

        self.attn_dropout = nn.Dropout(attn_dropout) if attn_dropout else nn.Identity()

        self.proj_out = nn.Sequential(
            nn.Linear(inner_dim, input_dim),
            nn.Dropout(proj_dropout) if proj_dropout else nn.Identity(),
        )

    def forward(self, input: torch.Tensor, mask: Optional[torch.Tensor] = None):
        qkv = [self.query(input), self.key(input), self.value(input)]
        q, k, v = map(
            lambda t: rearrange(
                t, "b p n (h d) -> b p h n d", h=self.n_heads, d=self.head_dim
            ),
            qkv,
        )

        energy = torch.einsum(
            "bphqd, bphkd -> bphqk", q, k
        )  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        attn = F.softmax(energy * self.scale, dim=-1)
        attn = self.attn_dropout(attn)

        output = torch.einsum("bphal, bphlv -> bphav", attn, v)
        output = rearrange(output, "b p h n d -> b p n (h d)")
        return self.proj_out(output)


class _TransformerEncoderBlock(nn.Sequential):
    def __init__(
        self, embedding_dim: int, mlp_dim: int, mlp_dropout: float, **kwargs
    ) -> None:
        proj_dropout = kwargs.get("proj_dropout")

        super().__init__(
            PreNorm(
                embedding_dim,
                _MultiHeadAttention(embedding_dim, **kwargs),
            ),
            PreNorm(
                embedding_dim,
                MLPBlock(embedding_dim, mlp_dim, mlp_dropout, proj_dropout, nn.SiLU),
            ),
        )


class _TransformerEncoder(nn.Sequential):

    def __init__(
        self, n_blocks: int, embedding_dim: int, mlp_ratio: int, config: DictConfig
    ):
        super().__init__(
            *[
                _TransformerEncoderBlock(
                    embedding_dim, embedding_dim * mlp_ratio, **config
                )
                for _ in range(n_blocks)
            ],
            nn.LayerNorm(embedding_dim, eps=1e-6),
        )


class MobileViTBlock(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        patch_size: int,
        block_config: List[int],
        transformer_config: DictConfig,
        relu_type: Callable = nn.SiLU,
        init_type: Optional[str] = None,
    ) -> None:
        super().__init__()
        transformer_dim = block_config[1]
        self.patch_size = patch_size

        # Local representation
        self.local_representation = nn.Sequential(
            BasicBlock(
                input_dim,
                input_dim,
                is_bias=False,
                relu_type=relu_type,
                init_type=init_type,
            ),
            BasicBlock(
                input_dim,
                transformer_dim,
                kernel_size=1,
                padding=0,
                is_bias=False,
                is_bn=False,
                is_relu=False,
                init_type=init_type,
            ),
        )

        # Global representations
        self.transformer = _TransformerEncoder(*block_config, config=transformer_config)

        # Fusion
        self.fusion_block1 = BasicBlock(
            transformer_dim,
            input_dim,
            kernel_size=1,
            padding=0,
            is_bias=False,
            relu_type=relu_type,
            init_type=init_type,
        )
        self.fusion_block2 = BasicBlock(
            2 * input_dim,
            output_dim,
            is_bias=False,
            relu_type=relu_type,
            init_type=init_type,
        )

    def forward(self, input: torch.Tensor):
        _, _, h, w = input.shape
        x = rearrange(
            self.local_representation(input),
            "b d (ph h) (pw w) -> b (ph pw) (h w) d",
            h=self.patch_size,
            w=self.patch_size,
        )
        x = rearrange(
            self.transformer(x),
            "b (ph pw) (h w) d -> b d (ph h) (pw w)",
            ph=h // self.patch_size,
            pw=w // self.patch_size,
            h=self.patch_size,
            w=self.patch_size,
        )
        return self.fusion_block2(torch.cat((self.fusion_block1(x), input), dim=1))


class MobileViT(nn.Sequential):

    relu_type = nn.SiLU

    def __init__(self, input_dim: int, config: DictConfig) -> None:
        dims = config.get("dims")
        dims.insert(0, dims[0])
        redisual_config = config.get("redisual")
        trunk_config = config.get("trunk")
        pretrained = config.get("pretrained", None)
        self.dilation = 1

        strides = redisual_config.pop("strides")
        n_stages_list = redisual_config.pop("n_stages_list")
        self.length = len(strides)
        inverted_residuals = [
            self._make_inverted_residuals(
                dims[i], dims[i + 1], stride, n_stages, **redisual_config
            )
            for i, (stride, n_stages) in enumerate(zip(strides, n_stages_list))
        ]

        strides = trunk_config.pop("strides")
        is_dilates = trunk_config.pop("is_dilates")
        transformer_config = trunk_config.pop("transformer")
        patch_size = transformer_config.pop("patch_size")
        block_configs = transformer_config.pop("block_configs")
        mobilevit_blocks = [
            self._make_mobilevit_blocks(
                dims[i],
                dims[i + 1],
                stride,
                is_dilate,
                patch_size=patch_size,
                block_config=block_config,
                transformer_config=transformer_config,
                **redisual_config,
            )
            for i, (stride, is_dilate, block_config) in enumerate(
                zip(strides, is_dilates, block_configs), start=self.length
            )
        ]

        last_output_dim = dims[-1] * config.get("last_output_dim_ratio")
        blocks = [
            BasicBlock(
                input_dim, dims[0], stride=2, is_bias=False, relu_type=self.relu_type
            ),
            BasicBlock(
                dims[-1],
                last_output_dim,
                kernel_size=1,
                padding=0,
                is_bias=False,
                relu_type=self.relu_type,
            ),
        ]
        blocks = blocks[:1] + inverted_residuals + mobilevit_blocks + blocks[1:]

        super().__init__(*blocks)

        if pretrained:
            path = pretrained.pop("path")
            self._load_from(torch.load(path), **pretrained)

    def _make_inverted_residuals(
        self, input_dim: int, output_dim: int, stride: int, n_stages: int, ratio: int
    ) -> nn.Sequential:
        stages = list()
        for i in range(n_stages):
            s = stride if i == 0 else 1
            stages.append(
                InvertedResidual(
                    input_dim,
                    output_dim,
                    stride=s,
                    ratio=ratio,
                    relu_type=self.relu_type,
                )
            )
            input_dim = output_dim
        return nn.Sequential(*stages)

    def _make_mobilevit_blocks(
        self,
        input_dim: int,
        output_dim: int,
        stride: int,
        is_dilate: bool,
        ratio: int,
        patch_size: int,
        block_config: List[int],
        transformer_config: DictConfig,
    ) -> nn.Sequential:
        stages = list()
        if stride == 2:
            if is_dilate:
                self.dilation *= 2
                stride = 1
            stages.append(
                InvertedResidual(
                    input_dim,
                    output_dim,
                    stride=stride,
                    dilation=self.dilation,
                    ratio=ratio,
                    relu_type=self.relu_type,
                )
            )
            input_dim = output_dim

        stages.append(
            MobileViTBlock(
                input_dim,
                output_dim,
                patch_size,
                block_config,
                transformer_config,
                self.relu_type,
            )
        )
        return nn.Sequential(*stages)

    def _load_from(self, weights: OrderedDict, weights_keys: DictConfig):
        state_dict = dict()
        for key in self.state_dict().keys():
            str_list = key.split(".")
            block_idx = int(str_list[0])
            root = weights_keys["root"].format(
                "convolution" if str_list[-2] == "0" else "normalization",
                str_list[-1],
            )

            if not str_list[1].isnumeric():
                weights_key = root.format(
                    weights_keys["stem" if block_idx == 0 else "last_conv"]
                )
            else:
                block_idx = str(block_idx - 1)
                if str_list[2].isnumeric():
                    keys_dict = weights_keys["inverted_residual"]
                    blocks = keys_dict["blocks"]
                    weights_key = root.format(
                        keys_dict["root"].format(
                            block_idx,
                            (
                                keys_dict["downsample"]
                                if int(block_idx) >= self.length
                                else ".".join([keys_dict["normal"], str_list[1]])
                            ),
                            blocks[int(str_list[2])],
                        )
                    )
                else:
                    keys_dict = weights_keys["mobilevit_block"]
                    block_root = keys_dict["root"].format(block_idx)
                    if (
                        "local_representation" not in str_list
                        and "transformer" not in str_list
                    ):
                        weights_key = root.format(
                            block_root.format(keys_dict[str_list[2]])
                        )
                    else:
                        layer_idx = int(str_list[3])
                        if "local_representation" in str_list:
                            weights_key = root.format(
                                block_root.format(
                                    keys_dict["local_representation"][layer_idx]
                                )
                            )
                        elif "transformer" in str_list:
                            keys_dict = keys_dict["transformer"]
                            if "norm" not in str_list and "fn" not in str_list:
                                weights_key = keys_dict["layernorm"].format(
                                    block_idx, str_list[-1]
                                )
                            else:
                                encoder_block_idx = int(str_list[4])
                                root = keys_dict["root"].format(
                                    block_idx, layer_idx, str_list[-1]
                                )
                                if "norm" in str_list:
                                    weights_key = root.format(
                                        keys_dict["norms"][encoder_block_idx]
                                    )
                                elif "fn" in str_list:
                                    if encoder_block_idx == 0:
                                        keys_dict = keys_dict["attn"]
                                        weights_key = root.format(
                                            keys_dict["proj_out"]
                                            if "proj_out" in str_list
                                            else keys_dict["attention"].format(
                                                str_list[-2]
                                            )
                                        )
                                    else:
                                        weights_key = root.format(
                                            keys_dict["mlp"][
                                                0 if str_list[-2] == "0" else 1
                                            ]
                                        )

            state_dict[key] = weights[weights_key]
        self.load_state_dict(state_dict)
