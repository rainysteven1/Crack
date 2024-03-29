import numpy as np
import os
import torch
import torch.nn as nn

from ml_collections import ConfigDict
from scipy import ndimage
from torch.nn.modules.utils import _pair
from typing import Optional

from .transformer_resnet import ResNetV2
from ..base import Conv2dSame
from ...utils import np2th

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


class _Embeddings(nn.Module):
    """
    Construct the embeddings from patch, position embeddings
    """

    def __init__(
        self,
        input_dim: int,
        image_size: int,
        config: ConfigDict,
    ):
        super().__init__()
        image_size = _pair(image_size)
        patch_size = config.get("patch_size")
        embeddings_dim = config.get("hidden_size")
        n_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)

        if config.patches.get("grid") is not None:  # ResNet
            grid_size = config.patches["grid"]
            patch_size = (
                image_size[0] // 16 // grid_size[0],
                image_size[1] // 16 // grid_size[1],
            )
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (image_size[0] // patch_size_real[0]) * (
                image_size[1] // patch_size_real[1]
            )
            self.hybrid = True
            self.hybrid_model = ResNetV2(
                block_units=config.resnet.num_layers,
                width_factor=config.resnet.width_factor,
            )
            input_dim = self.hybrid_model.width * 16
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (image_size[0] // patch_size[0]) * (
                image_size[1] // patch_size[1]
            )
            self.hybrid = False

        self.classifier = config.get("classifier")
        self.patch_embeddings = Conv2dSame(
            input_dim, embeddings_dim, kernel_size=patch_size, stride=patch_size
        )
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, n_patches, embeddings_dim)
        )
        self.dropout = nn.Dropout(config.transformer["dropout_rate"])

    def forward(self, input):
        x = input
        features = None
        if self.hybrid:
            y = self.hybrid_model(input)
            x = y[0]
            features = y[1:]
        x = self.patch_embeddings(x)
        x = x.flatten(2).transpose(-1, -2)
        output = self.dropout(x + self.position_embeddings).unsqueeze(-1)
        return (output, *features) if features else output

    def load_from(self, weights):
        self.patch_embeddings.weight.copy_(
            np2th(weights["embedding/kernel"], conv=True)
        )
        self.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

        posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

        posemb_new = self.position_embeddings
        if posemb.size() == posemb_new.size():
            self.position_embeddings.copy_(posemb)
        elif posemb.size()[1] - 1 == posemb_new.size()[1]:
            posemb = posemb[:, 1:]
            self.position_embeddings.copy_(posemb)
        else:
            ntok_new = posemb_new.size(1)
            if self.classifier == "seg":
                _, posemb_grid = posemb[:, :1], posemb[0, 1:]
            gs_old = int(np.sqrt(len(posemb_grid)))
            gs_new = int(np.sqrt(ntok_new))
            posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
            zoom = (gs_new / gs_old, gs_new / gs_old, 1)
            posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
            posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
            posemb = posemb_grid
            self.position_embeddings.copy_(np2th(posemb))

        if self.hybrid:
            self.hybrid_model.root.get_submodule("0").weight.copy_(
                np2th(weights["conv_root/kernel"], conv=True)
            )
            self.hybrid_model.root.get_submodule("1").weight.copy_(
                np2th(weights["gn_root/scale"]).view(-1)
            )
            self.hybrid_model.root.get_submodule("1").bias.copy_(
                np2th(weights["gn_root/bias"]).view(-1)
            )
            for bname, block in self.hybrid_model.body.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=bname, n_unit=uname)


class _Attention(nn.Module):
    def __init__(self, input_dim: int, config: ConfigDict):
        super().__init__()
        attn_dropout_rate = config["transformer"]["attn_dropout_rate"]
        self.n_heads = config["transformer"]["n_heads"]
        self.head_size = input_dim // self.n_heads
        self.all_head_size = self.n_heads * self.head_size

        self.query = nn.Linear(input_dim, self.all_head_size)
        self.key = nn.Linear(input_dim, self.all_head_size)
        self.value = nn.Linear(input_dim, self.all_head_size)

        self.proj = nn.Linear(input_dim, input_dim)
        self.attn_dropout = nn.Dropout(attn_dropout_rate)
        self.proj_dropout = nn.Dropout(attn_dropout_rate)
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.n_heads,
            self.head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input):
        x1 = self.transpose_for_scores(self.query(input))
        x2 = self.transpose_for_scores(self.key(input))
        x3 = self.transpose_for_scores(self.value(input))

        attention_scores = torch.matmul(x1, x2.transpose(-1, -2))
        attention_scores = attention_scores / np.sqrt(self.head_size)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, x3)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        proj_probs = self.proj(context_layer)
        proj_probs = self.proj_dropout(proj_probs)
        return proj_probs


class _MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        output_dim = output_dim or input_dim
        hidden_dim = hidden_dim or input_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias)
        nn.init.normal_(self.fc2.bias)

    def forward(self, input):
        x = self.fc1(input)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        output = self.dropout(x)
        return output


class _TransLayer(nn.Module):
    def __init__(self, input_dim: int, config: ConfigDict) -> None:
        super().__init__()

        self.hidden_size = input_dim
        self.attn = _Attention(input_dim, config)
        self.attn_norm = nn.LayerNorm(input_dim, eps=1e-6)
        self.mlp = _MLP(input_dim, config["transformer"]["mlp_dim"])
        self.mlp_norm = nn.LayerNorm(input_dim, eps=1e-6)
        self.dropout = nn.Dropout(0.0)

    def forward(self, input):
        h = input
        x = self.attn_norm(input)
        x = self.attn(x)
        x = self.dropout(x) + h

        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        output = self.dropout(x) + h
        return output

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = (
                np2th(weights[os.path.join(ROOT, ATTENTION_Q, "kernel")])
                .view(self.hidden_size, self.hidden_size)
                .t()
            )
            key_weight = (
                np2th(weights[os.path.join(ROOT, ATTENTION_K, "kernel")])
                .view(self.hidden_size, self.hidden_size)
                .t()
            )
            value_weight = (
                np2th(weights[os.path.join(ROOT, ATTENTION_V, "kernel")])
                .view(self.hidden_size, self.hidden_size)
                .t()
            )
            out_weight = (
                np2th(weights[os.path.join(ROOT, ATTENTION_OUT, "kernel")])
                .view(self.hidden_size, self.hidden_size)
                .t()
            )

            query_bias = np2th(weights[os.path.join(ROOT, ATTENTION_Q, "bias")]).view(
                -1
            )
            key_bias = np2th(weights[os.path.join(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[os.path.join(ROOT, ATTENTION_V, "bias")]).view(
                -1
            )
            out_bias = np2th(weights[os.path.join(ROOT, ATTENTION_OUT, "bias")]).view(
                -1
            )

            mlp_weight_0 = np2th(weights[os.path.join(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[os.path.join(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[os.path.join(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[os.path.join(ROOT, FC_1, "bias")]).t()

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.proj.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.proj.bias.copy_(out_bias)

            self.mlp.fc1.weight.copy_(mlp_weight_0)
            self.mlp.fc2.weight.copy_(mlp_weight_1)
            self.mlp.fc1.bias.copy_(mlp_bias_0)
            self.mlp.fc2.bias.copy_(mlp_bias_1)

            self.attn_norm.weight.copy_(
                np2th(weights[os.path.join(ROOT, ATTENTION_NORM, "scale")])
            )
            self.attn_norm.bias.copy_(
                np2th(weights[os.path.join(ROOT, ATTENTION_NORM, "bias")])
            )
            self.mlp_norm.weight.copy_(
                np2th(weights[os.path.join(ROOT, MLP_NORM, "scale")])
            )
            self.mlp_norm.bias.copy_(
                np2th(weights[os.path.join(ROOT, MLP_NORM, "bias")])
            )


class _TransEncoder(nn.Module):
    """
    Transformer Encoder
    """

    def __init__(self, config: ConfigDict, return_intermediate: bool):
        super().__init__()
        embedding_dim = config.get("hidden_size")
        n_layers = config["transformer"].get("n_layers")
        self.intermediate_indicies = [
            config["transformer"]["n_heads"] / 4 * i - 1 for i in range(1, 5)
        ]
        self.return_intermediate = return_intermediate

        layer_list = [_TransLayer(embedding_dim, config) for _ in range(n_layers)]

        self.layers = (
            nn.ModuleList(layer_list)
            if return_intermediate
            else nn.Sequential(*layer_list)
        )
        self.norm = nn.LayerNorm(embedding_dim, eps=1e-6)

    def forward(self, input):
        if not self.return_intermediate:
            return self.norm(self.layers(input))
        intermediate_outputs = list()
        x = input
        for idx, module in enumerate(self.layers):
            x = module(x)
            if idx in self.intermediate_indicies:
                intermediate_outputs.append(x)
        return tuple(intermediate_outputs)

    def load_from(self, weights):
        self.norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
        self.norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

        for _, block in self.named_children():
            for uname, unit in block.named_children():
                unit.load_from(weights, n_block=uname)


class TransModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        img_size: int,
        config: ConfigDict,
        return_intermediate: bool = False,
    ):
        super().__init__()
        self.return_intermediate = return_intermediate

        self.embeddings = _Embeddings(input_dim, img_size, config)
        self.encoder = _TransEncoder(config, return_intermediate)

    def forward(self, input):
        x = self.embeddings(input)
        if isinstance(x, tuple):
            features = list(x[1:])
            x = x[0]
        else:
            features = None
        x = x.squeeze()
        x = self.encoder(x)
        if not self.return_intermediate:
            output = x.unsqueeze(-1)
            return (output, *features) if features is not None else output
        outputs = [output.unsqueeze(-1) for output in x]
        outputs = (outputs + features) if features is not None else outputs
        return tuple(outputs)

    def load_from(self, weights):
        self.embeddings.load_from(weights)
        self.encoder.load_from(weights)
