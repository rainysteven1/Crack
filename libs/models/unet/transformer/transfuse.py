import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from ...modules.attention import AttentionBlock
from ...modules.base import BasicBlock
from ...modules.resnet import BottleNeck, resnet34, resnet50
from ...transformer.vit import VisionTransformer


class _ChannelPool(nn.Module):
    def forward(self, input):
        return torch.cat(
            [torch.max(input, 1)[0].unsqueeze(1), torch.mean(input, 1).unsqueeze(1)],
            dim=1,
        )


class _BiFusionBlock(nn.Module):
    def __init__(
        self,
        ch_dim1: int,
        ch_dim2: int,
        r_dim: int,
        input_dim: int,
        output_dim: int,
        drop_rate: float,
        init_type: Optional[str],
    ):
        super().__init__()
        self.drop_rate = drop_rate

        # bi-linear modelling for both
        self.W_g = BasicBlock(
            ch_dim1,
            input_dim,
            kernel_size=1,
            padding=0,
            is_relu=False,
            init_type=init_type,
        )
        self.W_t = BasicBlock(
            ch_dim2,
            input_dim,
            kernel_size=1,
            padding=0,
            is_relu=False,
            init_type=init_type,
        )
        self.W = BasicBlock(input_dim, input_dim, kernel_size=3, init_type=init_type)

        # spatial attention for F_l
        self.spatial_attn = nn.Sequential(
            _ChannelPool(),
            BasicBlock(
                2,
                1,
                kernel_size=7,
                padding=3,
                is_relu=False,
                is_bias=False,
                init_type=init_type,
            ),
            nn.Sigmoid(),
        )

        # channel attention for F_g, use SE Block
        middle_dim = ch_dim2 // r_dim
        self.channel_attn = nn.Sequential(
            BasicBlock(
                ch_dim2,
                middle_dim,
                kernel_size=1,
                padding=0,
                is_bn=False,
                init_type=init_type,
            ),
            BasicBlock(
                middle_dim,
                ch_dim2,
                kernel_size=1,
                padding=0,
                is_bn=False,
                is_relu=False,
                init_type=init_type,
            ),
            nn.Sigmoid(),
        )

        is_identity = ch_dim1 + ch_dim2 + input_dim == output_dim
        self.redisual = BottleNeck(
            ch_dim1 + ch_dim2 + input_dim,
            output_dim // 4,
            output_dim // 2,
            skip_padding=0,
            is_identity=is_identity,
            reversed=True,
            init_type=init_type,
        )
        self.dropout = nn.Dropout2d(drop_rate)

    def forward(self, g: torch.Tensor, t: torch.Tensor):
        # bilinear pooling
        W_g = self.W_g(g)
        W_t = self.W_t(t)
        bp = self.W(W_g * W_t)

        # spatial attention for cnn branch
        g_in = g
        g_out = self.spatial_attn(g) * g_in

        # channel attetion for transformer branch
        t_in = t
        t = t.mean((2, 3), keepdim=True)
        t_out = self.channel_attn(t) * t_in

        fuse = self.redisual(torch.cat((g_out, t_out, bp), dim=1))
        return fuse if self.drop_rate == 0 else self.dropout(fuse)


class _UpsampleBlock(nn.Module):
    def __init__(
        self,
        input_dim1: int,
        input_dim2: int,
        output_dim: int,
        init_type: Optional[str],
        is_attn: bool = False,
    ) -> None:
        super().__init__()
        self.is_attn = is_attn

        input_dim = input_dim1 + input_dim2
        self.up_block = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_block1 = nn.Sequential(
            BasicBlock(input_dim, output_dim, init_type=init_type),
            BasicBlock(output_dim, output_dim, is_relu=False, init_type=init_type),
        )
        self.conv_block2 = BasicBlock(
            input_dim,
            output_dim,
            kernel_size=1,
            padding=0,
            is_relu=False,
            init_type=init_type,
        )
        self.attn_block = AttentionBlock(input_dim1, input_dim2, output_dim)

    def forward(self, input1: torch.Tensor, input2: torch.Tensor = None):
        x1 = self.up_block(input1)
        x2 = input2

        if x2 is not None:
            diffY = torch.tensor([input2.size()[2] - x1.size()[2]])
            diffX = torch.tensor([input2.size()[3] - x1.size()[3]])

            x1 = F.pad(
                x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
            )
            if self.is_attn:
                x2 = self.attn_block(x1, x2)
            x1 = torch.cat((x2, x1), dim=1)

        output = F.relu(self.conv_block1(x1) + self.conv_block2(x1))
        return output


class TransFuse(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        drop_rate: int,
        img_size: int,
        config: DictConfig,
        init_type: Optional[str],
    ) -> None:
        super().__init__()
        self.config = config
        dims = config.resnet.dims

        self.transformer = VisionTransformer(input_dim, img_size, config)
        self.resnet = (
            resnet34(input_dim, pretrained=True)
            if config.resnet.type == "resnet34"
            else resnet50(input_dim, pretrained=True)
        )
        self.layers = list(self.resnet.children())

        hidden_size = config["hidden_size"]
        self.up_block1 = _UpsampleBlock(hidden_size, 0, dims[1])
        self.up_block2 = _UpsampleBlock(dims[1], 0, dims[2])

        self.bifusion = _BiFusionBlock(
            dims[0], hidden_size, 4, dims[0], dims[0], drop_rate / 2, init_type
        )
        self.bifusion1 = _BiFusionBlock(
            dims[1], dims[1], 2, dims[1], dims[1], drop_rate / 2, init_type
        )
        self.bifusion1_up = _UpsampleBlock(dims[0], dims[1], dims[1], True, init_type)
        self.bifusion2 = _BiFusionBlock(
            dims[2], dims[2], 1, dims[2], dims[2], drop_rate / 2, init_type
        )
        self.bifusion2_up = _UpsampleBlock(dims[1], dims[2], dims[2], True, init_type)

        def final_common(radio: int):
            return nn.Sequential(
                BasicBlock(dims[2], dims[2], init_type=init_type),
                BasicBlock(
                    dims[2],
                    output_dim,
                    is_bn=False,
                    is_relu=False,
                    init_type=init_type,
                ),
                nn.Upsample(scale_factor=radio, mode="bilinear", align_corners=True),
                nn.Sigmoid(),
            )

        self.final_x = nn.Sequential(
            BasicBlock(dims[0], dims[2], kernel_size=1, padding=0, init_type=init_type),
            final_common(16),
        )
        self.final_1 = final_common(4)
        self.final_2 = final_common(4)

        self.drop = nn.Dropout2d(drop_rate)

        self._load_from()

    def forward(self, input: torch.Tensor):
        # transformer
        x_t = self.transformer(input)
        if isinstance(x_t, tuple):
            x_t = x_t[0]
        x_t = self._reshape_input(x_t.squeeze())
        x_t = self.drop(x_t)

        x_t_1 = self.drop(self.up_block1(x_t))
        x_t_2 = self.drop(self.up_block2(x_t_1))

        # cnn
        x_u = self.layers[0](input)

        x_u_2 = self.drop(self.layers[1](x_u))
        x_u_1 = self.drop(self.layers[2](x_u_2))
        x_u = self.drop(self.layers[3](x_u_1))

        # bifusion
        x_c = self.bifusion(x_u, x_t)

        x_c_1 = self.bifusion1(x_u_1, x_t_1)
        x_c_1 = self.bifusion1_up(x_c, x_c_1)

        x_c_2 = self.bifusion2(x_u_2, x_t_2)
        x_c_2 = self.bifusion2_up(x_c_1, x_c_2)

        # decoder
        output_x = self.final_x(x_c)
        output_1 = self.final_1(x_t_2)
        output_2 = self.final_2(x_c_2)
        outputs = [output_x, output_1, output_2]

        return outputs

    def _reshape_input(self, input: torch.Tensor):
        (n_batch, n_patch, hidden) = input.size()
        height, width = int(math.sqrt(n_patch)), int(math.sqrt(n_patch))
        x = input.permute(0, 2, 1)
        x = x.contiguous().view(n_batch, hidden, height, width)
        return x

    def _load_from(self):
        weights = np.load(self.config.transformer.pretrained_path)
        with torch.no_grad():
            self.transformer.load_from(weights)
