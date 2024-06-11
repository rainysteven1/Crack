from typing import List, Optional, OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from omegaconf import DictConfig

from ...backbone import resnet
from ...modules.attention import AttentionBlock
from ...modules.base import BasicBlock
from ...transformer.vit import VisionTransformer

__all__ = ["TransFuse"]


class _ChannelPool(nn.Module):
    def forward(self, input: torch.Tensor):
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
        self.redisual = resnet.BottleNeck(
            ch_dim1 + ch_dim2 + input_dim,
            output_dim // 4,
            output_dim // 2,
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
        is_attn: bool,
        init_type: Optional[str],
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

    def forward(self, input1: torch.Tensor, input2: Optional[torch.Tensor] = None):
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


def _final_common(
    dims: List[int], output_dim: int, ratio: int, init_type: Optional[str]
):
    return nn.Sequential(
        BasicBlock(dims[2], dims[2], init_type=init_type),
        BasicBlock(
            dims[2],
            output_dim,
            is_bn=False,
            is_relu=False,
            init_type=init_type,
        ),
        nn.Upsample(scale_factor=ratio, mode="bilinear", align_corners=True),
        nn.Sigmoid(),
    )


class TransFuse(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        drop_rate: int,
        backbone_config: DictConfig,
        config: DictConfig,
        init_type: Optional[str],
    ) -> None:
        super().__init__()
        dims = backbone_config.get("dims")
        embedding_dim = config.get("shared_params").get("embedding_dim")
        self.patch_size = config.get("patch_embedding").get("patch_size")

        self.resnet = backbone_config.get("backbone")
        self.transformer = VisionTransformer(input_dim, config)
        self.resnet_blocks = list(self.resnet.children())

        self.dropout = nn.Dropout(2 * drop_rate)
        self.up_blocks = nn.ModuleList(
            [
                _UpsampleBlock(embedding_dim, 0, dims[1], False, init_type),
                _UpsampleBlock(dims[1], 0, dims[2], False, init_type),
            ]
        )

        self.bifusion_blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        _BiFusionBlock(
                            dims[i],
                            embedding_dim if i == 0 else dims[i],
                            4 // (2**i),
                            dims[i],
                            dims[i],
                            drop_rate,
                            init_type,
                        ),
                        (
                            nn.Identity()
                            if i == 0
                            else _UpsampleBlock(
                                dims[i - 1], dims[i], dims[i], True, init_type
                            )
                        ),
                    ]
                )
                for i in range(len(self.up_blocks) + 1)
            ]
        )

        self.final_x = nn.Sequential(
            BasicBlock(dims[0], dims[2], kernel_size=1, padding=0, init_type=init_type),
            _final_common(dims, output_dim, 16, init_type),
        )
        self.final_1 = _final_common(dims, output_dim, 4, init_type)
        self.final_2 = _final_common(dims, output_dim, 4, init_type)

    def forward(self, input: torch.Tensor):
        # transformer
        x_t = self.transformer(input)
        x_t = self.dropout(
            rearrange(
                x_t[0][:, 1:],
                "b (h w) d -> b d h w",
                h=self.patch_size,
                w=self.patch_size,
            )
        )
        x_t_list = [x_t]
        for up_block in self.up_blocks:
            x_t = self.dropout(up_block(x_t))
            x_t_list.append(x_t)

        # cnn
        x_u = self.resnet_blocks[0](input)
        x_u_list = list()
        for resnet_block in self.resnet_blocks[1 : 1 + len(x_t_list)]:
            x_u = self.dropout(resnet_block(x_u))
            x_u_list.append(x_u)
        x_u_list = list(reversed(x_u_list))

        # bifusion
        x_b_list = list()
        for i, (bifusion_block, x_u, x_t) in enumerate(
            zip(self.bifusion_blocks, x_u_list, x_t_list)
        ):
            x_temp = bifusion_block[0](x_u, x_t)
            if i == 0:
                x_b = bifusion_block[1](x_temp)
            else:
                x_b = bifusion_block[1](x_b, x_temp)
            x_b_list.append(x_b)

        output_x = self.final_x(x_b_list[0])
        output_1 = self.final_1(x_t_list[-1])
        output_2 = self.final_2(x_b_list[-1])
        return [output_2, output_1, output_x]
