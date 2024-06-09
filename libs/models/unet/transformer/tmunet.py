from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from omegaconf import DictConfig

from ...backbone.resnet import resnet50
from ...modules.base import BasicBlock, OutputBlock, SqueezeExciteBlock
from ...transformer.vit import VisionTransformer

__all__ = ["TMUNet"]

_UPSAMPLE_TYPES = Literal["conv_transpose", "bilinear"]


class _DoubleConv(nn.Sequential):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        middle_dim: Optional[int] = None,
        is_pool: bool = False,
    ) -> None:
        middle_dim = middle_dim or output_dim
        layers = [BasicBlock(input_dim, middle_dim), BasicBlock(middle_dim, output_dim)]
        if is_pool:
            layers.append(nn.MaxPool2d(kernel_size=2))

        super().__init__(*layers)


class _UNetDecoderBlock(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        upsample_type: _UPSAMPLE_TYPES = "conv_transpose",
        up_conv_input_dim: Optional[int] = None,
        up_conv_output_dim: Optional[int] = None,
    ):
        super().__init__()

        if upsample_type == "conv_transpose":
            up_conv_input_dim = up_conv_input_dim or input_dim
            up_conv_output_dim = up_conv_output_dim or output_dim
            self.upsample = nn.ConvTranspose2d(
                up_conv_input_dim, up_conv_output_dim, kernel_size=2, stride=2
            )
        if upsample_type == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(input_dim, output_dim, kernel_size=1),
            )
        self.conv_block = _DoubleConv(input_dim, output_dim)

    def forward(self, input: torch.Tensor, skip: torch.Tensor):
        x = self.upsample(input)
        return self.conv_block(torch.cat((x, skip), dim=1))


class _EncoderPatch(nn.Sequential):
    def __init__(self, input_dim: int, embedding_dim: int) -> None:
        super().__init__(
            _DoubleConv(input_dim, 128, is_pool=True),
            _DoubleConv(128, 256, is_pool=True),
            _DoubleConv(256, embedding_dim, is_pool=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )


class _PatchProjection(nn.Sequential):
    def __init__(
        self, input_dim: int, embedding_dim: int, patch_size: int, n_patch: int
    ) -> None:
        super().__init__(
            Rearrange(
                "b c (ph h) (pw w) -> b c (ph pw) h w",
                c=input_dim,
                h=patch_size,
                w=patch_size,
                ph=n_patch,
                pw=n_patch,
            ),
            Rearrange("b c p h w -> (b p) c h w"),
            _EncoderPatch(input_dim, embedding_dim),
            Rearrange("(b p) d -> b p d", p=n_patch**2),
        )


class _DependencyMap(nn.Module):

    def __init__(
        self,
        output_dim: int,
        embedding_dim: int,
        img_size: int,
        patch_size: int,
        is_cls: bool,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.n_patch = img_size // patch_size
        self.is_cls = is_cls

        self.gpool = nn.AdaptiveAvgPool1d(1)
        self.output_block1 = nn.Sequential(
            BasicBlock(
                embedding_dim, output_dim, kernel_size=1, padding=0, is_relu=False
            ),
            nn.Sigmoid(),
        )
        self.output_block2 = nn.Sigmoid()

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor]:
        x = self.gpool(input)
        if self.is_cls:
            input = input[:, :-1, :]
            x = x[:, :-1, :]

        coeff = rearrange(
            input, "b (ph pw) c -> b c ph pw", ph=self.n_patch, pw=self.n_patch
        )
        coeff = repeat(
            coeff,
            "b c ph pw -> b c (ph h) (pw w)",
            h=self.patch_size,
            w=self.patch_size,
        )

        coeff2 = rearrange(
            x, "b (ph pw) 1 -> b 1 ph pw", ph=self.n_patch, pw=self.n_patch
        )
        coeff2 = repeat(
            coeff2,
            "b 1 ph pw -> b 1 (ph h) (pw w)",
            h=self.patch_size,
            w=self.patch_size,
        )

        global_contexual = self.output_block1(coeff)
        regional_distribution = self.output_block2(coeff2)
        return global_contexual, regional_distribution, self.output_block2(x)


class TMUNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, config: DictConfig) -> None:
        super().__init__()
        resnet = resnet50(input_dim, pretrained=True)
        resnet_blocks = list(resnet.children())
        depth = len(resnet_blocks)
        decoder_dims = (
            [input_dim]
            + resnet.dims[:2]
            + [dim * resnet.block_type.expansion for dim in resnet.dims]
        )
        length = len(decoder_dims)

        self.input_block = list(resnet_blocks[0].children())[0]
        self.input_block_pool = list(resnet_blocks[0].children())[1]
        self.unet_encoder = nn.ModuleList(resnet_blocks[1:])
        self.bridge = _DoubleConv(decoder_dims[-1], decoder_dims[-1])
        self.unet_decoder = nn.ModuleList(
            [
                _UNetDecoderBlock(decoder_dims[i], decoder_dims[i - 1])
                for i in range(length - 1, length - 4, -1)
            ]
            + [
                _UNetDecoderBlock(
                    input_dim=decoder_dims[i] + decoder_dims[i - 1],
                    output_dim=decoder_dims[i],
                    up_conv_input_dim=decoder_dims[i + 1],
                    up_conv_output_dim=decoder_dims[i],
                )
                for i in range(depth - 3, 0, -1)
            ]
        )
        del decoder_dims[0]

        self.transformer = VisionTransformer(input_dim, config, _PatchProjection)
        del config["patch_embedding"]["dropout"]
        self.dependency_map = _DependencyMap(
            decoder_dims[0], **config.get("patch_embedding")
        )

        self.boundary = nn.Sequential(
            BasicBlock(decoder_dims[0], decoder_dims[0] // 2, kernel_size=1, padding=0),
            nn.Conv2d(decoder_dims[0] // 2, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.se_block = SqueezeExciteBlock(decoder_dims[0], ratio=16)

        self.output_block = OutputBlock(2 * decoder_dims[0], output_dim)

    def forward(self, input: torch.Tensor):
        x = self.transformer(input)
        global_contexual, regional_distribution, _ = self.dependency_map(x)

        x_list = list()
        x_list.append(input)
        x = self.input_block(input)
        x_list.append(x)
        x = self.input_block_pool(x)

        for uent_encoder_block in self.unet_encoder:
            x = uent_encoder_block(x)
            x_list.append(x)
        del x_list[-1:]

        x = self.bridge(x)
        for unet_decoder_block in self.unet_decoder:
            x = unet_decoder_block(x, x_list.pop())

        x = self.se_block(x)
        boundary_out: torch.Tensor = self.boundary(x)
        boundary = repeat(boundary_out, "b 1 h w-> b c h w", c=x.shape[1])
        x = x + boundary

        attn = repeat(regional_distribution, "b 1 h w -> b c h w", c=x.shape[1])
        x = x * attn
        x = torch.cat((x, global_contexual), dim=1)
        return self.output_block(x)
