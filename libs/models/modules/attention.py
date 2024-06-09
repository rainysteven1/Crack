from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .base import BasicBlock


class AttentionBlock(nn.Module):
    """Soft Attention."""

    def __init__(
        self, F_g: int, F_l: int, F_int: int, init_type: Optional[str] = None
    ) -> None:
        super().__init__()

        self.W_g = BasicBlock(
            F_g, F_int, kernel_size=1, padding=0, is_relu=False, init_type=init_type
        )
        self.W_x = BasicBlock(
            F_l, F_int, kernel_size=1, padding=0, is_relu=False, init_type=init_type
        )
        self.psi = BasicBlock(
            F_int, 1, kernel_size=1, padding=0, is_relu=False, init_type=init_type
        )

    def forward(self, input: torch.Tensor, skip: torch.Tensor):
        x = F.relu(self.W_g(input) + self.W_x(skip))
        return skip * F.sigmoid(self.psi(x))


class MHSA(nn.Module):
    """Multi-Head Self-Attention module."""

    def __init__(
        self, input_dim: int, n_heads: int, qkv_bias: bool, dropout: float
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.scale = input_dim**-0.5

        self.qkv = nn.Linear(input_dim, input_dim * 3, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input: torch.Tensor, mask: Optional[torch.Tensor] = None):
        qkv = rearrange(
            self.qkv(input), "b n (h d qkv) -> (qkv) b h n d", h=self.n_heads, qkv=3
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        energy = torch.einsum(
            "bhqd, bhkd -> bhqk", q, k
        )  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        attn = F.softmax(energy * self.scale, dim=-1)
        attn = self.dropout(attn)

        output = torch.einsum("bhal, bhlv -> bhav", attn, v)
        output = rearrange(output, "b h n d -> b n (h d)")
        return output


class _MHSA(nn.Module):
    """Multi-Head Self-Attention module."""

    def __init__(self, input_dim: int, n_heads: int = 4) -> None:
        assert (input_dim % n_heads) == 0
        super().__init__()
        self.n_heads = n_heads
        self.head_size = input_dim // self.n_heads
        self.all_head_size = input_dim

        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

    def _transpose_attn_scores(self, x: torch.Tensor):
        new_x_shape = [x.size()[0], -1, self.n_heads, self.head_size]
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def _transpose_context_scores(self, x: torch.Tensor):
        x = x.transpose(1, 2).contiguous()
        new_x_shape = x.size()[:2] + (self.all_head_size,)
        return x.view(*new_x_shape)

    def _qkv(self, input: torch.Tensor):
        q = self._transpose_attn_scores(self.query(input))
        k = self._transpose_attn_scores(self.key(input))
        v = self._transpose_attn_scores(self.value(input))
        return q, k, v

    def _self_attention(self, input: torch.Tensor):
        q, k, v = self._qkv(input)

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(
            self.head_size
        )
        attention_probs = F.softmax(attention_scores, dim=-1)

        return torch.matmul(attention_probs, v)

    def forward(self, input: torch.Tensor):
        return self._transpose_context_scores(self._self_attention(input))


class BoT_MHSA(_MHSA):
    """
    Bottleneck Transformers Multi-Head Self-Attention module
    reference: https://github.com/leaderj1001/BottleneckTransformers/blob/main/model.py
    """

    def __init__(self, input_dim: int, img_size: int = 256, n_heads: int = 4) -> None:
        super().__init__(input_dim, n_heads)

        self.query = BasicBlock(
            input_dim, input_dim, kernel_size=1, padding=0, is_bn=False, is_relu=False
        )
        self.key = BasicBlock(
            input_dim, input_dim, kernel_size=1, padding=0, is_bn=False, is_relu=False
        )
        self.value = BasicBlock(
            input_dim, input_dim, kernel_size=1, padding=0, is_bn=False, is_relu=False
        )

        self.height, self.width = int(np.sqrt(img_size)), int(np.sqrt(img_size))
        # 二维位置编码
        self.rel_h = nn.Parameter(
            torch.randn([1, self.n_heads, self.head_size, 1, self.height]),
            requires_grad=True,
        )
        self.rel_w = nn.Parameter(
            torch.randn([1, self.n_heads, self.head_size, self.width, 1]),
            requires_grad=True,
        )

    def _transpose_context_scores(self, x: torch.Tensor):
        x = x.transpose(1, 2).contiguous()
        new_x_shape = [x.size()[0], self.all_head_size, self.width, self.height]
        return x.view(*new_x_shape)

    def _self_attention(self, input: torch.Tensor):
        q, k, v = self._qkv(input)

        attn_content = torch.matmul(q, k.transpose(-2, -1))
        attn_position = (self.rel_h + self.rel_w).view(
            1, self.n_heads, self.head_size, -1
        )
        attn_position = torch.matmul(q, attn_position)
        attn_scores = attn_content + attn_position
        attn_probs = F.softmax(attn_scores, dim=-1)

        return torch.matmul(attn_probs, v)
