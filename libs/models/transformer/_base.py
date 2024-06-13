import torch.nn
import torch.nn as nn

__all__ = ["MLPBlock", "PreNorm"]


class MLPBlock(nn.Sequential):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        mlp_dropout: float,
        proj_dropout: float,
        relu_type: nn.Module = nn.GELU,
    ) -> None:
        super().__init__(
            nn.Linear(embedding_dim, mlp_dim),
            relu_type(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_dim, embedding_dim),
            nn.Dropout(proj_dropout),
        )


class PreNorm(nn.Sequential):
    def __init__(self, embedding_dim: int, fn: nn.Module):
        super().__init__()

        self.norm = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.fn = fn

    def forward(self, input: torch.tensor, **kwargs):
        x = self.norm(input)
        return self.fn(x, **kwargs) + input
