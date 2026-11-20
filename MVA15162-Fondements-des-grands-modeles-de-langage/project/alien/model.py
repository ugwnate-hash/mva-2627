"""Part 2: the transformer.

A decoder-only transformer: token embedding, n_layers pre-norm blocks (RMSNorm, causal attention with RoPE, RMSNorm,
ReLU MLP, each with a residual connection), a final RMSNorm, and an unembedding matrix that is not tied to the
embedding. The grader loads your checkpoints into its own copy of this model, so the names and shapes of the
parameters are fixed: they are created in the constructors below, which are provided. Do not rename them.

Allowed: tensor operations, nn.Module, nn.Parameter, nn.Linear, nn.Embedding, nn.ModuleList.
Not allowed: F.scaled_dot_product_attention, nn.MultiheadAttention, nn.RMSNorm, nn.Transformer*, F.rms_norm,
and any library that implements these layers for you.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    vocab_size: int = 0  # the number of codons: set it from your vocabulary
    d_model: int = 128
    n_layers: int = 1
    n_heads: int = 4
    d_mlp: int = 512
    attn_window: int = 0  # 0: each codon attends to itself and all previous codons; w > 0: to itself and the w - 1 previous ones
    rope_base: float = 10000.0


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """RMSNorm over the last dimension: x / sqrt(mean(x^2) + eps) * weight."""
    raise NotImplementedError


def rope(x: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """Rotary position embedding of x, of shape (..., T, head_dim), positions 0 .. T-1.

    The channels are grouped in pairs (x[2i], x[2i+1]), for i = 0 .. head_dim/2 - 1. At position t, the pair i is
    rotated by the angle t * theta_i, with theta_i = base ** (-2i / head_dim):
        y[2i]   = x[2i] cos(t theta_i) - x[2i+1] sin(t theta_i)
        y[2i+1] = x[2i] sin(t theta_i) + x[2i+1] cos(t theta_i)
    """
    raise NotImplementedError


def causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window: int = 0):
    """Scaled dot-product attention with a causal mask, for q, k, v of shape (..., T, head_dim).

    Position i attends to the positions j <= i, and, if window > 0, only to those with j > i - window (itself and
    the window - 1 previous positions). Scores are divided by sqrt(head_dim).
    Returns the output, of shape (..., T, head_dim), and the attention weights, of shape (..., T, T)."""
    raise NotImplementedError


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        return rmsnorm(x, self.weight, self.eps)


class Attention(nn.Module):
    """Multi-head attention. Head h uses the channels h * head_dim .. (h+1) * head_dim - 1 of the queries, keys and
    values, with head_dim = d_model / n_heads. RoPE is applied to the queries and keys of each head."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.rope_base = cfg.rope_base
        self.q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.weights = None  # attention weights of the last forward pass, (batch, heads, T, T), for plots

    def forward(self, x: torch.Tensor, window: int = 0) -> torch.Tensor:
        """x: (batch, T, d_model) -> (batch, T, d_model)."""
        raise NotImplementedError


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.up = nn.Linear(cfg.d_model, cfg.d_mlp, bias=False)
        self.down = nn.Linear(cfg.d_mlp, cfg.d_model, bias=False)

    def forward(self, x):
        """down(relu(up(x)))."""
        raise NotImplementedError


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x, window: int = 0):
        """x + attn(norm1(x)), then x + mlp(norm2(x))."""
        raise NotImplementedError


class TransformerLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0 and (cfg.d_model // cfg.n_heads) % 2 == 0, "head_dim must be even"
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        for module in self.modules():  # the initialization is yours to choose
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, idx: torch.Tensor, attn_window: int | None = None) -> torch.Tensor:
        """Logits of shape (batch, T, vocab_size) for token ids idx of shape (batch, T).

        attn_window overrides the window of the config for this call only (None: use cfg.attn_window). This is
        how you cut the attention of a trained model to the last w codons."""
        raise NotImplementedError


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
