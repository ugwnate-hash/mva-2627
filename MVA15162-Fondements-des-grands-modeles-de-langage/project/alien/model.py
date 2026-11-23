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
    d = x.shape[-1]
    sum_x2 = (x**2).sum(dim=-1, keepdim=True)
    denominator = torch.sqrt( 1 / d * sum_x2 + eps)
    norm = x / denominator * weight
    return norm


def rope(x: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """Rotary position embedding of x, of shape (..., T, head_dim), positions 0 .. T-1.

    The channels are grouped in pairs (x[2i], x[2i+1]), for i = 0 .. head_dim/2 - 1. At position t, the pair i is
    rotated by the angle t * theta_i, with theta_i = base ** (-2i / head_dim):
        y[2i]   = x[2i] cos(t theta_i) - x[2i+1] sin(t theta_i)
        y[2i+1] = x[2i] sin(t theta_i) + x[2i+1] cos(t theta_i)
    """

    head_dim = x.shape[-1]
    T = x.shape[-2]
    thetas = torch.arange(0, head_dim, step=2, dtype=x.dtype, device=x.device)
    thetas = base ** (- thetas / head_dim)
    position = torch.arange(0, T, dtype=x.dtype, device=x.device)
    angles = torch.mm(position.unsqueeze(-1), thetas.unsqueeze(0))
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    even = x_even * cos - x_odd * sin
    odd = x_even * sin + x_odd * cos 
    stack = torch.stack([even, odd], dim=-1)
    RoPE = stack.flatten(-2)
    return RoPE


def causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window: int = 0):
    """Scaled dot-product attention with a causal mask, for q, k, v of shape (..., T, head_dim).

    Position i attends to the positions j <= i, and, if window > 0, only to those with j > i - window (itself and
    the window - 1 previous positions). Scores are divided by sqrt(head_dim).
    Returns the output, of shape (..., T, head_dim), and the attention weights, of shape (..., T, T)."""
    
    head_dim = q.shape[-1]
    T = q.shape[-2]
    if window == 0:
        mask = torch.triu(torch.full((T, T), -torch.inf, device=q.device, dtype=q.dtype), diagonal=1)
    else:
        mask = torch.triu(torch.full((T, T), -torch.inf, device=q.device, dtype=q.dtype), diagonal=1) + torch.tril(torch.full((T, T), -torch.inf, device=q.device, dtype=q.dtype), diagonal=-window)

    A = torch.nn.functional.softmax(1 / math.sqrt(head_dim) * torch.matmul(q, torch.transpose(k, -1, -2)) + mask, dim=-1)
    output = A @ v
    attention_weights = A 
    return output, attention_weights


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

        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)
        B, T, d_model = Q.shape
        n_heads = self.n_heads
        head_dim = d_model//n_heads
        Q = torch.reshape(Q, (B, T, n_heads, head_dim))
        Q = torch.transpose(Q, 1, 2)
        K = torch.reshape(K, (B, T, n_heads, head_dim))
        K = torch.transpose(K, 1, 2)
        V = torch.reshape(V, (B, T, n_heads, head_dim))
        V = torch.transpose(V, 1, 2)

        Q = rope(Q, self.rope_base)
        K = rope(K, self.rope_base)

        attn_out, attn_weights = causal_attention(Q, K, V, window)
        attn_out = torch.transpose(attn_out, 1, 2)
        attn_out = torch.reshape(attn_out, (B, T, d_model))
        out = self.out(attn_out)
        self.weights = attn_weights.detach()
        return out


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.up = nn.Linear(cfg.d_model, cfg.d_mlp, bias=False)
        self.down = nn.Linear(cfg.d_mlp, cfg.d_model, bias=False)

    def forward(self, x):
        """down(relu(up(x)))."""
        up = self.up
        down = self.down

        x = torch.nn.functional.relu(up(x))
        out = down(x)
        return out


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x, window: int = 0):
        """x + attn(norm1(x)), then x + mlp(norm2(x))."""
        attn = self.attn
        norm1 = self.norm1
        norm2 = self.norm2
        mlp = self.mlp

        x = x + attn(norm1(x), window=window)
        out = x + mlp(norm2(x))
        return out


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

        x = self.embed(idx)
        if attn_window is not None:
            for block in self.blocks:
                x = block(x, attn_window)
        else:
            for block in self.blocks:
                x = block(x, self.cfg.attn_window)
        x = self.norm(x)
        logits = self.unembed(x)

        return logits

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
