"""Adapters: the bridge between the tests and your code.

The tests never import your code directly: they call the functions below. They are already connected to the
skeleton of the `alien` package. If you organize your code differently, change these functions so that they call
your own code, but do not change their signatures. The grader uses your adapters.py with its own copy of the tests.
"""

import numpy as np
import torch

from alien import analysis, data, model, train


# Part 1: reading the genome

def run_find_codon_length(lines: list[str]) -> int:
    return data.find_codon_length(lines)


def run_split_into_codons(line: str, codon_length: int) -> list[str]:
    return data.split_into_codons(line, codon_length)


def run_build_vocabulary(lines: list[str], codon_length: int) -> list[str]:
    return data.build_vocabulary(lines, codon_length)


def run_encode(lines: list[str], vocabulary: list[str], codon_length: int) -> torch.Tensor:
    return data.encode(lines, vocabulary, codon_length)


# Part 2: the transformer

def run_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return model.rmsnorm(x, weight, eps)


def run_rope(x: torch.Tensor, base: float) -> torch.Tensor:
    return model.rope(x, base)


def run_causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window: int):
    """Returns (output, attention weights)."""
    return model.causal_attention(q, k, v, window)


def run_multihead_attention(d_model: int, n_heads: int, weights: dict[str, torch.Tensor], x: torch.Tensor,
                            window: int, rope_base: float) -> torch.Tensor:
    """weights has the keys q.weight, k.weight, v.weight and out.weight, each of shape (d_model, d_model)."""
    cfg = model.ModelConfig(d_model=d_model, n_heads=n_heads, rope_base=rope_base)
    attention = model.Attention(cfg)
    attention.load_state_dict(weights)
    return attention(x, window)


def run_build_model(config: dict) -> torch.nn.Module:
    """A TransformerLM built from a dict with the fields of ModelConfig."""
    return model.TransformerLM(model.ModelConfig(**config))


def run_transformer_lm(config: dict, weights: dict[str, torch.Tensor], idx: torch.Tensor,
                       attn_window: int | None = None) -> torch.Tensor:
    """Logits of a TransformerLM built from config, with the given state_dict, for the token ids idx."""
    lm = run_build_model(config)
    lm.load_state_dict(weights)
    return lm(idx, attn_window)


# Part 3: training

def run_get_batch(data_: torch.Tensor, batch_size: int, generator: torch.Generator):
    return train.get_batch(data_, batch_size, generator)


def run_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return train.cross_entropy(logits, targets)


def run_amino_acid_accuracy(logits: torch.Tensor, targets: torch.Tensor, amino_acid_of: torch.Tensor) -> float:
    return train.amino_acid_accuracy(logits, targets, amino_acid_of)


# Parts 4 and 6: reading the models

def run_find_synonyms(vectors: np.ndarray) -> list[tuple[int, int]]:
    return analysis.find_synonyms(vectors)


def run_fourier_energy(rows: np.ndarray) -> np.ndarray:
    return analysis.fourier_energy(rows)


def run_find_rule(sequences: np.ndarray, p: int) -> tuple[int, int, int]:
    return analysis.find_rule(sequences, p)
