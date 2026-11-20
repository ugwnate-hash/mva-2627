"""Part 3: training.

The three functions at the top are unit-tested. The training script below them is yours to write and organize as
you like (here, in a notebook, or in other files); the grader only sees the checkpoints it produces, saved with
alien.checkpoint.save_checkpoint.
"""

import argparse
import json
import os
import time

import torch

from .checkpoint import save_checkpoint
from .data import build_vocabulary, encode, find_codon_length, load_split
from .model import ModelConfig, TransformerLM, count_parameters


def get_batch(data: torch.Tensor, batch_size: int, generator: torch.Generator | None = None):
    """A batch of batch_size sequences drawn at random (with replacement) from the rows of data, of shape (N, T).
    Returns (x, y), each of shape (batch_size, T - 1): x is the sequence without its last codon, y without its
    first codon, so that y[:, t] is the codon that follows x[:, t]. Use `generator` for the random draw."""
    raise NotImplementedError


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy loss, for logits of shape (N, V) and targets of shape (N,) (token ids).
    Do not use F.cross_entropy or F.log_softmax. It must be stable for large logits."""
    raise NotImplementedError


def amino_acid_accuracy(logits: torch.Tensor, targets: torch.Tensor, amino_acid_of: torch.Tensor) -> float:
    """Accuracy at the level of amino acids, for logits of shape (N, V) and targets of shape (N,).

    amino_acid_of[i] is the amino acid (an integer from 0 to A - 1) of codon i. The probability of an amino acid is
    the sum of the probabilities (softmax of the logits) of its codons. A prediction is correct when the most
    probable amino acid is the amino acid of the target codon."""
    raise NotImplementedError


# Write your training script here (see Part 3 of the handout).
