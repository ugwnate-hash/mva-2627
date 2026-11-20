"""Part 1: reading the genome.

The corpus is a list of lines of letters, one sequence per line. The functions below find the codons and turn the
lines into tensors of token ids. Token id i is the i-th codon of the vocabulary, sorted in alphabetical order.
"""

import torch
from huggingface_hub import hf_hub_download

REPO = "dlouapre/alien-genome"


def load_split(split: str) -> list[str]:
    """The lines of `train.txt` or `valid.txt` (split = "train" or "valid"), downloaded from the Hugging Face Hub.
    Provided."""
    path = hf_hub_download(REPO, f"{split}.txt", repo_type="dataset")
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def find_codon_length(lines: list[str]) -> int:
    """The length of the codons (in letters) in which the lines are written.

    You may assume that every line is a whole number of codons and starts with a codon, that codons all have the
    same length (at most 10 letters), that the codons in use are at most a tenth of all the strings of that length
    that the alphabet allows, and that the corpus holds at least a few thousand codons."""
    raise NotImplementedError


def split_into_codons(line: str, codon_length: int) -> list[str]:
    """The codons of one line, in order."""
    raise NotImplementedError


def build_vocabulary(lines: list[str], codon_length: int) -> list[str]:
    """All the distinct codons that appear in the lines, sorted in alphabetical order."""
    raise NotImplementedError


def encode(lines: list[str], vocabulary: list[str], codon_length: int) -> torch.Tensor:
    """A tensor of token ids of shape (number of lines, codons per line), dtype torch.long. All lines have the
    same length. The token id of a codon is its index in `vocabulary`."""
    raise NotImplementedError
