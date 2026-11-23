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
    N_letters = len(get_letters(lines)) 
    for k in range(1,11):
        flag = all(L%k == 0 for L in [len(line) for line in lines])
        if flag:
            n_possible = N_letters**k
            chunks = get_chunks(k, lines)
            if 10 * len(set(chunks)) <= n_possible:
                return k
    raise ValueError("no codon length found")

def get_chunks(k, lines):
    chunks = list()
    for line in lines:
        L = len(line)
        for i in range(L//k):
            chunks.append(line[i*k: (i+1)*k])
    return chunks

def get_letters(lines):
    letters = set()
    for line in lines:
        letters.update(line)
    return letters

def split_into_codons(line: str, codon_length: int) -> list[str]:
    """The codons of one line, in order."""
    return get_chunks(codon_length, [line])


def build_vocabulary(lines: list[str], codon_length: int) -> list[str]:
    """All the distinct codons that appear in the lines, sorted in alphabetical order."""
    return sorted(set(get_chunks(codon_length, lines)))


def encode(lines: list[str], vocabulary: list[str], codon_length: int) -> torch.Tensor:
    """A tensor of token ids of shape (number of lines, codons per line), dtype torch.long. All lines have the
    same length. The token id of a codon is its index in `vocabulary`."""

    n_lines = len(lines)
    n_codons_per_line = int(len(get_chunks(codon_length, lines))/n_lines) 
    T = torch.zeros(n_lines, n_codons_per_line, dtype=torch.long)
    word_to_id = build_word_to_id(vocabulary)
    for i, line in enumerate(lines):
        codons = split_into_codons(line, codon_length)
        for j in range(n_codons_per_line):
            T[i,j] = word_to_id[codons[j]]
    return T

def build_word_to_id(vocab):
    dico = {}
    for i, word in enumerate(vocab):
        dico[word] = i
    return dico