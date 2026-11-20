"""Parts 4 and 6: reading the trained models.

All functions take and return numpy arrays (use tensor.detach().cpu().numpy() to convert a tensor).
"""

import numpy as np


def find_synonyms(vectors: np.ndarray) -> list[tuple[int, int]]:
    """Pairs of rows that are each other's most similar row, by cosine similarity after subtracting the mean row.

    vectors has shape (n, d), with one row per codon (for example rows of the embedding matrix). Returns the
    sorted list of pairs (i, j) with i < j such that j is the most similar row to i and i is the most similar row
    to j (a row is never compared with itself)."""
    raise NotImplementedError


def fourier_energy(rows: np.ndarray) -> np.ndarray:
    """Share of the energy of each frequency k = 1 .. (p-1)/2, for rows of shape (p, d) given in their order
    x = 0 .. p-1 (p odd).

    Subtract the mean row, then compute F[k] = sum_x rows[x] exp(-2 i pi k x / p), a vector of d complex numbers.
    The energy of frequency k is |F[k]|^2 + |F[p-k]|^2, summed over the d dimensions (k and p - k describe the
    same circle). Returns the (p-1)/2 energies divided by their sum."""
    raise NotImplementedError


def find_rule(sequences: np.ndarray, p: int) -> tuple[int, int, int]:
    """The coefficients (a, b, c), each in 0 .. p-1, of the rule x_t = (a x_{t-1} + b x_{t-2} + c) mod p that every
    sequence follows. sequences has shape (n, T) and holds integers mod p. You may assume that the rule exists and
    is unique."""
    raise NotImplementedError
