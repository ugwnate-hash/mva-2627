import torch

from .adapters import run_build_vocabulary, run_encode, run_find_codon_length, run_split_into_codons


def test_find_codon_length(data_fixtures):
    for corpus in data_fixtures:
        assert run_find_codon_length(corpus["lines"]) == corpus["codon_length"], corpus["file"]


def test_split_into_codons(data_fixtures):
    for corpus in data_fixtures:
        L = corpus["codon_length"]
        for line in corpus["lines"][:20]:
            codons = run_split_into_codons(line, L)
            assert "".join(codons) == line
            assert all(len(c) == L for c in codons)


def test_build_vocabulary(data_fixtures):
    for corpus in data_fixtures:
        assert list(run_build_vocabulary(corpus["lines"], corpus["codon_length"])) == corpus["vocabulary"]


def test_encode(data_fixtures):
    for corpus in data_fixtures:
        lines = corpus["lines"][: len(corpus["encoded_head"])]
        ids = run_encode(lines, corpus["vocabulary"], corpus["codon_length"])
        assert ids.dtype == torch.long
        assert torch.equal(ids, torch.tensor(corpus["encoded_head"]))
