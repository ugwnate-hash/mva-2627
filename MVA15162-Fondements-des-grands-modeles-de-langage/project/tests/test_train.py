import math

import torch

from .adapters import run_amino_acid_accuracy, run_cross_entropy, run_get_batch


def test_get_batch():
    data = torch.arange(50 * 9).reshape(50, 9)
    x, y = run_get_batch(data, 16, torch.Generator().manual_seed(0))
    assert x.shape == (16, 8) and y.shape == (16, 8)
    assert torch.equal(x[:, 1:], y[:, :-1])
    rows = set(map(tuple, data.tolist()))
    for a, b in zip(x.tolist(), y.tolist()):
        assert tuple(a + b[-1:]) in rows
    x2, _ = run_get_batch(data, 16, torch.Generator().manual_seed(0))
    assert torch.equal(x, x2), "the same generator seed must give the same batch"
    seen = set()
    for seed in range(20):
        xs, _ = run_get_batch(data, 16, torch.Generator().manual_seed(seed))
        seen |= {int(r[0]) // 9 for r in xs}
    assert len(seen) > 40, "batches must be drawn at random from all the rows"


def test_cross_entropy(train_fixtures):
    t = train_fixtures
    torch.testing.assert_close(run_cross_entropy(t["ce.logits"], t["ce.targets"]), t["ce.value"], atol=1e-5, rtol=1e-6)


def test_cross_entropy_large_logits(train_fixtures):
    t = train_fixtures
    value = run_cross_entropy(t["ce.logits"] * 1000, t["ce.targets"])
    assert math.isfinite(value.item())
    torch.testing.assert_close(value, t["ce.value_large"], atol=1e-2, rtol=1e-5)


def test_amino_acid_accuracy(train_fixtures):
    t = train_fixtures
    # Two codons of amino acid 0 with probability 0.3 each beat one codon of amino acid 1 with probability 0.4.
    logits = torch.log(torch.tensor([[0.3, 0.3, 0.4, 1e-9]]))
    amino_acid_of = torch.tensor([0, 0, 1, 1])
    assert run_amino_acid_accuracy(logits, torch.tensor([1]), amino_acid_of) == 1.0
    assert run_amino_acid_accuracy(logits, torch.tensor([2]), amino_acid_of) == 0.0
    value = run_amino_acid_accuracy(t["aa.logits"], t["aa.targets"], t["aa.amino_acid_of"])
    assert abs(value - t["aa.value"].item()) < 1e-6
