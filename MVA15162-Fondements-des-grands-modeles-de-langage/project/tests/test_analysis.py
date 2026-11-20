import numpy as np

from .adapters import run_find_rule, run_find_synonyms, run_fourier_energy


def test_find_synonyms(analysis_fixtures):
    f = analysis_fixtures
    pairs = sorted(tuple(sorted(map(int, pair))) for pair in run_find_synonyms(f["synonyms.vectors"]))
    assert pairs == [tuple(map(int, pair)) for pair in f["synonyms.pairs"]]


def test_fourier_energy(analysis_fixtures):
    f = analysis_fixtures
    np.testing.assert_allclose(run_fourier_energy(f["fourier.rows"]), f["fourier.energy"], atol=1e-6)


def test_find_rule(analysis_fixtures):
    f = analysis_fixtures
    for i in range(int(f["rule.count"])):
        a, b, c = run_find_rule(f[f"rule.sequences{i}"], int(f[f"rule.p{i}"]))
        assert (int(a), int(b), int(c)) == tuple(int(v) for v in f[f"rule.abc{i}"])
