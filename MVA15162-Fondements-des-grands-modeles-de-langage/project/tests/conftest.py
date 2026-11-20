"""Shared fixtures of the tests. The reference values are in tests/fixtures/. The grader runs the same tests
with other reference values (other corpora, weights and synthetic data), so a function that only works on these
inputs will fail."""

import json
import os

import numpy as np
import pytest
from safetensors.torch import load_file

FIXTURES = os.environ.get("ALIEN_FIXTURES", os.path.join(os.path.dirname(__file__), "fixtures"))


@pytest.fixture(scope="session")
def fixtures_dir():
    return FIXTURES


@pytest.fixture(scope="session")
def data_fixtures():
    with open(os.path.join(FIXTURES, "data.json")) as f:
        corpora = json.load(f)
    for corpus in corpora:
        with open(os.path.join(FIXTURES, corpus["file"])) as f:
            corpus["lines"] = f.read().split()
    return corpora


@pytest.fixture(scope="session")
def model_fixtures():
    with open(os.path.join(FIXTURES, "model.json")) as f:
        meta = json.load(f)
    return meta, load_file(os.path.join(FIXTURES, "model.safetensors"))


@pytest.fixture(scope="session")
def train_fixtures():
    return load_file(os.path.join(FIXTURES, "train.safetensors"))


@pytest.fixture(scope="session")
def analysis_fixtures():
    with np.load(os.path.join(FIXTURES, "analysis.npz"), allow_pickle=False) as f:
        return dict(f)
