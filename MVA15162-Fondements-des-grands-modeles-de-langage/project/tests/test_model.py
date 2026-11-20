import torch

from .adapters import (run_build_model, run_causal_attention, run_multihead_attention, run_rmsnorm, run_rope,
                       run_transformer_lm)

ATOL = 1e-5


def weights_of(tensors, prefix):
    return {k[len(prefix):]: v for k, v in tensors.items() if k.startswith(prefix)}


def test_rmsnorm(model_fixtures):
    _, t = model_fixtures
    torch.testing.assert_close(run_rmsnorm(t["rmsnorm.x"], t["rmsnorm.weight"], 1e-5), t["rmsnorm.y"], atol=ATOL, rtol=0)


def test_rope(model_fixtures):
    _, t = model_fixtures
    torch.testing.assert_close(run_rope(t["rope.x"], 10000.0), t["rope.y"], atol=ATOL, rtol=0)


def test_rope_other_base(model_fixtures):
    meta, t = model_fixtures
    torch.testing.assert_close(run_rope(t["rope.x"], meta["rope_other_base"]), t["rope.y_other_base"], atol=ATOL, rtol=0)


def test_causal_attention(model_fixtures):
    _, t = model_fixtures
    out, weights = run_causal_attention(t["attention.q"], t["attention.k"], t["attention.v"], 0)
    torch.testing.assert_close(out, t["attention.out.window0"], atol=ATOL, rtol=0)
    torch.testing.assert_close(weights, t["attention.weights.window0"], atol=ATOL, rtol=0)


def test_causal_attention_window(model_fixtures):
    meta, t = model_fixtures
    for w in meta["attention_windows"]:
        out, weights = run_causal_attention(t["attention.q"], t["attention.k"], t["attention.v"], w)
        torch.testing.assert_close(out, t[f"attention.out.window{w}"], atol=ATOL, rtol=0)
        torch.testing.assert_close(weights, t[f"attention.weights.window{w}"], atol=ATOL, rtol=0)


def test_multihead_attention(model_fixtures):
    meta, t = model_fixtures
    m = meta["mha"]
    for w in (0, 2):
        out = run_multihead_attention(m["d_model"], m["n_heads"], weights_of(t, "mha.weights."), t["mha.x"], w,
                                      m["rope_base"])
        torch.testing.assert_close(out, t[f"mha.out.window{w}"], atol=ATOL, rtol=0)


def test_parameter_names_and_shapes(model_fixtures):
    meta, _ = model_fixtures
    for name, spec in meta["lm"].items():
        lm = run_build_model(spec["config"])
        shapes = {k: list(v.shape) for k, v in lm.state_dict().items()}
        assert shapes == spec["shapes"], name


def test_transformer_lm(model_fixtures):
    meta, t = model_fixtures
    for name, spec in meta["lm"].items():
        logits = run_transformer_lm(spec["config"], weights_of(t, f"{name}.weights."), t[f"{name}.idx"])
        torch.testing.assert_close(logits, t[f"{name}.logits"], atol=ATOL, rtol=0)


def test_transformer_lm_cut_attention(model_fixtures):
    meta, t = model_fixtures
    for name, spec in meta["lm"].items():
        for w in spec["cuts"]:
            logits = run_transformer_lm(spec["config"], weights_of(t, f"{name}.weights."), t[f"{name}.idx"], w)
            torch.testing.assert_close(logits, t[f"{name}.logits.cut{w}"], atol=ATOL, rtol=0)


def test_transformer_lm_is_causal(model_fixtures):
    meta, t = model_fixtures
    name, spec = next(iter(meta["lm"].items()))
    idx = t[f"{name}.idx"].clone()
    before = run_transformer_lm(spec["config"], weights_of(t, f"{name}.weights."), idx)
    idx[:, -3:] = (idx[:, -3:] + 1) % spec["config"]["vocab_size"]
    after = run_transformer_lm(spec["config"], weights_of(t, f"{name}.weights."), idx)
    torch.testing.assert_close(before[:, :-3], after[:, :-3], atol=ATOL, rtol=0)
