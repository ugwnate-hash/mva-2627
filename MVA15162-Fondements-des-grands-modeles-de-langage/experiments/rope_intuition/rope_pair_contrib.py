"""Personal experiment: RoPE intuition, independent of the graded project.

The RoPE attention score splits into one 2D dot product per channel pair:
    score(delta) = sum_i q_i^T R(delta * theta_i) k_i,   theta_i = base ** (-2i / head_dim),   delta = n - m.
This script plots each term against the pair index i for a few offsets delta, and the total score against delta,
for three hand-built (q, k):
    A. same content (k = q), energy in every pair: fast pairs get scrambled, slow pairs keep matching content;
    B. a "previous-token" head (k_i = R(theta_i) q_i), energy in fast pairs: all pairs align at delta = -1;
    C. a "distant-token" head (k_i = R(L theta_i) q_i, L = 20), energy in fast pairs: all pairs align at delta = -L.

For comparison, the same figure is drawn without positional encoding (score = sum_i q_i^T k_i): nothing rotates,
so every delta gives the same terms and the score is flat in delta (it equals the RoPE score at delta = 0).

Things to try: change the deltas in `cases`, the energy profile `amp`, the distance L, head_dim D or BASE.

Run from the experiments/ folder, with its own venv:
    uv run python rope_intuition/rope_pair_contrib.py
The figures are saved next to this script: rope_pair_contrib.png (RoPE) and rope_pair_contrib_norope.png (no
positional encoding, same axis ranges).
"""
import os
from pathlib import Path

# On this machine (ARM64, WSL), numpy's bundled OpenBLAS auto-detects a CPU kernel that crashes with
# "Illegal instruction" on matrix products. Forcing the generic ARMv8 kernel avoids it. Must be set before
# numpy is imported.
os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")

import matplotlib

matplotlib.use("Agg")  # save to file, no window
import matplotlib.pyplot as plt
import numpy as np

D, BASE = 32, 10000.0
P = D // 2
i = np.arange(P)
theta = BASE ** (-2 * i / D)

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]


def rot(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def contributions(q, k, delta, use_rope=True):
    """q, k: (P, 2). Returns the (P,) array of q_i^T R(delta theta_i) k_i, or of q_i^T k_i without RoPE
    (no rotation: the same as RoPE at delta = 0, for every delta)."""
    angles = delta * theta if use_rope else np.zeros(P)
    return np.array([q[j] @ rot(angles[j]) @ k[j] for j in range(P)])


rng = np.random.default_rng(0)

# A: content match, k = q (random content, equal energy in every pair)
qA = rng.normal(size=(P, 2))
kA = qA.copy()

# B: "previous-token" head: k_i = R(theta_i) q_i so every pair peaks at delta = -1; energy in fast pairs
amp = np.exp(-i / 3.0)
qB = rng.normal(size=(P, 2))
qB = qB / np.linalg.norm(qB, axis=1, keepdims=True) * np.sqrt(amp)[:, None]
kB = np.stack([rot(theta[j]) @ qB[j] for j in range(P)])

# C: "distant-token" head: same recipe with a head start of L ticks, k_i = R(L theta_i) q_i, so every pair peaks at
# delta = -L. Energy in fast pairs again: they tell -L from -L +/- 1, and mixing several speeds avoids echoes.
# (Drawn after A and B so that their random values are unchanged.)
L = 20
qC = rng.normal(size=(P, 2))
qC = qC / np.linalg.norm(qC, axis=1, keepdims=True) * np.sqrt(amp)[:, None]
kC = np.stack([rot(L * theta[j]) @ qC[j] for j in range(P)])

cases = [
    ("A. Same content (k = q), energy in every pair", qA, kA, [0, -3, -50]),
    ("B. 'Previous-token' head (k_i = R(θ_i) q_i), energy in fast pairs", qB, kB, [-1, -2, -8]),
    (f"C. 'Distant-token' head (k_i = R({L}·θ_i) q_i), energy in fast pairs", qC, kC, [-L, -L + 1, -L + 6]),
]

plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
deltas_line = np.arange(-64, 1)


def total_scores(q, k, use_rope):
    return np.array([contributions(q, k, d, use_rope).sum() for d in deltas_line])


def make_figure(use_rope, ylims=None):
    """The 2x2 figure, with or without RoPE. ylims: per-panel y-limits to reuse, so both figures share scales.
    Without RoPE, the right panels also show the RoPE total as a dashed gray reference."""
    fig, axes = plt.subplots(len(cases), 2, figsize=(13, 4.25 * len(cases)), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1.25, 1]})
    term = "q_iᵀ R(Δ·θ_i) k_i" if use_rope else "q_iᵀ k_i"
    for row, (title, q, k, deltas) in enumerate(cases):
        axL, axR = axes[row]
        for ax in (axL, axR):
            ax.set_facecolor(SURFACE)
            ax.grid(True, color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.axhline(0, color=INK2, lw=0.8)

        # Without RoPE the three curves coincide exactly: shrink the markers so the ones underneath stay visible
        sizes = [6, 6, 6] if use_rope else [11, 7.5, 4]
        for c, d, ms in zip(SERIES, deltas, sizes):
            contrib = contributions(q, k, d, use_rope)
            axL.plot(i, contrib, "-o", color=c, lw=2, ms=ms, mec=SURFACE, mew=1.5,
                     label=f"Δ = {d}   (sum = {contrib.sum():.2f})")
        axL.set_xticks(i)
        axL.set_xlabel("pair index i   (fast rotation  →  slow rotation)" if use_rope else "pair index i")
        axL.set_ylabel(term)
        axL.set_title(title if use_rope else f"{title}\nno rotation: the three Δ curves are identical",
                      loc="left", fontsize=11, color=INK)
        axL.legend(frameon=False, loc="upper right")

        if not use_rope:
            axR.plot(deltas_line, total_scores(q, k, True), "--", color=INK2, lw=1.2, alpha=0.6,
                     label="with RoPE (for comparison)")
        axR.plot(deltas_line, total_scores(q, k, use_rope), color=SERIES[0], lw=2,
                 label="with RoPE" if use_rope else "without RoPE: flat")
        for c, d in zip(SERIES, deltas):
            axR.plot(d, contributions(q, k, d, use_rope).sum(), "o", color=c, ms=8, mec=SURFACE, mew=1.5)
        axR.set_xlabel("offset Δ = n − m   (key position − query position)")
        axR.set_ylabel(f"score = Σ_i {term}")
        axR.set_title("Total score vs offset (dots = the Δ values on the left)", loc="left",
                      fontsize=11, color=INK)
        if not use_rope:
            axR.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)

        if ylims is not None:
            axL.set_ylim(ylims[row][0])
            axR.set_ylim(ylims[row][1])

    head = "RoPE" if use_rope else "No positional encoding"
    fig.suptitle(f"{head}: per-pair contributions to the attention score  (head_dim = {D}, base = {BASE:g})",
                 x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    return fig, [(axes[r][0].get_ylim(), axes[r][1].get_ylim()) for r in range(len(cases))]


if __name__ == "__main__":  # rope_heatmap.py imports the setup above without drawing these figures
    here = Path(__file__).with_suffix("")
    fig_rope, ylims = make_figure(use_rope=True)
    fig_rope.savefig(f"{here}.png", dpi=130, facecolor=SURFACE)
    fig_none, _ = make_figure(use_rope=False, ylims=ylims)
    fig_none.savefig(f"{here}_norope.png", dpi=130, facecolor=SURFACE)
    print(f"{here}.png\n{here}_norope.png")
