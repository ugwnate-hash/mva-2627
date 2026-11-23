"""Personal experiment: the RoPE per-pair contributions for many offsets at once, as heatmaps.

Same (q, k) examples as rope_pair_contrib.py (imported from it). Each heatmap row is one offset delta, each column
one pair i, and the color is the term q_i^T R(delta * theta_i) k_i (blue > 0, red < 0, gray = 0). A row of the
heatmap is one of the curves of rope_pair_contrib.png; the right-hand panel shows the row sums, i.e. the total
score for each delta. The middle column is the same without positional encoding (q_i^T k_i, no rotation).
Within a case, both heatmaps share one color scale.

Things to try: change DELTAS (e.g. np.arange(0, -120, -4) to see the slow pairs move), or the setup in
rope_pair_contrib.py (amp, D, BASE).

Run from the experiments/ folder, with its own venv:
    uv run python rope_intuition/rope_heatmap.py
The figure is saved next to this script as rope_heatmap.png.
"""
import os
from pathlib import Path

# OpenBLAS workaround (see rope_pair_contrib.py): must be set before numpy is first imported
os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")

import matplotlib

matplotlib.use("Agg")  # save to file, no window
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from rope_pair_contrib import BASE, D, GRID, INK, INK2, P, SURFACE, cases, contributions

DELTAS = np.arange(0, -30, -1)  # 30 offsets: 0, -1, ..., -29

# Diverging map: red (negative) -> neutral gray (0) -> blue (positive)
DIVERGING = LinearSegmentedColormap.from_list(
    "red_gray_blue", ["#8f1f1e", "#e34948", "#f4b8b7", "#f0efec", "#9ec5f4", "#2a78d6", "#0d366b"])
ROPE_COLOR, NOROPE_COLOR = "#2a78d6", INK2


def grid(q, k, use_rope):
    """(len(DELTAS), P) array: row = one delta, column = one pair."""
    return np.stack([contributions(q, k, d, use_rope) for d in DELTAS])


plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
# Columns: RoPE heatmap | no-RoPE heatmap | colorbar | total score per delta
fig, axes = plt.subplots(2, 4, figsize=(15.5, 11), facecolor=SURFACE,
                         gridspec_kw={"width_ratios": [1, 1, 0.04, 0.75]})

for row, (title, q, k, _) in enumerate(cases):
    g_rope, g_none = grid(q, k, True), grid(q, k, False)
    lim = max(np.abs(g_rope).max(), np.abs(g_none).max())
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim)
    extent = (-0.5, P - 0.5, DELTAS[-1] - 0.5, DELTAS[0] + 0.5)  # y axis = delta, 0 at the top

    for col, (g, label) in enumerate([(g_rope, "with RoPE:  q_iᵀ R(Δ·θ_i) k_i"),
                                      (g_none, "without RoPE:  q_iᵀ k_i")]):
        ax = axes[row][col]
        ax.imshow(g, cmap=DIVERGING, norm=norm, aspect="auto", origin="upper", extent=extent,
                  interpolation="nearest")
        ax.set_xticks(range(P))
        ax.set_xlabel("pair index i   (fast  →  slow)")
        ax.set_ylabel("offset Δ")
        ax.set_title(label, loc="left", fontsize=11, color=INK)
        for s in ax.spines.values():
            s.set_visible(False)
    # Both heatmaps of the row share cmap and norm, so the colorbar is built from those directly
    fig.colorbar(ScalarMappable(norm=norm, cmap=DIVERGING), cax=axes[row][2], label="term value (0 = gray)")
    axes[row][0].text(0, 1.13, title, transform=axes[row][0].transAxes, fontsize=12, color=INK)

    axS = axes[row][3]
    axS.plot(g_none.sum(axis=1), DELTAS, "--", color=NOROPE_COLOR, lw=1.5, label="without RoPE (flat)")
    axS.plot(g_rope.sum(axis=1), DELTAS, "-o", color=ROPE_COLOR, lw=2, ms=4, mec=SURFACE, mew=1,
             label="with RoPE")
    axS.axvline(0, color=INK2, lw=0.8)
    axS.set_ylim(DELTAS[-1] - 0.5, DELTAS[0] + 0.5)  # same rows as the heatmaps
    axS.set_facecolor(SURFACE)
    axS.grid(True, color=GRID, lw=0.8)
    axS.set_axisbelow(True)
    for s in ("top", "right"):
        axS.spines[s].set_visible(False)
    axS.set_xlabel("total score = row sum")
    axS.set_title("Total score per Δ  (same rows)", loc="left", fontsize=11, color=INK)
    axS.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=1)

fig.suptitle(f"Per-pair score terms for {len(DELTAS)} offsets  (head_dim = {D}, base = {BASE:g})",
             x=0.01, ha="left", fontsize=13, color=INK)
fig.subplots_adjust(left=0.05, right=0.97, top=0.9, bottom=0.08, hspace=0.55, wspace=0.4)
out = Path(__file__).with_suffix(".png")
fig.savefig(out, dpi=130, facecolor=SURFACE)
print(out)
