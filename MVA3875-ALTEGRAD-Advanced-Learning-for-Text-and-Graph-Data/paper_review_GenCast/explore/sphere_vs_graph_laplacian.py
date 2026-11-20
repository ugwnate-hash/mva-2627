"""Spherical harmonics as the bridge between GenCast and graph learning.

Three small numerical checks (a few seconds on a laptop):

1. Spectrum. Build the icosahedral mesh (the kind GenCast's processor runs on),
   take its combinatorial graph Laplacian L = D - A, and compare its spectrum
   with the Laplace-Beltrami spectrum on the sphere: eigenvalue l(l+1) with
   multiplicity 2l+1.
2. Eigenvectors. Measure how much of each graph-Laplacian eigenvector lies in
   the span of the degree-l spherical harmonics sampled at the mesh nodes.
3. Noise isotropy. Compare i.i.d. Gaussian noise per lat-lon pixel with white
   noise sampled in spherical-harmonic space (as GenCast is described to do),
   by measuring the variance of the noise averaged over equal-area caps at
   different latitudes.

Usage (from the repo root, inside WSL):
    .venv/bin/python <this file> [--refine 4] [--lmax 10] [--out DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy import special

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def icosphere(refine: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (vertices on the unit sphere, triangular faces) of an icosahedron
    refined `refine` times by midpoint subdivision. #nodes = 10 * 4**refine + 2."""
    t = (1 + 5**0.5) / 2
    verts = [
        (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
        (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
        (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1),
    ]
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]
    verts = [np.array(v, dtype=float) / np.linalg.norm(v) for v in verts]

    for _ in range(refine):
        cache: dict[tuple[int, int], int] = {}

        def midpoint(i: int, j: int) -> int:
            key = (min(i, j), max(i, j))
            if key not in cache:
                m = verts[i] + verts[j]
                verts.append(m / np.linalg.norm(m))
                cache[key] = len(verts) - 1
            return cache[key]

        new_faces = []
        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = new_faces

    return np.array(verts), np.array(faces)


def graph_laplacian(n_nodes: int, faces: np.ndarray) -> tuple[sp.csr_matrix, np.ndarray]:
    """Combinatorial Laplacian L = D - A of the mesh graph, plus the edge list."""
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    ones = np.ones(len(edges))
    A = sp.coo_matrix((ones, (edges[:, 0], edges[:, 1])), shape=(n_nodes, n_nodes))
    A = (A + A.T).tocsr()
    D = sp.diags(np.asarray(A.sum(axis=1)).ravel())
    return (D - A).tocsr(), edges


def to_angles(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cartesian -> (colatitude theta in [0, pi], longitude phi in [-pi, pi])."""
    theta = np.arccos(np.clip(xyz[:, 2], -1, 1))
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])
    return theta, phi


def sph_harm_complex(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Complex Y_l^m at (colatitude, longitude), across SciPy versions."""
    if hasattr(special, "sph_harm_y"):  # SciPy >= 1.15
        return special.sph_harm_y(l, m, theta, phi)
    return special.sph_harm(m, l, phi, theta)  # older argument order


def real_sph_harm_basis(l: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """(n_points, 2l+1) matrix of real degree-l spherical harmonics."""
    cols = [sph_harm_complex(l, 0, theta, phi).real]
    for m in range(1, l + 1):
        y = sph_harm_complex(l, m, theta, phi)
        cols += [np.sqrt(2) * y.real, np.sqrt(2) * y.imag]
    return np.stack(cols, axis=1)


# ---------------------------------------------------------------------------
# Experiments 1 + 2: graph Laplacian of the icosahedral mesh vs Laplace-Beltrami
# ---------------------------------------------------------------------------


def laplacian_experiment(refine: int, lmax: int, out: Path) -> None:
    verts, faces = icosphere(refine)
    n = len(verts)
    L, edges = graph_laplacian(n, faces)
    h = np.linalg.norm(verts[edges[:, 0]] - verts[edges[:, 1]], axis=1).mean()
    print(f"[mesh] refine={refine}: {n} nodes, {len(edges)} edges, mean edge length h={h:.4f} (unit sphere)")

    k = (lmax + 1) ** 2  # number of eigenpairs covering degrees 0..lmax
    evals, evecs = np.linalg.eigh(L.toarray())
    evals, evecs = evals[:k], evecs[:, :k]

    # On a near-uniform triangular mesh, (L f)_i ~ -(3/2) h^2 (Laplace-Beltrami f)(x_i),
    # so lambda_k ~ c * l(l+1) with c ~ 1.5 h^2. Fit c on the l = 1 group (exact span).
    c = evals[1:4].mean() / 2.0
    print(f"[spectrum] fitted scale c={c:.5f}  vs  1.5*h^2={1.5 * h**2:.5f}")

    theta, phi = to_angles(verts)
    print("[eigvecs]  l  mult  mean lambda/c   l(l+1)   rel.err   energy in degree-l harmonics")
    rows = []
    for l in range(lmax + 1):
        idx = slice(l**2, (l + 1) ** 2)
        Y, _ = np.linalg.qr(real_sph_harm_basis(l, theta, phi))  # orthonormal basis of the span
        U = evecs[:, idx]
        energy = (np.linalg.norm(Y.T @ U, axis=0) ** 2).mean()  # each column of U has unit norm
        lam = evals[idx] / c
        target = l * (l + 1)
        rel = abs(lam.mean() - target) / target if target else 0.0
        rows.append((l, lam, energy))
        print(f"           {l:2d}  {2 * l + 1:4d}  {lam.mean():12.3f}  {target:7d}  {rel:8.2%}   {energy:.4f}")

    # --- Figure: staircase spectrum + energy per degree ---------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.6, 1]})
    idx_all = np.arange(k)
    stair = np.array([int(np.floor(np.sqrt(i))) for i in idx_all])
    ax1.step(idx_all, stair * (stair + 1), where="mid", color="#d76e1e", lw=2,
             label=r"sphere: $\ell(\ell+1)$, multiplicity $2\ell+1$")
    ax1.plot(idx_all, evals / c, "o", ms=3, color="#143c6e", label=r"graph Laplacian $L=D-A$ (rescaled)")
    ax1.set_xlabel("eigenvalue index $k$")
    ax1.set_ylabel(r"$\lambda_k / c$")
    ax1.set_title(f"Icosahedral mesh ({n} nodes): graph spectrum vs sphere")
    ax1.legend(frameon=False, fontsize=8, loc="upper left")

    ls = [r[0] for r in rows]
    ax2.bar(ls, [r[2] for r in rows], color="#143c6e")
    ax2.set_ylim(min(0.9, min(r[2] for r in rows) - 0.02), 1.0)
    ax2.set_xlabel(r"degree $\ell$")
    ax2.set_ylabel("energy in span$\\{Y_\\ell^m\\}$")
    ax2.set_title("Graph eigenvectors $\\approx$ spherical harmonics")
    fig.tight_layout()
    fig.savefig(out / "laplacian_spectrum.pdf")
    plt.close(fig)

    # --- Figure: a few eigenvectors on the globe ------------------------------
    picks = [1, 4, 9, 16, 36, 64]
    picks = [p for p in picks if p < k]
    lat, lon = 90 - np.degrees(theta), np.degrees(phi)
    fig, axes = plt.subplots(1, len(picks), figsize=(2.2 * len(picks), 1.6))
    for ax, p in zip(np.atleast_1d(axes), picks):
        ax.scatter(lon, lat, c=evecs[:, p], s=2, cmap="RdBu_r")
        ax.set_title(f"$u_{{{p}}}$  ($\\ell={int(np.sqrt(p))}$)", fontsize=9)
        ax.set_xticks([]), ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out / "laplacian_eigenvectors.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Experiment 3: per-pixel i.i.d. noise vs spherical-harmonic white noise
# ---------------------------------------------------------------------------


def noise_experiment(out: Path, res_deg: float = 2.0, lmax: int = 30, cap_deg: float = 10.0,
                     n_samples: int = 300, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    lats = np.arange(-90 + res_deg / 2, 90, res_deg)  # cell centres, avoids the pole singularity
    lons = np.arange(0, 360, res_deg)
    LAT, LON = np.meshgrid(lats, lons, indexing="ij")
    theta, phi = np.radians(90 - LAT).ravel(), np.radians(LON).ravel()
    area = np.cos(np.radians(LAT)).ravel()  # cell area is proportional to cos(latitude)

    # Spherical-harmonic white noise: i.i.d. N(0,1) coefficients for every (l, m), l <= lmax.
    basis = np.hstack([real_sph_harm_basis(l, theta, phi) for l in range(lmax + 1)])
    sh_noise = rng.standard_normal((n_samples, basis.shape[1])) @ basis.T
    pix_noise = rng.standard_normal((n_samples, theta.size))
    # Same total variance per unit area on average, so the two curves are comparable.
    sh_noise /= np.sqrt(np.average(sh_noise.var(axis=0), weights=area))

    # Variance of the area-weighted mean over an equal-area cap, centred at varying latitudes.
    xyz = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=1)
    centres = np.arange(0, 90 - cap_deg + 1e-9, 5.0)
    res = {"pixel": [], "sh": []}
    for lat0 in centres:
        t0 = np.radians(90 - lat0)
        c0 = np.array([np.sin(t0), 0.0, np.cos(t0)])
        w = area * (xyz @ c0 >= np.cos(np.radians(cap_deg)))
        w = w / w.sum()
        res["pixel"].append((pix_noise @ w).var())
        res["sh"].append((sh_noise @ w).var())

    pix, sh = np.array(res["pixel"]), np.array(res["sh"])
    print(f"[noise] cap-averaged variance, ratio (lat {centres[-1]:.0f}) / (lat 0):"
          f"  per-pixel i.i.d. = {pix[-1] / pix[0]:.2f},  spherical-harmonic = {sh[-1] / sh[0]:.2f}")

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(centres, pix / pix[0], "o-", color="#d76e1e", label="i.i.d. noise per lat–lon pixel")
    ax.plot(centres, sh / sh[0], "s-", color="#143c6e", label=f"spherical-harmonic white noise ($\\ell\\leq{lmax}$)")
    ax.axhline(1, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("latitude of the cap centre (°)")
    ax.set_ylabel("variance of cap mean\n(relative to equator)")
    ax.set_title(f"Large-scale noise power vs latitude ({cap_deg:.0f}° caps, {res_deg:.0f}° grid)", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "noise_isotropy.pdf")
    plt.close(fig)


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refine", type=int, default=4, help="icosahedron refinement level (4 -> 2562 nodes)")
    parser.add_argument("--lmax", type=int, default=10, help="highest degree l compared")
    parser.add_argument("--out", type=Path, default=here.parent / "slides" / "figures")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    laplacian_experiment(args.refine, args.lmax, args.out)
    noise_experiment(args.out)
    print(f"[done] figures written to {args.out}")


if __name__ == "__main__":
    main()
