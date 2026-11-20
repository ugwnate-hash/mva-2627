"""Study figures: what the icosahedral mesh is, and the graph its Laplacian is built on.

Reuses `icosphere` and `graph_laplacian` from sphere_vs_graph_laplacian.py, so the
figures show exactly the mesh/graph used in the spectrum experiment.

1_refinement.png           icosahedron refined 0..3 times
2_one_step.png             one subdivision step on a single face
3_graph_and_laplacian.png  the graph (refine=1), its adjacency A and L = D - A
4_ico_vs_latlon.png        node layout: icosphere vs lat-lon grid
5_uniformity.png           edge-length spread vs lat-lon spacing shrinking at the poles

Usage (from the repo root, inside WSL):
    .venv/bin/python <this file> [--out DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from sphere_vs_graph_laplacian import graph_laplacian, icosphere

_here = Path(__file__).resolve().parent
_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_parser.add_argument("--out", type=Path, default=_here / "figures")
out = _parser.parse_args().out
out.mkdir(parents=True, exist_ok=True)


def draw_mesh(ax, verts, faces, title, show_nodes=True):
    tri = verts[faces]
    pc = Poly3DCollection(tri, facecolor="#cfe0f3", edgecolor="#143c6e", lw=0.4, alpha=0.9)
    ax.add_collection3d(pc)
    if show_nodes:
        ax.scatter(*verts.T, s=6, c="#d76e1e", depthshade=False)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
    for f in (ax.set_xlim, ax.set_ylim, ax.set_zlim): f(-1, 1)
    ax.view_init(20, 30); ax.set_title(title, fontsize=9)

# 1. Refinement levels --------------------------------------------------------
fig = plt.figure(figsize=(13, 3.6))
for i, r in enumerate(range(4)):
    v, f = icosphere(r)
    _, e = graph_laplacian(len(v), f)
    ax = fig.add_subplot(1, 4, i + 1, projection="3d")
    draw_mesh(ax, v, f, f"refine={r}\n{len(v)} nodes, {len(e)} edges, {len(f)} faces", show_nodes=r < 3)
fig.suptitle("Icosahedral mesh: split every triangle into 4, push new midpoints onto the sphere", fontsize=10)
fig.tight_layout(); fig.savefig(out / "1_refinement.png", dpi=130); plt.close(fig)

# 2. One subdivision step, zoomed --------------------------------------------
v0, f0 = icosphere(0); v1, f1 = icosphere(1)
fig = plt.figure(figsize=(9, 4))
for k, (v, f, t) in enumerate([(v0, f0, "before: 1 face"), (v1, f1, "after: 4 faces")]):
    ax = fig.add_subplot(1, 2, k + 1, projection="3d")
    face = f0[0]
    # faces of level 1 that lie inside face 0 of level 0: centroid closest to that face's centroid
    c0 = v0[face].mean(0); c0 /= np.linalg.norm(c0)
    if k == 0:
        sel = f[[0]]
    else:
        cent = v[f].mean(1); cent /= np.linalg.norm(cent, axis=1, keepdims=True)
        sel = f[np.argsort(-(cent @ c0))[:4]]
    ax.add_collection3d(Poly3DCollection(v[sel], facecolor="#cfe0f3", edgecolor="#143c6e", lw=1.2))
    ids = np.unique(sel)
    ax.scatter(*v[ids].T, s=40, c=["#d76e1e" if i < 12 else "#2a9d55" for i in ids], depthshade=False)
    lim = np.abs(v[ids] - c0).max() + 0.05
    for f_, cc in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), c0): f_(cc - lim, cc + lim)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); ax.view_init(*np.degrees([np.arcsin(c0[2]), np.arctan2(c0[1], c0[0])]))
    ax.set_title(t + "\norange = original 12 vertices, green = new midpoints", fontsize=9)
fig.tight_layout(); fig.savefig(out / "2_one_step.png", dpi=130); plt.close(fig)

# 3. The graph itself (refine=1): 3D + adjacency + Laplacian ------------------
v, f = icosphere(1)
L, e = graph_laplacian(len(v), f)
deg = np.asarray(L.diagonal())
fig = plt.figure(figsize=(14, 4.4))
ax = fig.add_subplot(1, 3, 1, projection="3d")
ax.add_collection3d(Line3DCollection(v[e], colors="#143c6e", lw=0.8))
sc = ax.scatter(*v.T, c=deg, cmap="coolwarm", s=40, depthshade=False, vmin=5, vmax=6)
for i in range(len(v)):
    ax.text(*(v[i] * 1.12), str(i), fontsize=6)
ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); ax.view_init(20, 30)
ax.set_title(f"Graph G=(V,E), refine=1: {len(v)} nodes, {len(e)} edges\nblue = degree 5 (12 original vertices), red = degree 6", fontsize=9)
ax = fig.add_subplot(1, 3, 2)
A = np.diag(deg) - L.toarray()
ax.imshow(A, cmap="Greys"); ax.set_title("adjacency matrix A (black = edge)", fontsize=9)
ax.set_xlabel("node j"); ax.set_ylabel("node i")
ax = fig.add_subplot(1, 3, 3)
Ld = L.toarray(); m = np.abs(Ld).max()
im = ax.imshow(Ld, cmap="RdBu_r", vmin=-m, vmax=m); plt.colorbar(im, ax=ax, fraction=0.046)
ax.set_title("L = D - A: degree on diagonal, -1 per edge\neach row sums to 0", fontsize=9)
fig.tight_layout(); fig.savefig(out / "3_graph_and_laplacian.png", dpi=130); plt.close(fig)

# 4. Why not a lat-lon grid? node density ------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), subplot_kw={"projection": "3d"})
v, f = icosphere(3)
draw_mesh(axes[0], v, f, f"icosphere refine=3 ({len(v)} nodes)\nnodes spread almost evenly", show_nodes=True)
lat = np.radians(np.arange(-90, 91, 7.5)); lon = np.radians(np.arange(0, 360, 7.5))
LA, LO = np.meshgrid(lat, lon)
xyz = np.stack([np.cos(LA) * np.cos(LO), np.cos(LA) * np.sin(LO), np.sin(LA)], -1).reshape(-1, 3)
ax = axes[1]
ax.scatter(*xyz.T, s=4, c="#d76e1e", depthshade=False)
for la in lat: ax.plot(np.cos(la) * np.cos(lon), np.cos(la) * np.sin(lon), np.sin(la) + 0 * lon, c="#143c6e", lw=0.4)
for lo in lon: ax.plot(np.cos(lat) * np.cos(lo), np.cos(lat) * np.sin(lo), np.sin(lat), c="#143c6e", lw=0.4)
ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); ax.view_init(35, 30)
for f_ in (ax.set_xlim, ax.set_ylim, ax.set_zlim): f_(-1, 1)
ax.set_title(f"lat-lon grid 7.5° ({len(xyz)} points)\npoints crowd together at the poles", fontsize=9)
fig.tight_layout(); fig.savefig(out / "4_ico_vs_latlon.png", dpi=130); plt.close(fig)

# 5. Uniformity numbers -------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for r in range(1, 6):
    v, f = icosphere(r); _, e = graph_laplacian(len(v), f)
    le = np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1)
    axes[0].hist(le / le.mean(), bins=40, histtype="step", lw=1.5, label=f"refine={r} ({len(v)} nodes)")
axes[0].set_xlabel("edge length / mean edge length"); axes[0].set_ylabel("count"); axes[0].set_yscale("log")
axes[0].set_title("icosphere: all edges within about ±10% of the mean", fontsize=9); axes[0].legend(fontsize=7, frameon=False)
latd = np.arange(-88.75, 90, 2.5)
axes[1].plot(latd, np.cos(np.radians(latd)) / 1, c="#d76e1e")
axes[1].set_xlabel("latitude (°)"); axes[1].set_ylabel("east-west spacing / spacing at equator")
axes[1].set_title("lat-lon grid: spacing shrinks to 0 at the poles", fontsize=9)
fig.tight_layout(); fig.savefig(out / "5_uniformity.png", dpi=130); plt.close(fig)

print(f"[done] figures written to {out}")
