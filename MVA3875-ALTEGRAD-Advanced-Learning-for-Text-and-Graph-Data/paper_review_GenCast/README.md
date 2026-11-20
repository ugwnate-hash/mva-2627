# Paper review: GenCast

**Paper.** I. Price, A. Sanchez-Gonzalez, F. Alet, T. R. Andersson, A. El-Kadi, D. Masters, T. Ewalds, J. Stott, S. Mohamed, P. Battaglia, R. Lam, M. Willson.
*Probabilistic weather forecasting with machine learning.* Nature 637, 84–90 (2025).
Preprint: [arXiv:2312.15796](https://arxiv.org/abs/2312.15796) · Code: [google-deepmind/graphcast](https://github.com/google-deepmind/graphcast)

**Presentation:** 3 December 2026, **6 minutes, 8 slides max** (backup slides after `\appendix` don't count). No experiments from the codebase.

## Layout

```
paper_review_GenCast/
├── paper/            # put the PDF (and supplementary material) here
├── notes/
│   ├── review_notes.md   # structured reading notes / critical review
│   ├── lexicon.md        # weather-forecasting vocabulary
│   └── spherical_harmonics.md  # deep dive: sphere ↔ graph Laplacian (course bridge)
├── explore/
│   ├── sphere_vs_graph_laplacian.py  # small numerical checks (~5 s), writes figures to slides/figures/
│   ├── icosphere_mesh_figures.py     # study figures: icosahedral mesh + its graph / Laplacian
│   └── figures/                      # output of icosphere_mesh_figures.py (not for the slides)
└── slides/
    ├── main.tex          # Beamer presentation
    ├── references.bib
    └── figures/          # figures exported from the paper or made by me
```

## Build the slides

```bash
cd slides
latexmk -pdf main.tex      # clean with: latexmk -c
```

## Checklist

- [ ] Download the paper + supplementary information into `paper/`
- [ ] First full read, fill in `notes/review_notes.md`
- [ ] Read the background papers (GraphCast, EDM diffusion)
- [ ] Draft the slides and export key figures to `slides/figures/`
- [ ] Verify every number marked `\verify{...}` in the slides against the paper
- [ ] Timed rehearsal
