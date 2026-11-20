# Spherical harmonics: the bridge between GenCast and the course

**In three lines.** Spherical harmonics are the eigenfunctions of the Laplacian *on the sphere*; graph Fourier modes are the eigenvectors of the Laplacian *on a graph*. The icosahedral mesh that GenCast's GNN runs on is a graph whose Laplacian eigenvectors **are** (approximately) spherical harmonics. I checked this numerically below. So "spectral graph theory" (course) and "spectral methods on the sphere" (weather) are the same mathematics at two levels of discretisation.

Items marked ⚠️ are claims about the GenCast paper to check against the PDF.

---

## 1. Spherical harmonics in one page

**Fourier analogy.** On a circle, $e^{ik\varphi}$ are the eigenfunctions of $\frac{d^2}{d\varphi^2}$ with eigenvalue $-k^2$, and $|k|$ is a frequency. On the unit sphere $S^2$ the role of $\frac{d^2}{d\varphi^2}$ is played by the **Laplace–Beltrami operator** $\Delta_{S^2}$, and its eigenfunctions are the spherical harmonics:
$$
\Delta_{S^2} Y_\ell^m = -\ell(\ell+1)\,Y_\ell^m,\qquad
Y_\ell^m(\theta,\varphi) = N_{\ell m}\,P_\ell^m(\cos\theta)\,e^{im\varphi},
$$
with $\theta$ the colatitude, $\varphi$ the longitude, $P_\ell^m$ the associated Legendre functions, $\ell\ge 0$ and $|m|\le\ell$.

**Facts to remember.**
1. **Basis.** $\{Y_\ell^m\}$ is an orthonormal basis of $L^2(S^2)$: $f=\sum_{\ell,m}\hat f_{\ell m}Y_\ell^m$, with $\hat f_{\ell m}=\int_{S^2} f\,\overline{Y_\ell^m}\,d\Omega$.
2. **Degeneracy.** The eigenvalue $\ell(\ell+1)$ has multiplicity $2\ell+1$. The degree-$\ell$ space $\mathcal H_\ell=\mathrm{span}\{Y_\ell^m\}_m$ is what matters, not the individual $m$.
3. **Scale.** $\ell$ is a *total* wavenumber: wavelength $\approx 2\pi R/\ell \approx 40\,000\text{ km}/\ell$ on Earth. $m$ only sets the orientation (zonal vs meridional patterns).
4. **Smoothness.** $\int_{S^2}|\nabla f|^2\,d\Omega = \sum_{\ell,m}\ell(\ell+1)\,|\hat f_{\ell m}|^2$. Smooth fields put their energy in low $\ell$.
5. **Rotations.** A rotation $R\in SO(3)$ maps $\mathcal H_\ell$ to itself (it mixes the $m$'s, through Wigner D-matrices, but never the $\ell$'s). Hence the **angular power spectrum** $C_\ell=\frac{1}{2\ell+1}\sum_m|\hat f_{\ell m}|^2$ is rotation-invariant.
6. **Convolution theorem (Funk–Hecke).** Convolving with a *zonal* (axially symmetric) kernel multiplies each $\hat f_{\ell m}$ by a factor $g_\ell$ that depends only on $\ell$. Example: the heat kernel, $e^{t\Delta}$ ⇒ $g_\ell = e^{-t\ell(\ell+1)}$.
7. **Transform cost.** A band-limit $L$ needs a grid of about $2L\times 2L$ points. The naive spherical-harmonic transform (SHT) costs $O(L^3)$: an FFT in longitude plus a Legendre transform in latitude. A 0.25° grid gives $L\approx 720$.

---

## 2. The graph side (ALTEGRAD)

For a graph with adjacency $A$ and degree matrix $D$, the combinatorial Laplacian $L=D-A$ satisfies
$$
x^\top L x = \sum_{(i,j)\in E}(x_i-x_j)^2 ,
$$
the discrete Dirichlet energy. Its eigenvectors $L u_k=\lambda_k u_k$ form the **graph Fourier basis**: small $\lambda_k$ = smooth signals, large $\lambda_k$ = oscillating ones. The **graph Fourier transform** is $\hat x = U^\top x$. Spectral clustering (Lab 4) uses the first eigenvectors; spectral GNNs filter in this basis.

### Dictionary

| Concept | Sphere $S^2$ | Graph $G$ |
|---|---|---|
| Operator | $-\Delta_{S^2}$ | $L = D-A$ (or $L_{sym}$, $L_{rw}$) |
| Eigenpairs | $Y_\ell^m$, $\ell(\ell+1)$, multiplicity $2\ell+1$ | $u_k$, $\lambda_k$ |
| Smoothness | $\int\lvert\nabla f\rvert^2=\sum\ell(\ell+1)\lvert\hat f_{\ell m}\rvert^2$ | $x^\top Lx=\sum_{ij}(x_i-x_j)^2=\sum_k\lambda_k\hat x_k^2$ |
| Fourier transform | SHT, $\hat f_{\ell m}$ | $\hat x=U^\top x$ |
| Constant mode | $Y_0^0$, eigenvalue 0 | $\mathbf 1$, eigenvalue 0 (Lab 4, Q2) |
| Isotropic filtering | zonal convolution, $\hat f_{\ell m}\mapsto g_\ell\hat f_{\ell m}$ | spectral filter $g(L)=U\,g(\Lambda)\,U^\top$ |
| Local filters | polynomial in $\Delta$ | polynomial in $L$ = $K$-hop (ChebNet, GCN) |
| Diffusion | heat kernel $e^{t\Delta}$ | $e^{-tL}$ (graph diffusion kernels) |
| Symmetry | rotations $SO(3)$ | graph automorphisms / permutation equivariance |
| Learning on it | Spherical CNNs, SFNO | GNNs, graph transformers |

One extra link to GenCast: a $K$-step polynomial filter $\sum_{k\le K}\theta_k L^k$ only mixes information within $K$ hops. GenCast's processor restricts attention to a ⚠️ $k$-hop mesh neighbourhood, which is the same kind of locality, but with data-dependent weights instead of a fixed polynomial.

---

## 3. Numerical check: the icosahedral mesh "knows" the spherical harmonics

Script: [`explore/sphere_vs_graph_laplacian.py`](../explore/sphere_vs_graph_laplacian.py) (about 5 s). Setup: icosahedron refined 4 times = **2562 nodes, 7680 edges**, unweighted graph, $L=D-A$, dense eigendecomposition of the first $(\ell_{\max}+1)^2=121$ eigenpairs.

Figure: [laplacian_spectrum.pdf](../slides/figures/laplacian_spectrum.pdf) (eigenvector maps: [laplacian_eigenvectors.pdf](../slides/figures/laplacian_eigenvectors.pdf))

| $\ell$ | mult. $2\ell+1$ | mean $\lambda/c$ | $\ell(\ell+1)$ | rel. error | energy of the eigenvectors in $\mathcal H_\ell$ |
|---|---|---|---|---|---|
| 0 | 1 | 0.000 | 0 | — | 1.0000 |
| 1 | 3 | 2.000 | 2 | 0 (fit) | 0.9996 |
| 2 | 5 | 5.973 | 6 | 0.46% | 0.9970 |
| 3 | 7 | 11.969 | 12 | 0.26% | 0.9972 |
| 4 | 9 | 19.915 | 20 | 0.43% | 0.9917 |
| 5 | 11 | 29.747 | 30 | 0.84% | 0.9892 |
| 6 | 13 | 41.476 | 42 | 1.25% | 0.9859 |
| 8 | 17 | 70.358 | 72 | 2.28% | 0.9757 |
| 10 | 21 | 106.055 | 110 | 3.59% | 0.9629 |

**What it shows.**
1. **Same staircase.** The graph eigenvalues come in plateaus of width 1, 3, 5, 7, … at heights proportional to $\ell(\ell+1)$, the sphere's spectrum.
2. **Same eigenvectors.** For every $\ell\le 10$, more than 96% of each eigenvector's energy lies in the span of the degree-$\ell$ spherical harmonics sampled at the nodes.
3. **The scale is predictable.** The fitted constant $c=0.00848$ matches $\tfrac32 h^2=0.00855$, with $h$ the mean edge length. A Taylor expansion on a triangular lattice with 6 neighbours gives $(Lf)_i\approx-\tfrac32h^2\,\Delta f(x_i)$, so the unweighted mesh Laplacian is a rescaled Laplace–Beltrami operator.
4. **The symmetry is visible.** From $\ell=3$ on, each plateau splits into sub-groups. Measured sizes: $\ell=3\to3+4$, $\ell=4\to4+5$, $\ell=5\to3+3+5$, $\ell=6\to1+3+4+5$, $\ell=7\to3+3+4+5$, $\ell=8\to3+4+5+5$. These are exactly the dimensions of the irreducible representations of the **icosahedral group** (1, 3, 3, 4, 5). The mesh only has the icosahedron's 120 symmetries, not all of $SO(3)$, so the $2\ell+1$ degeneracy breaks according to how $\mathcal H_\ell$ decomposes under that group. A nice exam-style remark: *the discretisation's symmetry group is written in its spectrum*.
5. **Errors grow with $\ell$.** At $\ell=10$, the wavelength (≈0.63 rad) spans only ≈8 edges ($h\approx0.075$), so the discrete operator is less accurate. This is the graph analogue of truncation in spectral NWP models.

This is an instance of a known result: graph Laplacians built on samples of a manifold converge to its Laplace–Beltrami operator (Belkin & Niyogi, 2008). DeepSphere (Perraudin et al., 2019; Defferrard et al., 2020) builds on exactly this to make graph convolutions approximately rotation-equivariant on the sphere.

---

## 4. Why it matters for GenCast

### 4.1 Isotropic diffusion noise ⚠️
A diffusion model learns to remove noise. If the noise statistics depend on latitude, the denoising task also depends on latitude.

- **Per-pixel i.i.d. noise on a lat–lon grid is not isotropic.** Cell area ∝ cos(latitude), so near the poles there are many more pixels per km², and a fixed-size region averages more independent values. My check (`noise_isotropy.pdf`, 2° grid, 10° caps): the variance of the cap-averaged noise at 80°N is **0.19×** its value at the equator. Large-scale noise power almost vanishes near the poles.
- **Spherical-harmonic white noise is isotropic.** Draw i.i.d. $\mathcal N(0,1)$ coefficients $\hat\varepsilon_{\ell m}$ for $\ell\le L$ and synthesise on the grid. Same check: ratio **1.02**, flat with latitude up to Monte-Carlo noise (±10% with 300 samples).
- **Theory.** A Gaussian field on the sphere is isotropic (rotation-invariant in law) *iff* its spherical-harmonic coefficients are independent with a variance $C_\ell$ that depends only on $\ell$. White noise is the case $C_\ell=$ const; a decaying $C_\ell$ would give spatially correlated ("coloured") noise.
- ⚠️ To check in the paper: whether GenCast's noise is white or coloured in $\ell$, and its truncation.

Figure: [noise_isotropy.pdf](../slides/figures/noise_isotropy.pdf)

### 4.2 Why an icosahedral mesh rather than the lat–lon grid
The lat–lon grid seen as a graph is strongly anisotropic: east–west neighbours get closer and closer towards the poles, and the pole is a single point with 1440 neighbours. Its Laplacian is a poor Laplace–Beltrami operator. The icosahedral mesh is nearly uniform, so (section 3) its graph Laplacian *is* a good one. Message passing or local attention on it therefore treats all regions of the globe almost alike. Note that GenCast does not use the Laplacian explicitly: this is an argument about what the graph *encodes*.

### 4.3 Evaluation via power spectra ⚠️
The spectrum $E(\ell)=\sum_m|\hat f_{\ell m}|^2$ shows *at which scales* a forecast has realistic energy. MSE-trained models (and ensemble means) lose energy at high $\ell$ (blur); individual GenCast samples should match the ERA5 spectrum. Check whether the paper or its supplementary shows such spectra.

### 4.4 Why GenCast's network is not itself spectral
A spectral alternative exists: SFNO (Bonev et al., 2023) learns in spherical-harmonic space. Trade-offs:
- *Spectral (SFNO):* global receptive field in one layer and exact rotation structure, but each layer needs an SHT (costly at $L\approx720$), and nonlinearities must be applied back on the grid.
- *Spatial GNN on a mesh (GraphCast/GenCast):* local and scales linearly with the number of nodes, with flexible attention; isotropy is only approximate, through the mesh.

This is exactly the spectral-vs-spatial GNN debate from the course (ChebNet/GCN vs message passing), transposed to the sphere.

---

## 5. Q&A preparation

- **"Why spherical harmonics and not a 2-D FFT on the lat–lon image?"** The lat–lon grid is a distorted chart of the sphere. A 2-D FFT treats the polar rows as being as long as the equator, so "frequency" no longer means physical scale. Spherical harmonics are intrinsic to the sphere.
- **"Is the GNN rotation-equivariant?"** Not exactly. The mesh only has icosahedral symmetry, and the spectrum's splitting shows it. In weather this is acceptable or even wanted, because the Earth is *not* rotation-invariant (land, poles, Coriolis): GenCast also feeds positional and geographic features.
- **"Connection to the course in one sentence?"** The mesh is a graph whose Laplacian eigenvectors are the spherical harmonics, so spectral graph theory on that graph *is* harmonic analysis on the sphere.

## References

- M. Belkin, P. Niyogi. *Towards a theoretical foundation for Laplacian-based manifold methods.* JCSS, 2008.
- T. Cohen, M. Geiger, J. Köhler, M. Welling. *Spherical CNNs.* ICLR, 2018.
- N. Perraudin, M. Defferrard, T. Kacprzak, R. Sgier. *DeepSphere: efficient spherical CNN with HEALPix sampling for cosmological applications.* Astronomy and Computing, 2019.
- M. Defferrard, M. Milani, F. Gusset, N. Perraudin. *DeepSphere: a graph-based spherical CNN.* ICLR, 2020.
- B. Bonev, T. Kurth, C. Hundt, J. Pathak, M. Baust, K. Kashinath, A. Anandkumar. *Spherical Fourier Neural Operators: learning stable dynamics on the sphere.* ICML, 2023.
