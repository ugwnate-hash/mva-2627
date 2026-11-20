# GenCast lexicon: weather-forecasting vocabulary

Domain terms used in the GenCast paper, explained for an ML reader.
Each entry: **definition** → **ML analogy** (when useful) → **role in GenCast**.
Items marked ⚠️ are claims about the paper to check against the PDF (`paper/`).

---

## 1. Initial conditions and data

### Data assimilation (DA)
**Definition.** Estimating the current state of the atmosphere by combining two imperfect sources:
1. a short-range forecast from the previous cycle (the **background** or *first guess*), and
2. new **observations** (satellites, radiosondes, aircraft, surface stations, ...), which are sparse, noisy and only indirectly related to the model variables.

The two are weighted by their error covariances (background error covariance **B**, observation error covariance **R**). The result is the **analysis**. Operational centres repeat this in **cycles** (e.g. every 6 or 12 h): analysis → short forecast → background for the next cycle.

**Main methods.**
- *3D-Var / 4D-Var*: variational methods that minimise a cost function
  $J(\mathbf{x}) = (\mathbf{x}-\mathbf{x}_b)^\top \mathbf{B}^{-1}(\mathbf{x}-\mathbf{x}_b) + (\mathbf{y}-H(\mathbf{x}))^\top \mathbf{R}^{-1}(\mathbf{y}-H(\mathbf{x}))$,
  where $H$ is the **observation operator** (maps the model state to what an instrument would measure). 4D-Var also fits observations spread over a time window, using the forecast model as a constraint. ECMWF uses 4D-Var.
- *Ensemble Kalman filter (EnKF)*: estimates **B** from an ensemble of forecasts.

**ML analogy.** Bayesian posterior inference over the state: prior = background, likelihood = observations, analysis ≈ MAP estimate (4D-Var) or a posterior mean (Kalman filter).

**Role in GenCast.** GenCast does **not** do data assimilation. It needs an analysis as input (ERA5 for training, operational analyses at deployment). Its forecasts therefore still depend on a traditional NWP + DA system. This is a key limitation to mention.

### Analysis
The best estimate of the atmospheric state at a given time, produced by data assimilation. It is the "initial condition" of a forecast, and it is also commonly used as the "ground truth" to verify forecasts against.

### Ensemble analysis
**Definition.** A *set* of analyses instead of a single one, produced by an **ensemble data assimilation** system. Each member runs its own DA cycle with perturbed inputs: perturbed observations (noise drawn from **R**), perturbed sea-surface temperatures, and stochastic model perturbations. The members end up slightly different. Their **mean** is a good state estimate, and their **spread** estimates the analysis uncertainty: large where observations are scarce or the flow is unstable, small where it is well observed.

**At ECMWF** this is the **EDA (Ensemble of Data Assimilations)**:
- *Operational EDA*: feeds the background-error statistics of the main 4D-Var and supplies the initial-condition perturbations of the ENS ensemble.
- *ERA5 EDA*: 10 members at reduced resolution (≈63 km, versus ≈31 km for the main ERA5 analysis).

**ML analogy.** Samples from the posterior over the initial state, rather than a single point estimate.

**Role in GenCast.** ⚠️ To check in the paper: GenCast's ensemble members are initialised from ERA5 plus perturbations derived from the ERA5 EDA. This means initial-condition uncertainty comes from the ensemble analysis, while the diffusion model adds forecast (model) uncertainty on top.

### Reanalysis
A consistent historical record of the atmosphere, produced by re-running a *single, fixed* DA system and model over decades of archived observations. Operational analyses change whenever the system is upgraded; a reanalysis does not, so it is the standard training set for ML weather models.

### ERA5
ECMWF's fifth-generation global reanalysis (1940–present, hourly, ≈0.25°/31 km, 137 model levels, usually provided on 37 pressure levels). It is GenCast's training data and its evaluation ground truth.

### Background / first guess
The short-range forecast from the previous DA cycle, used as the prior in the next analysis.

### Observation operator ($H$)
Maps a model state to the quantity an instrument would measure (e.g. temperature profile → satellite radiance). It lets the system assimilate indirect observations.

---

## 2. Forecasting systems

### NWP (Numerical Weather Prediction)
Forecasting by numerically integrating the physical equations of the atmosphere (fluid dynamics + thermodynamics + parameterisations). This is the "physics-based" baseline that ML models are compared with.

### Parameterisation
An approximate sub-model for processes smaller than the grid (convection, clouds, turbulence, radiation). Parameterisations are a major source of **model error**.

### Deterministic forecast
A single forecast trajectory. Example: ECMWF **HRES**; ML example: GraphCast.

### HRES
ECMWF's high-resolution deterministic forecast. **HRES-fc0** is HRES at lead time 0, i.e. essentially the operational analysis, used as ground truth when verifying ECMWF forecasts.

### Ensemble forecast / ENS
Several forecasts (**members**) run from perturbed initial conditions, often with perturbed model physics too, to sample the distribution of possible futures. **ENS** is ECMWF's operational ensemble (50 perturbed members + 1 control) and is the main baseline in the GenCast paper.

### Initial-condition perturbations
Ways to make ENS members differ at time 0:
- **EDA perturbations**, which represent analysis uncertainty (see *Ensemble analysis*);
- **Singular vectors**, the perturbations that grow fastest over the first ≈48 h.

### Model uncertainty / stochastic physics (SPPT)
Random perturbations added to the parameterised tendencies during the forecast, to represent model error. In GenCast, the diffusion model's sampling noise plays this role.

### Control member
The ensemble member run from the *unperturbed* analysis.

### Lead time
The time elapsed since initialisation (e.g. +24 h, +10 days). Skill is usually reported as a function of lead time.

### Medium-range
Lead times of roughly 3–15 days. This is GenCast's target range.

### Autoregressive rollout
Producing a long forecast by feeding each predicted state back in as the next input (12 h steps for GenCast). Errors can build up along the trajectory. For a deterministic model, rolling out is just repeated function application.

### Autoregressive sampling
**Definition.** The probabilistic version of a rollout. Instead of predicting one next state, the model *samples* it from a conditional distribution, and the sample becomes the input for the following step:
$$
\mathbf{X}^{t+1} \sim p_\theta\big(\cdot \mid \mathbf{X}^{t}, \mathbf{X}^{t-1}\big), \qquad t = 0, 1, \dots, T-1 .
$$
Chaining the steps samples a whole trajectory from the joint distribution, by the Markov factorisation
$p(\mathbf{X}^{1:T}\mid \mathbf{X}^{0},\mathbf{X}^{-1}) = \prod_t p(\mathbf{X}^{t+1}\mid \mathbf{X}^{t},\mathbf{X}^{t-1})$.

**How it works in GenCast.**
- Each 12 h step is itself a *full diffusion sampling*: start from noise and run the ODE/SDE solver (⚠️ ≈20 solver steps) to obtain the next state.
- A 15-day forecast = 30 steps. One ensemble member therefore costs about 30 × (number of solver steps) calls to the denoiser.
- An **ensemble** = N independent trajectories, using different noise and (⚠️) different initial-condition perturbations from the ensemble analysis.

**Why it matters.**
- *Coherent trajectories.* Each member is a physically consistent path through time, not a set of independent samples per lead time. This is what makes it possible to track a cyclone or sum wind power over time within one member.
- *No blurring.* A deterministic MSE model predicts the conditional mean, which gets smoother at each step. Sampling keeps realistic small-scale structure at every lead time.
- *Uncertainty grows naturally.* Sampling noise accumulates along the trajectory, so the spread between members grows with lead time, as it should for a chaotic system.

**Pitfalls.**
- *Exposure bias / distribution shift.* Training conditions on true ERA5 states (teacher forcing), but at test time the model conditions on its own samples. Small errors can push the inputs off the training distribution and compound. (⚠️ To check: GraphCast fine-tuned on multi-step rollouts to reduce this, while GenCast trains on single steps only.)
- *Cost.* Sequential in time, and each step is an iterative sampler, so the cost is much higher than one forward pass per step.

**ML analogy.** Like token-by-token sampling in a language model: each generated token is fed back in, and different random seeds give different coherent continuations of the same prompt (= different ensemble members from the same initial state).

### Chaos / predictability limit
Small initial errors grow exponentially, so deterministic skill is lost after ~2 weeks. This is the fundamental motivation for probabilistic forecasting.

---

## 3. Variables and grids

### Pressure levels
Vertical coordinate given as pressure (hPa) instead of height. Example: 500 hPa ≈ 5.5 km altitude, 850 hPa ≈ 1.5 km.

### Surface vs atmospheric variables
- **Surface** (single-level): 2 m temperature (t2m), 10 m wind (u10/v10), mean sea-level pressure (MSLP), total precipitation, sea-surface temperature.
- **Atmospheric** (per pressure level): temperature, wind (u, v, w), geopotential, specific humidity.

### Geopotential (z), e.g. Z500
Gravitational potential energy per unit mass at a pressure level, roughly proportional to the height of that pressure surface. Z500 is the classic headline variable for large-scale circulation skill.

### Equiangular latitude–longitude grid
A regular grid in latitude/longitude, e.g. 0.25° → 721 × 1440 points. Grid cells shrink towards the poles, so latitude-weighting is needed in metrics and losses.

### Icosahedral mesh
A near-uniform mesh on the sphere, obtained by repeatedly subdividing an icosahedron. GenCast's GNN processor operates on this mesh (see the slides).

### Spherical harmonics
*Deep dive with numerical checks: [spherical_harmonics.md](spherical_harmonics.md).*

**Definition.** The "Fourier basis" of the sphere. The functions $Y_\ell^m(\theta,\varphi)$ are the eigenfunctions of the Laplace–Beltrami operator (the Laplacian on the sphere):
$$
\Delta_{S^2} Y_\ell^m = -\ell(\ell+1)\,Y_\ell^m, \qquad \ell = 0,1,2,\dots,\quad m = -\ell,\dots,\ell .
$$
They form an orthonormal basis of $L^2(S^2)$, so any field on the globe (temperature, geopotential, ...) can be expanded as
$$
f(\theta,\varphi) = \sum_{\ell=0}^{\infty}\sum_{m=-\ell}^{\ell} \hat f_{\ell m}\, Y_\ell^m(\theta,\varphi).
$$

**Reading the indices.**
- **Degree $\ell$ (total wavenumber)** sets the spatial *scale*: wavelength ≈ $2\pi R/\ell$ ≈ 40 000 km$/\ell$. Low $\ell$ = planetary-scale patterns; high $\ell$ = small-scale detail. A 0.25° grid resolves up to roughly $\ell \approx 720$.
- **Order $m$ (zonal wavenumber)** counts the oscillations around a circle of latitude. It sets the *orientation* of the pattern, not its scale.
- **Isotropy:** the eigenvalue depends only on $\ell$, so all $Y_\ell^m$ with the same $\ell$ have the same scale whatever their orientation or position. This is why spectral statistics can be "the same everywhere on the globe".

**Vocabulary that comes with it.**
- *Spectral transform / spectral model*: NWP models (including ECMWF's IFS) compute horizontal derivatives in spherical-harmonic space and the physics on a grid, switching between the two at every time step.
- *Truncation* (e.g. "T639", "TCo1279"): the maximum $\ell$ kept. It is the model's effective resolution.
- *Power spectrum* $E(\ell)=\sum_m |\hat f_{\ell m}|^2$: energy per scale. It is the standard diagnostic for **blurring**: MSE-trained models lose energy at high $\ell$, while a good generative model matches the ERA5 spectrum.

**Role in GenCast** (⚠️ to check in the paper).
- *Diffusion noise on the sphere:* noise is sampled in spherical-harmonic space (e.g. white noise: i.i.d. coefficients for every $(\ell,m)$ up to the grid's truncation) and then transformed to the lat–lon grid. Sampling i.i.d. Gaussian noise per grid pixel would be *non-uniform on the sphere*, because the lat–lon grid has far more points per km² near the poles. Spherical-harmonic noise is isotropic: the same statistics at the equator and at the poles.
- *Evaluation:* power spectra of GenCast samples vs ERA5 vs ENS, to show that samples keep realistic small-scale structure while the ensemble mean is smooth.

**ML analogy.** Fourier features / the DCT for images, but on the sphere. Diffusion in spectral space is also related to frequency-aware noise schedules for images.

**Link to ALTEGRAD (graphs).** Spherical harmonics are to the sphere what the **eigenvectors of the graph Laplacian** $L = D - A$ are to a graph: the graph Fourier basis, with small eigenvalues = smooth signals and large eigenvalues = oscillating ones. This is the same object used in spectral clustering (Lab 4) and spectral GNNs (ChebNet, GCN). On a fine icosahedral mesh, the low eigenvectors of the mesh's Laplacian approximate the low-degree spherical harmonics.

---

## 4. Verification (evaluation) vocabulary

### Skill / skill score
Performance relative to a reference: $\text{SS} = 1 - \text{score}/\text{score}_{\text{ref}}$. In GenCast, the reference is ENS.

### Scorecard
A grid of (variable × level × lead time) cells, coloured by relative skill against a baseline.

### CRPS (Continuous Ranked Probability Score)
A proper scoring rule for a predictive distribution $F$ and an observation $y$:
$\mathrm{CRPS}(F,y)=\mathbb{E}|X-y|-\tfrac12\mathbb{E}|X-X'|$.
Lower is better. It reduces to MAE for a deterministic forecast.

### Proper scoring rule
A score that is optimised in expectation only by forecasting the true distribution, so it cannot be gamed by hedging.

### Ensemble mean RMSE
RMSE of the average of the members. It measures how good the "best single guess" is.

### Spread
The standard deviation of the ensemble members around their mean, i.e. the forecast's own estimate of its uncertainty.

### Spread/skill ratio
Spread ÷ ensemble-mean RMSE. A ratio of ≈1 means the ensemble is well calibrated; <1 means **under-dispersive** (overconfident); >1 means **over-dispersive**.

### Rank histogram
A histogram of where the observation falls among the sorted members. Flat = calibrated; U-shaped = under-dispersive; dome-shaped = over-dispersive.

### Brier score
Mean squared error of probabilistic forecasts of a binary event (e.g. "wind > 25 m/s").

### Relative Economic Value (REV)
The value a user gets from acting on probabilistic forecasts in a cost/loss decision model, relative to climatology and to a perfect forecast. It is used to evaluate extreme-event forecasts.

### Climatology
The long-term statistics of the weather at a given place and time of year. It is the trivial baseline forecast and is also used to define extremes (e.g. the 99.99th percentile).

### Pooled / spatially aggregated CRPS
CRPS computed on quantities averaged or maxed over regions of increasing size. It tests whether the ensemble gets the *spatial structure* right, not just each grid point independently.

### Tropical cyclone track error
The distance between the predicted and observed cyclone centres, computed after running a cyclone **tracker** on each member.
