# GenCast: reading notes

## 1. One-paragraph summary

The paper proposes a new approach for weather forecasting as they introduce GenCast. Globaly, it beats ENS, the state-of-the-art NWP model as well as previous MLWP models. 

## 2. Problem setting

- Task: Predict the weather on Earth at mid-term resolution
- Inputs / outputs (variables, resolution, time step, horizon): Some atmospheric metrics at differents heights on a 2.5° grid. 12 hours time step for a total of 15 days predictions. 
- Why probabilistic (vs. deterministic, e.g. GraphCast): The philosophy of the model is to introduce incertainty through the sampling of Z^t on the sphere as noise. The conditioned denoiser then produces an ensemble (one possible trajectory in time) 
- Baseline: it competes with ECMWF ENS and DeepMind GraphCast

## 3. Method

### 3.1 Diffusion model
- Conditioning (which past states): The last two states (X^t is given by X^(t-1) and X^(t-2))
- Noise process / noise on the sphere: To be specified
- Training objective / preconditioning (EDM-style?): Gradient descent on a loss that penalize the denoiser at different noise levels
- Sampler and number of steps: ?? 

### 3.2 Denoiser architecture (graph part, most relevant for ALTEGRAD)
- Grid → mesh encoder:
- Mesh (icosahedral, refinement level, #nodes/#edges):
- Processor (graph transformer? attention neighbourhood?):
- Mesh → grid decoder:
- Differences w.r.t. GraphCast's message-passing processor:

### 3.3 Training
- Data (ERA5 years, splits): From 1950 to 2019 
- Two-stage training / resolutions:
- Compute:

## 4. Experiments

| Evaluation | Metric | Result | Figure / table |
|---|---|---|---|
| Marginal skill | CRPS | | |
| Ensemble mean | RMSE | | |
| Calibration | spread/skill, rank histograms | | |
| Spatial structure | pooled CRPS | | |
| Tropical cyclones | track error | | |
| Wind power | | | |
| Runtime | | | |

## 5. Critical review

### Strengths
- Very clear paper, the motivations are here and well introduced, as well as the metrics are well explained. The results are convincing especially regarding the fact that the model is being compared to a previous of their own model. 


### Weaknesses / limitations
-

### Questions I would ask the authors
-

### Possible extensions
-

## 6. Links with the ALTEGRAD course

- Message passing / GNNs:
- Graph transformers and attention:
- Other:

## 7. Related work to mention

- GraphCast (Lam et al., 2023)
- EDM (Karras et al., 2022)
- Other ML weather models (FourCastNet, Pangu-Weather, NeuralGCM, ...)
