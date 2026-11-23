# RoPE: how relative position becomes real for the model

Personal notes and toy experiments. Not part of the graded project.

## 1. The problem

Attention scores a key against a query with a dot product, $q \cdot k$. If two tokens have the same content, they get the same score, wherever they are. Attention alone has no notion of order.

## 2. What RoPE does

Split the `head_dim` channels into pairs $(x_{2i}, x_{2i+1})$, for $i = 0 \dots d/2 - 1$. At position $t$, **rotate pair $i$ by the angle $t\,\theta_i$**:

$$
R(\alpha) = \begin{pmatrix} \cos\alpha & -\sin\alpha \\ \sin\alpha & \cos\alpha \end{pmatrix},
\qquad
\theta_i = \text{base}^{-2i/d}
$$

Queries and keys are rotated. Values are not.

## 3. The key formula: only the offset survives

For a query at position $m$ and a key at position $n$, with $\Delta = n - m$:

$$
\text{score}(\Delta)
= \sum_i \big(R(m\theta_i)\,q_i\big)^\top \big(R(n\theta_i)\,k_i\big)
= \sum_i q_i^\top R(\Delta\,\theta_i)\, k_i
$$

because $R(a)^\top R(b) = R(b - a)$. The absolute positions cancel and **only the offset $\Delta$ is left**.

Each term is a cosine in $\Delta$:

$$
q_i^\top R(\Delta\theta_i)\, k_i = |q_i|\,|k_i| \cos(\Delta\,\theta_i + \varphi_i)
$$

where $\varphi_i$ is the angle from $q_i$ to $k_i$. The **content** sets the amplitude $|q_i||k_i|$ and the phase $\varphi_i$. **RoPE** sets the speed $\theta_i$. The score is a sum of 16 cosines at different speeds, a Fourier-like sum in $\Delta$.

## 4. Fast pairs and slow pairs

$\theta_i$ is how fast pair $i$ turns, in radians per token. It does one full turn every $2\pi/\theta_i$ tokens (head_dim 32, base 10000):

| pair $i$ | 0 | 1 | 2 | 3 | 4 | 8 | 15 |
|---|---|---|---|---|---|---|---|
| full turn every … tokens | 6 | 11 | 20 | 35 | 63 | 628 | ≈ 35 000 |

- **Fast pairs** change a lot from one offset to the next, so they can tell positions apart: they carry the **positional** signal.
- **Slow pairs** barely turn over the context, so they act as **content-only** channels: same score at any distance.

## 5. Building a head by hand

Choose the key as the query **turned $L$ ticks ahead**:

$$
k_i = R(L\,\theta_i)\, q_i
\quad\Longrightarrow\quad
q_i^\top R(\Delta\theta_i)\, k_i = |q_i|^2 \cos\big((\Delta + L)\,\theta_i\big)
$$

Every pair reaches its maximum at $\Delta = -L$ at the same time, so the score peaks there: **the head attends to the token $L$ positions back.** The key $L$ positions back gets $L$ fewer ticks of rotation from RoPE, and its head start makes up for exactly that.

- $L = 1$: a **previous-token head** (case B).
- $L = 20$: a **distant-token head** (case C). Its score curve is case B's, shifted by 19.

Two rules from the experiments:

- **Put the energy on fast pairs.** Only they can tell $-L$ from $-L \pm 1$. Slow pairs add the same amount at every offset and flatten the peak. In a quick test with $L = 20$ (64 offsets, same total energy), the softmax weight on $\Delta = -20$ was 0.60 with fast pairs, against 0.16 with equal energy on all pairs.
- **Mix several speeds to avoid echoes.** Pair 0 alone lines up again every $2\pi \approx 6.3$ tokens. Several fast pairs only agree at $\Delta = -L$, which leaves smaller bumps elsewhere. After the softmax they matter little: in case B (offsets 0 to −29), the weight on $\Delta = -1$ is 27%, or 75% if the scores are 3 times larger, against 0.2% for the largest bump in that second case.

## 6. Without RoPE

The terms are just $q_i^\top k_i$, with no $\Delta$ anywhere, so the **score is flat**: every distance looks the same, equal to the RoPE score at $\Delta = 0$. A previous-token head can't exist.

## 7. What the model actually "sees"

Never the distance itself. For each query:

$$
\text{scores} \;\xrightarrow{\text{softmax}}\; \text{weights} \;\longrightarrow\; \text{output} = \sum_n \text{weight}_n \, v_n
$$

The values $v_n$ are not rotated, so they carry content only. Distance acts as a **filter**: it decides whose content gets through. Downstream, the model sees *what* was copied ("the previous token was X"), not *how far* it came from.

## 8. Who decides what

| Part | Role |
|---|---|
| RoPE | fixed set of rotation speeds $\theta_i$ (not learned) |
| $W_q$, $W_k$ | **where to look**: which pairs carry energy, which phase shift (e.g. $W_k \approx R(L\theta)\,W_q$ for "$L$ back"), which contents |
| $W_v$, $W_{out}$ | **what to copy**, and where to write it |

So RoPE lets $W_q$ and $W_k$ learn scores that depend on **relative** position, alone or combined with content.

## 9. Why rotations, in one line

If each position $t$ applies a matrix $P_t$ and we want $P_m^\top P_n$ to depend only on $n - m$, then (up to a change of basis that $W_q, W_k$ absorb) $P_t = R^t$ for one orthogonal matrix $R$, and every orthogonal matrix is a set of 2×2 rotations. **RoPE is the only option of this kind. The only real choice is the angles $\theta_i$.**

## 10. Limits of the toy examples

They use the same $q, k$ at every position (identical tokens), one head and one layer. They show what RoPE **can** express, not what training **does**. To check on a trained model: plot `self.weights` per head, average it along each diagonal (weight vs $\Delta$), and compare a few sequences to see how much content matters.

## Files

| File | What it shows |
|---|---|
| `rope_pair_contrib.py` → `rope_pair_contrib.png` | per-pair terms vs $i$ for 3 offsets, and the total score vs $\Delta$; cases A, B, C |
| same script → `rope_pair_contrib_norope.png` | the same without RoPE (flat), RoPE as a dashed reference |
| `rope_heatmap.py` → `rope_heatmap.png` | all terms for 30 offsets (rows = $\Delta$, columns = $i$), with and without RoPE |

Cases: **A** same content ($k = q$), **B** previous-token head ($L = 1$), **C** distant-token head ($L = 20$).

Run from `experiments/`:

```bash
uv run python rope_intuition/rope_pair_contrib.py
uv run python rope_intuition/rope_heatmap.py
```
