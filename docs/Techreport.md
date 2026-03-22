# Technical Report: Neuro-Voltron

## 1. Overview

Neuro-Voltron is a hybrid multi-region latent dynamical system that combines FINDR-style local vector fields with MR-LFADS-style communication inferred from reconstructed activity. The model family is indexed along two axes:

- **Communication family** (A / B / C): controls message stochasticity and coupling type
- **Baseline mode** (0 / 1 / 2 / 3): controls observation-side nuisance modeling

Implemented variants: **A0, A1, A2, B1, B2, B3, C1, C2**.

---

## 2. Notation

- $i \in \{1,\dots,R\}$: region index
- $t \in \{1,\dots,T\}$: time bin index
- $b \in \{1,\dots,B\}$: trial index
- $L_i$: latent dimension of region $i$
- $N_i$: number of neurons in region $i$
- $K_i$: known-input dimension of region $i$
- $U$: hidden-input dimension
- $M$: message dimension

Latent state: $z_{b,i,t} \in \mathbb{R}^{L_i}$, spike counts: $y_{b,i,t,n} \in \mathbb{N}$, known inputs: $k_{b,i,t} \in \mathbb{R}^{K_i}$, hidden inputs: $u_{b,i,t} \in \mathbb{R}^{U}$.

---

## 3. Model Specification

### 3.1 Initial condition

Each region has a pre-trial spike-history encoder producing $h^{\text{hist}}_{b,i} \in \mathbb{R}^{H_i}$ via a bidirectional GRU. The initial latent state posterior is:

$$
q(z_{b,i,0} \mid h^{\text{hist}}_{b,i})
= \mathcal{N}\big(\mu^{z0}_{b,i},\, \operatorname{diag}(\sigma^{z0}_{b,i})^2\big)
$$

with $\mu^{z0}_{b,i} = W^{z0}_{\mu,i} h^{\text{hist}}_{b,i} + b^{z0}_{\mu,i}$ and $\sigma^{z0}_{b,i} = \operatorname{softplus}(W^{z0}_{\sigma,i} h^{\text{hist}}_{b,i} + b^{z0}_{\sigma,i})$.

### 3.2 Task-relevant readout

Each region has a linear readout $C_i \in \mathbb{R}^{N_i \times L_i}$:

$$
\eta^{\text{task}}_{b,i,t} = C_i z_{b,i,t}, \qquad r^{\text{task}}_{b,i,t} = \operatorname{softplus}(\eta^{\text{task}}_{b,i,t}).
$$

Communication is derived from the task-relevant rate $r^{\text{task}}$, not the full nuisance-contaminated rate. This prevents nuisance baseline structure from being attributed to inter-region communication.

### 3.3 Messages

The message source is the task-relevant rate with one-bin lag: $s_{b,i,t} = r^{\text{task}}_{b,i,t}$, and $m_{j \to i,t}$ depends on $s_{j,t-1}$.

**Stochastic messages (families A, C):**

$$
q(m_{b,j\to i,t} \mid s_{b,j,t-1})
= \mathcal{N}\big(\mu^m_{b,j\to i,t},\, \operatorname{diag}(\sigma^m_{b,j\to i,t})^2\big)
$$

$$
\mu^m_{b,j\to i,t} = W^{m,\mu}_{j\to i} s_{b,j,t-1} + b^{m,\mu}_{j\to i}
$$

$$
\sigma^m_{b,j\to i,t} = \operatorname{softplus}(W^{m,\sigma}_{j\to i} s_{b,j,t-1} + b^{m,\sigma}_{j\to i})
$$

$$
m_{b,j\to i,t} = \mu^m_{b,j\to i,t} + \sigma^m_{b,j\to i,t} \odot \epsilon^m, \quad \epsilon^m \sim \mathcal{N}(0,I)
$$

**Deterministic messages (family B):**

$$
m_{b,j\to i,t} = W^{m}_{j\to i} s_{b,j,t-1} + b^{m}_{j\to i}
$$

### 3.4 Message mapping

Additive communication drive into region $i$:

$$
\Delta^{\text{msg}}_{b,i,t} = \sum_{j \neq i} A_{j\to i} m_{b,j\to i,t}, \qquad A_{j\to i} \in \mathbb{R}^{L_i \times M}
$$

Family C adds a gain-like map per edge: $g_{b,j\to i,t} = G_{j\to i} m_{b,j\to i,t}$, with total gain $\Gamma_{b,i,t} = \sum_{j \neq i} g_{b,j\to i,t}$.

### 3.5 Hidden inferred inputs

A controller GRU runs per region using the sequence encoder feature $e_{b,i,t}$ and current latent state:

$$
h^{\text{ctrl}}_{b,i,t} = \operatorname{GRU}(h^{\text{ctrl}}_{b,i,t-1},\, [e_{b,i,t}, z_{b,i,t}])
$$

$$
q(u_{b,i,t} \mid e_{b,i,t}, z_{b,i,t})
= \mathcal{N}(\mu^u_{b,i,t}, \operatorname{diag}(\sigma^u_{b,i,t})^2)
$$

The sampled hidden input enters latent space as $\Delta^{\text{hid}}_{b,i,t} = H_i u_{b,i,t} + c_i$.

### 3.6 Known-input drive

$$
\Delta^{\text{known}}_{b,i,t} = B_i k_{b,i,t} + d_i
$$

### 3.7 Prior local dynamics

The interpretable prior drift uses a gated relaxation MLP for each region:

$$
g_i(z) = \sigma(W^{g}_i z + b^g_i)
$$

$$
\tilde z_i(z) = \tanh(W^{o}_i \phi_i(z) + b^o_i)
$$

$$
f_i(z) = g_i(z) \odot (-z + \tilde z_i(z))
$$

**Families A, B (additive coupling):**

$$
\mu^{\text{prior}}_{b,i,t} = f_i(z_{b,i,t}) + \Delta^{\text{known}}_{b,i,t} + \Delta^{\text{msg}}_{b,i,t}
$$

**Family C (additive + gain coupling):**

$$
\mu^{\text{prior}}_{b,i,t} = f_i(z_{b,i,t}) + \Delta^{\text{known}}_{b,i,t} + \Delta^{\text{msg}}_{b,i,t} + \tanh(\Gamma_{b,i,t}) \odot z_{b,i,t}
$$

### 3.8 Posterior correction and latent update

During training, the model uses an inference-guided posterior drift:

$$
\mu^{\text{post}}_{b,i,t} = \mu^{\text{prior}}_{b,i,t} + \lambda_{\text{post}} \, c_i^{\text{post}}(z_{b,i,t}, e_{b,i,t}, k_{b,i,t}, \Delta^{\text{msg}}_{b,i,t})
$$

The latent update:

$$
z_{b,i,t+1} = z_{b,i,t}
+ \alpha_i \big( \mu^{\text{post}}_{b,i,t} + \Delta^{\text{hid}}_{b,i,t} \big)
+ \sqrt{\alpha_i} \, \sigma^{\text{proc}}_i \odot \epsilon_{b,i,t}
$$

where $\alpha_i = dt / \tau_i$ and $\epsilon_{b,i,t} \sim \mathcal{N}(0,I)$.

Known inputs and messages enter the interpretable prior field; hidden inputs are a separate inferred drive; encoder information appears only through the posterior correction.

### 3.9 Observation model

$$
\eta^{\text{full}}_{b,i,t} = \eta^{\text{task}}_{b,i,t} + d^{\text{nuis}}_{b,i,t}
$$

$$
r^{\text{full}}_{b,i,t} = \operatorname{softplus}(\eta^{\text{full}}_{b,i,t})
$$

$$
y_{b,i,t,n} \sim \operatorname{Poisson}(dt \cdot r^{\text{full}}_{b,i,t,n})
$$

---

## 4. Variant Families

### 4.1 Family A (identification-first)

Stochastic messages, additive coupling, hidden inputs enabled. Provides the best balance between dynamical interpretation and communication recovery.

- **A0**: no nuisance baseline (synthetic benchmarks)
- **A1**: constant per-neuron bias (minimal control)
- **A2**: jointly learned smooth baseline (flagship real-data model)

### 4.2 Family B (interpretability-first)

Deterministic messages, smaller flow networks, stronger regularization. Optimized for clean vector fields and geometric analysis.

- **B1**: constant bias
- **B2**: jointly learned baseline
- **B3**: pre-fit FINDR-style baseline (strongest nuisance separation)

### 4.3 Family C (richer coupling)

Stochastic messages with additive + gain-like modulation. For systems where communication gates or amplifies local dynamics.

- **C1**: constant bias
- **C2**: jointly learned baseline

---

## 5. Baseline / Nuisance Modes

### Mode 0: no baseline

$d^{\text{nuis}}_{b,i,t} = 0$. Strongest pressure on latent dynamics to explain all variance.

### Mode 1: constant per-neuron bias

$d^{\text{nuis}}_{b,i,t} = b_i, \quad b_i \in \mathbb{R}^{N_i}$

### Mode 2: jointly learned smooth baseline

$$
d^{\text{nuis}}_{b,i,t} = \Phi_t w^{\text{within}}_i + T_i(h^{\text{hist}}_{b,i})
$$

where $\Phi_t$ is a fixed raised-cosine time basis, $w^{\text{within}}_i$ are learned coefficients, and $T_i(h^{\text{hist}}_{b,i}) = W^{\text{trial}}_i \tanh(V^{\text{trial}}_i h^{\text{hist}}_{b,i}) + c^{\text{trial}}_i$ is an optional trial-varying term. Regularized with $L_2$ and smoothness penalties.

### Mode 3: pre-fit FINDR-style baseline

Nuisance rate is decomposed before training into across-trial ($r^{\text{across}}$, fit via radial basis + ridge regression) and within-trial ($r^{\text{within}}$, fit via raised-cosine basis + ridge regression) components. The result is fixed during model training.

---

## 6. Training Objective

$$
\mathcal{L}
=
\mathcal{L}_{\text{NLL}}
+ \beta_{\text{traj}} \mathcal{L}_{\text{traj}}
+ \beta_{\text{msg}} \mathcal{L}_{\text{msg}}
+ \beta_{\text{hid}} \mathcal{L}_{\text{hid}}
+ \beta_{z0} \mathcal{L}_{z0}
+ \beta_{\text{base}} \mathcal{L}_{\text{base}}
$$

| Term | Definition |
|------|-----------|
| $\mathcal{L}_{\text{NLL}}$ | Poisson negative log-likelihood: $-\sum_{b,i,t,n} \log p(y_{b,i,t,n} \mid r^{\text{full}}_{b,i,t,n})$ |
| $\mathcal{L}_{\text{traj}}$ | Trajectory consistency (prior vs posterior drift): $\frac{1}{2} \sum \frac{\alpha_i (\mu^{\text{post}} - \mu^{\text{prior}})^2}{(\sigma^{\text{proc}})^2}$ |
| $\mathcal{L}_{\text{msg}}$ | Message KL (A/C) or $L_2$ penalty (B) |
| $\mathcal{L}_{\text{hid}}$ | Hidden-input KL: $\operatorname{KL}(q(u) \| \mathcal{N}(0, \sigma_u^2 I))$ |
| $\mathcal{L}_{z0}$ | Initial-condition KL: $\operatorname{KL}(q(z_0) \| \mathcal{N}(0, \sigma_{z0}^2 I))$ |
| $\mathcal{L}_{\text{base}}$ | Baseline regularization (mode 2 only) |

All KL/regularization terms use a ramp schedule to prevent pathological variance allocation during early training.

---

## 7. Synthetic Data Generator

The generator produces 3-region datasets with known ground-truth dynamics, communication, and nuisance structure for benchmarking.

### `three_region_ring` (recommended)

Three regions connected in a uni-directional ring (R0 → R1 → R2 → R0):

| Region | Flow type | Description |
|--------|-----------|-------------|
| R0 | Limit cycle (Hopf normal form) | Stable orbit at radius 1 |
| R1 | Line attractor | Memory along z1, attraction to z2=0 |
| R2 | Double well | Two stable states at z1 = ±1 |

Ground-truth drift:

$$
\dot z_{i,t} = f_i(z_{i,t}) + B_i k_{i,t} + H_i h_{i,t} + \Delta^{\text{msg}}_{i,t}
$$

(additive scenario), or with an additional $\tanh(\Gamma_{i,t}) \odot z_{i,t}$ term (modulatory scenario).

Ground-truth messages use task-relevant rates with one-bin lag: $m^{\star}_{b,j\to i,t} = W^{\star}_{j\to i} r^{\text{task},\star}_{b,j,t-1}$.

Messages are bounded via $\tanh(z)$ source. All three active edges have coupling strength 0.4; inactive edges are exactly zero.

| Parameter | Value |
|-----------|-------|
| Latent dim per region | 2 |
| Neurons per region | 50 |
| Mean firing rate | ~1.6 Hz |
| Task variance fraction | 35-81% |
| Active edges | 3 (ring) |

### Known inputs

- Region 0: `click_left`, `click_right`, `stim_on`
- Region 1: `rule_A`, `rule_B`
- Region 2: `go_cue`

### Hidden inputs

Low-dimensional AR(1) processes: $h_t = \rho h_{t-1} + \sigma_h \epsilon_t$, mixed differently into each region.

### Nuisance baseline

Full log-rate: $\eta^{\text{full},\star}_{b,i,t} = C_i z^{\star}_{b,i,t} + d^{\text{within},\star}_{i,t} + d^{\text{across},\star}_{b,i} + c_i$.

---

## 8. Implementation Notes

The recurrence is sequential in time but the step body is fully batched:

- Latent state: `[B, R, L_max]` padded tensor
- Edge messages: single `BatchedEdges` call over all edges
- Flow evaluation: single `BatchedFlowMLP` call over all regions
- Controller, readout, and input projections: batched via `einsum`

Additional optimizations: fused `nn.GRU` encoders with packed sequences, cached basis functions, pre-allocated output tensors, one-time device placement for static tensors.
