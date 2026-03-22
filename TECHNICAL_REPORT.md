# Technical Report: Hybrid MR-FINDR Suite

## 1. Objective

This package implements a model family that combines two ideas:

- **FINDR-style local latent vector fields** for region-wise interpretable dynamics.
- **MR-LFADS(R)-style communication inferred from reconstructed activity** for better message identification.

The guiding decomposition is:

- intrinsic local dynamics,
- inter-region communication,
- known exogenous inputs,
- hidden inputs from unobserved sources,
- initial-condition variability,
- process noise,
- observation-side nuisance baseline.

The requested variant families are included directly:

- **A0 / A1 / A2**
- **B1 / B2 / B3**
- **C1 / C2**

---

## 2. Core mathematical design

For region `i`:

### 2.1 Latent state

\[
z_{i,t} \in \mathbb{R}^{L_i}
\]

### 2.2 Task-relevant activity

\[
\eta^{\text{task}}_{i,t} = C_i z_{i,t},
\qquad
r^{\text{task}}_{i,t} = \mathrm{softplus}(\eta^{\text{task}}_{i,t})
\]

### 2.3 Full reconstructed activity

\[
\eta^{\text{full}}_{i,t} = C_i z_{i,t} + d^{\text{nuis}}_{i,t},
\qquad
r^{\text{full}}_{i,t} = \mathrm{softplus}(\eta^{\text{full}}_{i,t})
\]

`d_nuis` depends on the baseline mode.

### 2.4 Messages

Messages are inferred from **task-relevant reconstructed source activity only**:

\[
q(m_{j\to i,t} \mid r^{\text{task}}_{j,t-1})
\]

or deterministically:

\[
m_{j\to i,t} = M_{j\to i}(r^{\text{task}}_{j,t-1})
\]

The one-bin lag is deliberate: it gives a clean causal message path and avoids same-time algebraic loops.

### 2.5 Local stochastic dynamics

\[
z_{i,t+1} = z_{i,t} + \alpha_i \Big[
\mu_i(z_{i,t})
+ B_i u^{\text{known}}_{i,t}
+ H_i u^{\text{hidden}}_{i,t}
+ \sum_{j\neq i} \Psi_{j\to i}(m_{j\to i,t})
\Big]
+ \sqrt{\alpha_i}\,\Sigma_i^{1/2}\epsilon_{i,t}
\]

with \(\alpha_i = dt / \tau_i\).

In the C family, communication can also modulate the target state multiplicatively through an additive-gain term.

---

## 3. Important design rule: nuisance baseline is excluded from message inference

This is the most important hybrid rule.

The model may reconstruct spikes using full rates:

\[
r^{\text{full}}_{i,t} = \mathrm{softplus}(C_i z_{i,t} + d^{\text{nuis}}_{i,t})
\]

but messages are **never** computed from the full reconstructed rates. They are computed from the task-relevant part only:

\[
r^{\text{task}}_{i,t} = \mathrm{softplus}(C_i z_{i,t})
\]

Otherwise nuisance PSTH structure can be mistaken for communication.

---

## 4. Variant families

## 4.1 A family: identification-first

Properties:

- stochastic messages,
- hidden inputs enabled,
- additive communication,
- intended as the default scientific compromise.

Variants:

- **A0**: no nuisance baseline
- **A1**: constant bias baseline
- **A2**: jointly learned smooth low-capacity nuisance baseline

This family is the default starting point.

---

## 4.2 B family: interpretability-first

Properties:

- deterministic messages,
- stronger regularization,
- simpler flow network sizes,
- cleaner vector-field interpretation.

Variants:

- **B1**: constant bias baseline
- **B2**: jointly learned nuisance baseline
- **B3**: pre-fit FINDR-style nuisance baseline

This family is best when the top priority is local field visualization and geometric interpretability.

---

## 4.3 C family: richer coupling

Properties:

- stochastic messages,
- additive + gain-like communication,
- intended when additive-only coupling is too restrictive.

Variants:

- **C1**: constant bias baseline
- **C2**: jointly learned nuisance baseline

This family should usually be tried after A2, not before.

---

## 5. Baseline modes

### Baseline-0
No nuisance baseline.

Use for:

- synthetic identification benchmarks,
- clean datasets with weak nuisance structure.

### Baseline-1
Constant per-neuron bias only.

Use for:

- minimal realistic control,
- simple ablations.

### Baseline-2
Jointly learned smooth low-capacity nuisance baseline.

Use for:

- repeated-trial real neural data,
- datasets with strong PSTH-like nuisance structure,
- the default flagship hybrid.

### Baseline-3
Pre-fit FINDR-style nuisance baseline.

Use for:

- highly repeated and well-aligned datasets,
- cases where nuisance structure is strong and stereotyped,
- interpretability-heavy analyses.

This is why B3 exists but A3/C3 are not treated as defaults.

---

## 6. Synthetic dataset design

The generator is not a toy spike simulator. It is designed to stress the exact ambiguities the hybrid model is supposed to resolve.

### 6.1 Three regions

The default synthetic benchmark has **3 regions**.

Each region has:

- a 2D latent state,
- a region-specific nonlinear local flow field,
- region-specific known inputs,
- region-specific hidden inputs,
- region-specific neuron readout.

### 6.2 Known inputs

Known inputs are encoded as generic per-bin channels so the same interface handles both:

- **event-like clicks**: sparse pulse channels,
- **step / onset inputs**: channels that turn on and stay on,
- **context cues**: sustained multi-channel task signals.

Default synthetic setup:

- **Region 0**: click-left, click-right, stimulus-on step
- **Region 1**: rule-A / rule-B sustained context channels
- **Region 2**: go-cue onset step

This directly addresses the concern that known inputs may have very different temporal semantics. In the math and code, they are all handled as explicit time-series input channels.

### 6.3 Hidden inputs

Hidden inputs are low-dimensional AR(1) processes mixed differently into each region.
They simulate drive from unobserved sources and structured within-trial variability.

### 6.4 Messages

Ground-truth communication is emitted from source task-relevant rates with one-bin lag.
Ground-truth per-edge message trajectories are saved in the dataset for recovery analysis.

### 6.5 Two scenarios

- `three_region_additive`
- `three_region_modulatory`

The modulatory scenario is included specifically so the C family has a nontrivial target.

### 6.6 Observation nuisance

The synthetic data also includes nuisance structure:

- smooth within-trial baseline,
- slow across-trial drift,
- neuron-specific offsets.

That makes baseline ablations meaningful.

---

## 7. Training objective

The implemented loss is:

\[
\mathcal{L} = \mathcal{L}_{\text{NLL}}
+ \beta_{\text{traj}} \mathcal{L}_{\text{traj}}
+ \beta_{\text{msg}} \mathcal{L}_{\text{msg}}
+ \beta_{\text{hidden}} \mathcal{L}_{\text{hidden}}
+ \beta_{z0} \mathcal{L}_{z0}
+ \beta_{\text{baseline}} \mathcal{L}_{\text{baseline}}
\]

with:

- Poisson negative log-likelihood,
- prior/posterior trajectory consistency,
- message KL or energy regularization,
- hidden-input KL regularization,
- initial-condition KL regularization,
- nuisance-baseline smoothness/L2 regularization when applicable.

The schedule ramps hidden-input, message, trajectory, and initial-condition penalties over training.

---

## 8. Implementation map

### `config.py`
Defines the full A/B/C + baseline family, with requested variants already pre-specified.

### `synthetic.py`
Generates the benchmark datasets.

### `data.py`
Loads `.npz` datasets, pads region-wise arrays, and optionally pre-fits the FINDR-style nuisance baseline.

### `model.py`
Implements the hybrid architecture in PyTorch.

### `train.py`
Handles optimization, evaluation, metric export, and artifact saving.

### `evaluate.py`
Computes effectome-style summary metrics and message recovery scores.

### `scripts/`
Provides runnable entry points for generation, training, and smoke testing.

---

## 9. Recommended roadmap

### Synthetic additive benchmark
Run:

- A0, A1, A2
- B1, B2, B3

Questions:

- Does A2 improve vector-field stability relative to A0/A1?
- Does B-family improve interpretability at the cost of fit or message recovery?
- Does B3 help when nuisance structure is strong and stereotyped?

### Synthetic modulatory benchmark
Run:

- A1 / A2
- C1 / C2

Questions:

- Does C-family recover coupling better when the truth is modulatory?
- Does additive-only A underfit the richer coupling structure?

### Real datasets
Suggested order:

1. **A2**
2. **A1**
3. **B2**
4. **C2**
5. **B3** only if repeated-trial nuisance is very strong and clearly stereotyped

---

## 10. Practical default recommendation

If you need one model to start with, use:

> **A2** = identification-first hybrid + jointly learned smooth nuisance baseline.

If you want the cleanest vector-field companion model, use:

> **B2**.

If you want to test whether communication is genuinely state-dependent or modulatory, use:

> **C2**.

That is the clean roadmap embodied by this repository.
