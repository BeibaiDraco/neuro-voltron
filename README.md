# Technical Report: Neuro-Voltron

## 1. Executive summary

**Neuro-Voltron** is a hybrid multi-region latent dynamical system designed to combine two strengths:

1. **FINDR-style local vector fields** for region-wise interpretable latent dynamics.
2. **MR-LFADS(R)-style communication** in which messages are constrained by the source region's reconstructed task-relevant activity rather than by freer latent variables.

The model family is indexed by two axes:

- **A / B / C**: the **communication-and-coupling family**.
- **0 / 1 / 2 / 3**: the **baseline / nuisance model**.

The concrete variants implemented in the repository are:

- **A0, A1, A2**
- **B1, B2, B3**
- **C1, C2**

This report explains the exact mathematics implemented in the code, the scientific motivation for each variant, and why the current implementation is organized the way it is.

---

## 2. What was checked in the renamed repository

I checked the renamed **Neuro-Voltron** repository and confirmed that the major performance fixes are present in the code:

- the sequence encoder and history encoder now use fused `nn.GRU` with packed sequences,
- the recurrence body is tensorized across **regions** and **edges**,
- messages are computed in one batched edge module instead of per-edge Python calls,
- basis functions are cached instead of recomputed every forward pass,
- static tensors are moved to the device once rather than copied on every mini-batch.

I also validated the current code path by running a small synthetic **A2** forward-and-loss pass successfully. I did **not** rerun a full benchmark suite here, so the performance discussion below is based on code inspection plus that functional check, not on new timing numbers.

---

## 3. Design goal and decomposition

The guiding decomposition is:

- **local dynamics**: what a region would do on its own,
- **inter-region communication**: what observed regions transmit to one another,
- **known inputs**: experimentally controlled, explicitly supplied exogenous drives,
- **hidden inputs**: structured drive from unobserved sources,
- **initial condition variability**: slow trial-to-trial context,
- **process noise**: residual fast stochasticity,
- **baseline / nuisance activity**: observation-space rate structure that should not automatically be attributed to local computation or communication.

This decomposition is important because the same variance in the spikes can often be explained in several different ways. A good model should fit the data **and** make those explanations distinguishable.

---

## 4. Notation

Let:

- \(i \in \{1,\dots,R\}\) index regions,
- \(t \in \{1,\dots,T\}\) index time bins,
- \(b \in \{1,\dots,B\}\) index trials,
- \(L_i\) be the latent dimension of region \(i\),
- \(N_i\) be the number of neurons in region \(i\),
- \(K_i\) be the known-input dimension of region \(i\),
- \(U\) be the hidden-input dimension,
- \(M\) be the message dimension.

The main latent state is
\[
z_{b,i,t} \in \mathbb{R}^{L_i}.
\]

Observed spike counts are
\[
y_{b,i,t,n} \in \mathbb{N}.
\]

Known exogenous inputs are
\[
k_{b,i,t} \in \mathbb{R}^{K_i},
\]
and hidden inferred inputs are
\[
u_{b,i,t} \in \mathbb{R}^{U}.
\]

---

## 5. Unified mathematical model implemented in Neuro-Voltron

The implemented model is easiest to understand as a **prior dynamical system**, an **inference-side posterior correction**, and an **observation model**.

### 5.1 Initial condition from pre-trial history

Each region has a pre-trial spike-history encoder. Let
\[
h^{\text{hist}}_{b,i} \in \mathbb{R}^{H_i}
\]
be the history summary from a bidirectional GRU run on the history window.

The initial latent state has a diagonal Gaussian approximate posterior:
\[
q(z_{b,i,0} \mid h^{\text{hist}}_{b,i})
= \mathcal{N}\big(\mu^{z0}_{b,i},\, \operatorname{diag}(\sigma^{z0}_{b,i})^2\big),
\]
with
\[
\mu^{z0}_{b,i} = W^{z0}_{\mu,i} h^{\text{hist}}_{b,i} + b^{z0}_{\mu,i},
\qquad
\sigma^{z0}_{b,i} = \operatorname{softplus}\!\big(W^{z0}_{\sigma,i} h^{\text{hist}}_{b,i} + b^{z0}_{\sigma,i}\big).
\]

If `use_initial_condition=False`, the sampled \(z_{b,i,0}\) is replaced by zero in the forward rollout.

### 5.2 Task-relevant readout

Each region has a linear latent-to-neuron readout:
\[
\eta^{\text{task}}_{b,i,t} = C_i z_{b,i,t},
\qquad C_i \in \mathbb{R}^{N_i \times L_i}.
\]

The corresponding task-relevant reconstructed rate is
\[
r^{\text{task}}_{b,i,t} = \operatorname{softplus}(\eta^{\text{task}}_{b,i,t}).
\]

This quantity is crucial because communication is derived from the **task-relevant** source representation, not from the full nuisance-contaminated rate.

### 5.3 Message source representation

The code supports two source choices:

- `message_source = "task_rate"`: use \(r^{\text{task}}\),
- `message_source = "task_lograte"`: use \(\eta^{\text{task}}\).

All provided variants use:
\[
s_{b,i,t} = r^{\text{task}}_{b,i,t}.
\]

Messages are **lagged by one bin**:
\[
m_{j\to i,t} \text{ depends on } s_{j,t-1},
\]
which avoids same-time algebraic loops and gives a clean directed communication path.

### 5.4 Message inference / generation

For each directed edge \(j \to i\), define a low-dimensional message
\[
m_{b,j\to i,t} \in \mathbb{R}^M.
\]

#### Stochastic message family (A and C)

For A and C, the message distribution is diagonal Gaussian:
\[
q(m_{b,j\to i,t} \mid s_{b,j,t-1})
= \mathcal{N}\big(\mu^m_{b,j\to i,t},\, \operatorname{diag}(\sigma^m_{b,j\to i,t})^2\big),
\]
with
\[
\mu^m_{b,j\to i,t} = W^{m,\mu}_{j\to i} s_{b,j,t-1} + b^{m,\mu}_{j\to i},
\]
\[
\sigma^m_{b,j\to i,t} = \operatorname{softplus}\!\big(W^{m,\sigma}_{j\to i} s_{b,j,t-1} + b^{m,\sigma}_{j\to i}\big).
\]
Then
\[
m_{b,j\to i,t} = \mu^m_{b,j\to i,t} + \sigma^m_{b,j\to i,t} \odot \epsilon^m,
\qquad \epsilon^m \sim \mathcal{N}(0,I).
\]

#### Deterministic message family (B)

For B, messages are deterministic:
\[
m_{b,j\to i,t} = W^{m}_{j\to i} s_{b,j,t-1} + b^{m}_{j\to i}.
\]

This makes the communication channel easier to interpret geometrically, at the cost of giving up explicit posterior uncertainty over messages.

### 5.5 Mapping messages into target latent space

Every edge has an additive latent-space map:
\[
a_{b,j\to i,t} = A_{j\to i} m_{b,j\to i,t},
\qquad A_{j\to i} \in \mathbb{R}^{L_i \times M}.
\]

The total additive communication drive into target region \(i\) is
\[
\Delta^{\text{msg}}_{b,i,t} = \sum_{j \neq i} a_{b,j\to i,t}.
\]

For family C, each edge also has a gain-like map
\[
g_{b,j\to i,t} = G_{j\to i} m_{b,j\to i,t},
\qquad G_{j\to i} \in \mathbb{R}^{L_i \times M},
\]
with total gain signal
\[
\Gamma_{b,i,t} = \sum_{j \neq i} g_{b,j\to i,t}.
\]

### 5.6 Hidden inferred inputs

A controller GRU runs per region over time using the current inference signal and latent state:
\[
h^{\text{ctrl}}_{b,i,t} = \operatorname{GRU}\big(h^{\text{ctrl}}_{b,i,t-1},\, [e_{b,i,t}, z_{b,i,t}]\big),
\]
where \(e_{b,i,t}\) is the bidirectional sequence encoder feature at time \(t\).

The controller emits a diagonal Gaussian hidden input posterior:
\[
q(u_{b,i,t} \mid e_{b,i,t}, z_{b,i,t})
= \mathcal{N}\big(\mu^u_{b,i,t}, \operatorname{diag}(\sigma^u_{b,i,t})^2\big),
\]
with
\[
\mu^u_{b,i,t} = W^{u,\mu}_i h^{\text{ctrl}}_{b,i,t} + b^{u,\mu}_i,
\qquad
\sigma^u_{b,i,t} = \operatorname{softplus}\!\big(W^{u,\sigma}_i h^{\text{ctrl}}_{b,i,t} + b^{u,\sigma}_i\big).
\]

The sampled hidden input is mapped into latent space via
\[
\Delta^{\text{hid}}_{b,i,t} = H_i u_{b,i,t} + c_i.
\]

### 5.7 Known-input drive

Known exogenous input channels are mapped linearly into latent space:
\[
\Delta^{\text{known}}_{b,i,t} = B_i k_{b,i,t} + d_i.
\]

This design deliberately treats different temporal semantics in a unified way:

- a **click train** is just a sparse event-valued channel,
- a **stimulus onset that remains on** is just a step channel,
- a **task context** is just a sustained channel.

Mathematically, they all enter through \(k_{b,i,t}\); the interpretation comes from how the synthetic data or real dataset defines those channels.

### 5.8 Prior local dynamics

The prior drift is the interpretable local dynamical system. It has the form
\[
\mu^{\text{prior}}_{b,i,t} = f_i(z_{b,i,t}) + \Delta^{\text{known}}_{b,i,t} + \Delta^{\text{msg}}_{b,i,t}.
\]

The local intrinsic field \(f_i\) is implemented as a gated relaxation MLP:
\[
g_i(z) = \sigma(W^{g}_i z + b^g_i),
\]
\[
\tilde z_i(z) = \tanh\big(W^{o}_i \phi_i(z) + b^o_i\big),
\]
\[
f_i(z) = g_i(z) \odot (-z + \tilde z_i(z)).
\]

This means the model learns a region-wise nonlinear vector field that can be visualized as a low-dimensional flow.

For family C, the prior drift includes a gain-modulated term:
\[
\mu^{\text{prior}}_{b,i,t} = f_i(z_{b,i,t})
+ \Delta^{\text{known}}_{b,i,t}
+ \Delta^{\text{msg}}_{b,i,t}
+ \tanh(\Gamma_{b,i,t}) \odot z_{b,i,t}.
\]

This is the precise mathematical distinction between additive coupling (A/B) and additive-plus-gain coupling (C).

### 5.9 Posterior correction and latent update

The model does **not** roll out only the prior field during training. Instead, it uses an inference-guided posterior drift:
\[
\mu^{\text{post}}_{b,i,t} = \mu^{\text{prior}}_{b,i,t} + \lambda_{\text{post}} \, c_i^{\text{post}}(z_{b,i,t}, e_{b,i,t}, k_{b,i,t}, \Delta^{\text{msg}}_{b,i,t}),
\]
where \(c_i^{\text{post}}\) is another gated MLP and `posterior_correction_scale = \lambda_post` in the code.

The latent update is then
\[
z_{b,i,t+1} = z_{b,i,t}
+ \alpha_i \Big( \mu^{\text{post}}_{b,i,t} + \Delta^{\text{hid}}_{b,i,t} \Big)
+ \sqrt{\alpha_i} \, \sigma^{\text{proc}}_i \odot \epsilon_{b,i,t},
\]
with
\[
\alpha_i = \frac{dt}{\tau_i},
\qquad
\epsilon_{b,i,t} \sim \mathcal{N}(0,I).
\]

The key modeling choice here is:

- **known inputs** and **messages** enter the interpretable prior field,
- **hidden inputs** are kept as a separate inferred drive,
- **encoder information** appears only through the posterior correction.

This preserves a FINDR-like distinction between an interpretable dynamical law and an inference-side correction used during learning.

### 5.10 Observation model

The full log-rate is
\[
\eta^{\text{full}}_{b,i,t} = \eta^{\text{task}}_{b,i,t} + d^{\text{nuis}}_{b,i,t},
\]
and the full rate is
\[
r^{\text{full}}_{b,i,t} = \operatorname{softplus}(\eta^{\text{full}}_{b,i,t}).
\]

Observed spike counts are modeled as Poisson:
\[
y_{b,i,t,n} \sim \operatorname{Poisson}\big(dt \cdot r^{\text{full}}_{b,i,t,n}\big).
\]

This matters because the model explicitly allows the observation to contain structure that is **not** routed through the task latent state.

---

## 6. Why messages come from task-relevant reconstructed activity

The central hybrid rule is:
\[
\text{messages are inferred from } r^{\text{task}} \text{ or } \eta^{\text{task}}, \text{ not from } r^{\text{full}}.
\]

The reason is scientific, not cosmetic.

### 6.1 Why not from full reconstructed rates?

If messages were inferred from
\[
r^{\text{full}}_{b,i,t} = \operatorname{softplus}(C_i z_{b,i,t} + d^{\text{nuis}}_{b,i,t}),
\]
then nuisance baseline structure could masquerade as communication.

That would make it ambiguous whether an inferred message reflects:

- transmitted task-relevant activity,
- shared PSTH structure,
- slow across-trial drift,
- neuron-specific baseline modulation.

### 6.2 Why not directly from latent state?

If messages are inferred directly from \(z_{i,t}\), then communication can use latent directions that help the fit but are only weakly constrained by observed activity.

The design choice here is therefore:

- **latent state** for local computation,
- **task-relevant reconstructed activity** for communication.

This is the cleanest compromise between dynamical interpretability and communication identifiability.

---

## 7. Variant families A, B, and C

## 7.1 Family A: identification-first

### Definition

Family A uses:

- **stochastic messages**,
- **additive communication only**,
- **hidden inferred inputs enabled**,
- an interpretable prior flow plus posterior correction,
- the standard KL hierarchy favoring communication over hidden inputs.

The core equations are:
\[
q(m_{j\to i,t} \mid s_{j,t-1}) = \mathcal{N}(\mu^m, \operatorname{diag}(\sigma^m)^2),
\]
\[
\mu^{\text{prior}}_{i,t} = f_i(z_{i,t}) + B_i k_{i,t} + \sum_{j \neq i} A_{j\to i} m_{j\to i,t},
\]
\[
\mu^{\text{post}}_{i,t} = \mu^{\text{prior}}_{i,t} + \lambda_{\text{post}} c_i^{\text{post}}(z_{i,t}, e_{i,t}, k_{i,t}, \Delta^{\text{msg}}_{i,t}).
\]

### Why family A exists

A is the **default scientific compromise**:

- messages are still uncertain and therefore regularized probabilistically,
- communication remains low-dimensional and additive,
- the prior field remains interpretable,
- hidden inputs are available but penalized more strongly than messages.

### Intended use

Use A when you want the main result of the paper or analysis: the best balance between local dynamical interpretation and communication recovery.

---

## 7.2 Family B: interpretability-first

### Definition

Family B changes three things:

1. messages are **deterministic** rather than stochastic,
2. the flow networks are **smaller**,
3. hidden-input and state regularization are **stronger**.

The message equation becomes
\[
m_{j\to i,t} = W^m_{j\to i} s_{j,t-1} + b^m_{j\to i},
\]
with additive coupling only:
\[
\mu^{\text{prior}}_{i,t} = f_i(z_{i,t}) + B_i k_{i,t} + \sum_{j \neq i} A_{j\to i} m_{j\to i,t}.
\]

### Why family B exists

B is for the case where the primary output is not uncertainty-aware communication inference but **clean vector fields and geometric interpretation**.

Deterministic messages make it easier to inspect the map
\[
s_{j,t-1} \mapsto m_{j\to i,t} \mapsto \Delta^{\text{msg}}_{i,t}
\]
without an additional stochastic layer.

The stronger regularization and smaller hidden layers push the model toward simpler dynamical explanations.

### Intended use

Use B when the top priority is phase portraits, slow points, interpretable communication arrows, or comparing latent geometry across regions.

---

## 7.3 Family C: richer coupling

### Definition

Family C keeps the stochastic message machinery of A but augments communication with a gain-like target-state modulation:
\[
\mu^{\text{prior}}_{i,t} = f_i(z_{i,t}) + B_i k_{i,t} + \sum_{j \neq i} A_{j\to i} m_{j\to i,t} + \tanh\!\left(\sum_{j \neq i} G_{j\to i} m_{j\to i,t}\right) \odot z_{i,t}.
\]

### Why family C exists

Additive coupling assumes communication acts like a force or displacement. Some systems are better described by communication that **gates**, **amplifies**, or **suppresses** parts of the local vector field depending on the current state.

C is therefore the expressive extension for datasets where additive coupling is too restrictive.

### Intended use

Use C only after A is stable. It is more expressive, but that extra flexibility can make mechanistic interpretation harder if the dataset does not actually require modulatory coupling.

---

## 8. Baseline / nuisance modes 0, 1, 2, and 3

The nuisance term \(d^{\text{nuis}}\) is where the model family differs most across baseline modes.

## 8.1 Baseline-0: no nuisance baseline

### Definition
\[
d^{\text{nuis}}_{b,i,t} = 0.
\]
So
\[
\eta^{\text{full}}_{b,i,t} = C_i z_{b,i,t}.
\]

### Why

Use this when you want the cleanest identification benchmark and do **not** want the observation model to absorb extra structure.

### Strengths

- strongest pressure on the latent dynamics to explain the data,
- simplest interpretation,
- useful for synthetic communication benchmarks.

### Risks

If the real data have strong nuisance PSTHs or slow baseline drift, the model may incorrectly push that structure into the latent dynamics, hidden inputs, or communication.

---

## 8.2 Baseline-1: constant per-neuron bias

### Definition
For each region, learn a neuron-wise constant bias:
\[
d^{\text{nuis}}_{b,i,t} = b_i,
\qquad b_i \in \mathbb{R}^{N_i}.
\]
So
\[
\eta^{\text{full}}_{b,i,t} = C_i z_{b,i,t} + b_i.
\]

### Why

This gives the model a minimal amount of observation-side flexibility without introducing time-varying nuisance structure.

### Strengths

- strong control model,
- inexpensive,
- keeps interpretation simple.

### Risks

It cannot capture trial-varying or within-trial nuisance rate structure.

---

## 8.3 Baseline-2: jointly learned low-capacity nuisance baseline

### Definition
In the current code, baseline-2 is
\[
d^{\text{nuis}}_{b,i,t} = \Phi_t w^{\text{within}}_i + T_i(h^{\text{hist}}_{b,i}),
\]
where:

- \(\Phi_t\) is a fixed raised-cosine time basis,
- \(w^{\text{within}}_i\) are learned coefficients,
- \(T_i(h^{\text{hist}}_{b,i})\) is an optional low-rank trial term derived from the history summary.

More explicitly,
\[
T_i(h^{\text{hist}}_{b,i}) = W^{\text{trial}}_i \tanh(V^{\text{trial}}_i h^{\text{hist}}_{b,i}) + c^{\text{trial}}_i.
\]
This contribution is constant across time within a trial but varies from trial to trial.

So the full rate model is
\[
\eta^{\text{full}}_{b,i,t} = C_i z_{b,i,t} + \Phi_t w^{\text{within}}_i + T_i(h^{\text{hist}}_{b,i}).
\]

### Why

Baseline-2 is the default nuisance model because it is flexible enough to absorb smooth PSTH-like structure but still small and regularized enough that it is less likely to steal the entire explanation from the latent dynamics.

The loss includes regularization on this branch:

- an \(L_2\) penalty on the within-trial basis weights,
- a second-difference smoothness penalty on the resulting time curve,
- a mild regularizer on the trial branch.

### Strengths

- best default for repeated-trial neural data,
- separates task-relevant latent structure from nuisance rate structure,
- still learned jointly with the full model.

### Risks

If the basis rank is too large, it can absorb task-relevant signal that should live in the latent dynamics.

---

## 8.4 Baseline-3: prefit FINDR-style nuisance baseline

### Definition
Baseline-3 computes a nuisance baseline **before** the model rollout, using only the training split.

It decomposes nuisance rate into an across-trial term and a within-trial term:
\[
r^{\text{nuis}}_{b,i,t,n} = r^{\text{across}}_{b,i,n} + r^{\text{within}}_{i,t,n}.
\]

#### Across-trial component
A radial basis over trial times is fit by ridge regression to mean firing rates across trials:
\[
r^{\text{across}}_{b,i,n} \approx \Psi(b) \, a_{i,n}.
\]

#### Within-trial component
A raised-cosine basis over time is then fit, again by ridge regression, to residual rate after subtracting the across-trial component:
\[
r^{\text{within}}_{i,t,n} \approx \Phi(t) \, c_{i,n}.
\]

The result is clipped to a valid positive rate and transformed into log-rate via the inverse softplus:
\[
d^{\text{nuis}}_{b,i,t,n} = \operatorname{softplus}^{-1}\!\big(r^{\text{nuis}}_{b,i,t,n}\big).
\]

In baseline-3, the forward model uses
\[
\eta^{\text{full}}_{b,i,t} = C_i z_{b,i,t} + d^{\text{nuis}}_{b,i,t},
\]
where \(d^{\text{nuis}}\) is fixed rather than jointly learned.

### Why

This is closest to a FINDR-style nuisance treatment for highly repeated, well-aligned datasets with strong stereotyped PSTHs.

### Strengths

- strongest nuisance separation,
- often produces very clean vector fields,
- useful when stereotyped baseline structure is obvious.

### Risks

Because the nuisance term is fixed in advance, it can remove signal that the full joint model might otherwise allocate to communication or latent dynamics. That is why B3 is included as a focused interpretability variant rather than as the default family-wide choice.

---

## 9. Why the implemented variants are A0/A1/A2, B1/B2/B3, C1/C2

This family layout is deliberate.

### A0 / A1 / A2

A is the default family, so it is paired with the three most useful nuisance choices:

- **A0**: pure identification benchmark,
- **A1**: minimal realistic control,
- **A2**: flagship real-data model.

### B1 / B2 / B3

B is the interpretability family, so it is the right place to test the stronger prefit nuisance option:

- **B1**: simplest interpretable model,
- **B2**: interpretable joint-nuisance model,
- **B3**: FINDR-style prefitted nuisance companion.

### C1 / C2

C already adds coupling flexibility. Combining that with a fixed prefit nuisance branch would make attribution harder, so the implemented C variants stop at:

- **C1**: richer coupling + constant bias,
- **C2**: richer coupling + joint nuisance.

This keeps the model family broad enough to test the key hypotheses without exploding the number of overlapping explanations.

---

## 10. Training objective

The implemented loss is
\[
\mathcal{L}
=
\mathcal{L}_{\text{NLL}}
+ \beta_{\text{traj}} \mathcal{L}_{\text{traj}}
+ \beta_{\text{msg}} \mathcal{L}_{\text{msg}}
+ \beta_{\text{hid}} \mathcal{L}_{\text{hid}}
+ \beta_{z0} \mathcal{L}_{z0}
+ \beta_{\text{base}} \mathcal{L}_{\text{base}}.
\]

### 10.1 Poisson observation term
\[
\mathcal{L}_{\text{NLL}}
= - \sum_{b,i,t,n} \log p(y_{b,i,t,n} \mid r^{\text{full}}_{b,i,t,n}).
\]
In code this is the Poisson negative log-likelihood with rate \(dt \cdot r^{\text{full}}\).

### 10.2 Trajectory consistency term
The model penalizes disagreement between the posterior drift and the interpretable prior drift:
\[
\mathcal{L}_{\text{traj}}
= \frac{1}{2} \sum_{b,i,t,\ell}
\frac{\alpha_i \big(\mu^{\text{post}}_{b,i,t,\ell} - \mu^{\text{prior}}_{b,i,t,\ell}\big)^2}{(\sigma^{\text{proc}}_{i,\ell})^2}.
\]

This is the core FINDR-like idea that prevents the interpretable prior system from drifting too far from the inference-guided posterior rollout used during training.

### 10.3 Hidden-input KL
For variants with hidden inputs,
\[
\mathcal{L}_{\text{hid}}
= \operatorname{KL}\Big(q(u_{b,i,t}) \;\|\; \mathcal{N}(0, \sigma_u^2 I)\Big).
\]

### 10.4 Initial-condition KL
If initial-condition inference is enabled,
\[
\mathcal{L}_{z0}
= \operatorname{KL}\Big(q(z_{b,i,0}) \;\|\; \mathcal{N}(0, \sigma_{z0}^2 I)\Big).
\]

### 10.5 Message regularization

- For **A/C** (stochastic messages):
\[
\mathcal{L}_{\text{msg}}
= \operatorname{KL}\Big(q(m_{b,j\to i,t}) \;\|\; \mathcal{N}(0, \sigma_m^2 I)\Big).
\]

- For **B** (deterministic messages):
\[
\mathcal{L}_{\text{msg}} = \frac{1}{2\sigma_m^2} \|m_{b,j\to i,t}\|^2.
\]

### 10.6 Baseline regularization
For baseline-2 only, the model adds an explicit regularizer on the nuisance branch:

- \(L_2\) on within-basis weights,
- smoothness via second differences of the within-trial curve,
- mild weight penalty on the trial branch.

### 10.7 KL schedule
The code uses a ramp schedule so that different penalties come online gradually. This is important because if the model is strongly penalized before it has learned a sensible reconstruction, it can allocate variance in pathological ways.

---

## 11. Synthetic dataset generator

The synthetic generator is designed to stress exactly the ambiguities the model family is supposed to resolve.

## 11.1 Regions and state dimensions

The default synthetic benchmark has **3 regions**, each with a 2D latent state:
\[
L_1 = L_2 = L_3 = 2.
\]
Each region has its own intrinsic nonlinear flow, readout, known-input channels, hidden-input mixing, and communication edges.

## 11.2 Known inputs

The generator treats all known inputs as explicit time-series channels:

- region 0: `click_left`, `click_right`, `stim_on`,
- region 1: `rule_A`, `rule_B`,
- region 2: `go_cue`.

Mathematically, these are all just components of \(k_{i,t}\), but semantically they cover:

- sparse event-like pulses,
- sustained step inputs,
- context variables.

This is important because the implementation should not assume that all exogenous inputs are pulse-like or all are sustained; they are simply channels with different temporal structure.

## 11.3 Hidden inputs

Hidden inputs are generated as low-dimensional AR(1) processes with common and region-specific components:
\[
h_t = \rho h_{t-1} + \sigma_h \epsilon_t.
\]
They are then mixed differently into each region. This creates structured unobserved drive that cannot be explained solely by observed inter-region communication.

## 11.4 Ground-truth local dynamics

Each region uses a FINDR-style gated intrinsic flow:
\[
f_i(z) = \sigma(G_i z + b^g_i) \odot \big(-z + \tanh(F_i z + b^f_i)\big).
\]
The full ground-truth drift is
\[
\dot z_{i,t} = f_i(z_{i,t}) + B_i k_{i,t} + H_i h_{i,t} + \Delta^{\text{msg}}_{i,t}
\]
in the additive scenario.

In the modulatory scenario,
\[
\dot z_{i,t} = f_i(z_{i,t}) + B_i k_{i,t} + H_i h_{i,t} + \Delta^{\text{msg}}_{i,t} + \tanh(\Gamma_{i,t}) \odot z_{i,t}.
\]

## 11.5 Ground-truth communication

Ground-truth messages are emitted from the source region's task-relevant rates with one-bin lag:
\[
m^{\star}_{b,j\to i,t} = W^{\star}_{j\to i} r^{\text{task},\star}_{b,j,t-1}.
\]
Those messages are then mapped additively, and optionally gain-wise, into the target latent dynamics.

This means the synthetic benchmark is aligned with the design goal of the model family rather than being a generic LDS simulator.

## 11.6 Ground-truth nuisance baseline

The full log-rate used to generate spikes is
\[
\eta^{\text{full},\star}_{b,i,t} = C_i z^{\star}_{b,i,t} + d^{\text{within},\star}_{i,t} + d^{\text{across},\star}_{b,i} + c_i.
\]
Thus the synthetic data include:

- smooth within-trial nuisance structure,
- across-trial drift,
- neuron-specific offsets.

That makes the 0/1/2/3 baseline ablations meaningful rather than artificial.

---

## 12. Implementation notes and why the code is now fast

The current code is fast because the implementation now matches the mathematical structure of the model.

### 12.1 What was wrong before

The original prototype effectively executed:

- a Python loop over time,
- inside that, Python loops over regions and edges,
- many tiny per-region and per-edge module calls on small tensors.

That structure is mathematically correct but GPU-hostile.

### 12.2 What Neuro-Voltron does now

The recurrence is still sequential in **time**—because the state update is genuinely recurrent—but the **step body** is batched:

- latent state is stored as a padded tensor of shape `[B, R, L_max]`,
- all edge messages are computed in one call through `BatchedEdges`,
- all regional flow evaluations are computed in one call through `BatchedFlowMLP`,
- the hidden-input controller is a batched GRU step over all regions,
- readout and input projections are batched with `einsum`.

So the actual computation is now:

1. one batched edge pass,
2. one batched hidden-input/controller pass,
3. one batched known-input pass,
4. one batched prior/posterior flow pass,
5. one batched state update,
6. one batched readout.

That is exactly the right implementation shape for this model class.

### 12.3 Other concrete fixes

The repository also now includes:

- fused sequence/history GRUs via `nn.GRU`,
- cached time bases in both forward and loss code paths,
- one-time device movement for static tensors,
- preallocated rollout output tensors.

These changes do not alter the mathematics; they remove avoidable overhead.

---

## 13. Recommended usage

If you want one default model, use:

> **A2**

because it combines:

- stochastic rate-constrained messages,
- additive coupling,
- hidden inferred inputs,
- explicit local vector fields,
- a jointly learned low-capacity nuisance baseline.

If you want the cleanest geometric companion analysis, use:

> **B2**

If you suspect genuinely modulatory communication, test:

> **C2**

If you want the clearest repeated-trial, strongly stereotyped nuisance comparison, add:

> **B3**

---

## 14. Final model summary in one sentence

**Neuro-Voltron is a multi-region latent dynamical system in which local computation lives in explicit low-dimensional regional vector fields, communication is inferred from task-relevant reconstructed source activity, nuisance baseline is modeled separately in observation space, and trial variability is decomposed across initial condition, hidden inputs, and process noise.**

---

## 15. Synthetic dataset: `three_region_ring`

### Motivation

The original `three_region_additive` dataset had several problems that made model
validation unreliable:

- **Region 2 had a saddle-point flow** (one positive eigenvalue), causing latents to
  diverge to -47.
- **Region 0 had only 5.8% task-driven lograte variance**; the baseline dominated,
  making latent inference from spikes nearly impossible.
- **Unbounded message cascade**: `softplus(z @ readout)` as the message source
  created a positive feedback loop between regions.
- **Firing rates were extremely low** (~0.7 Hz mean, 99.4% zero bins).

### Design: `three_region_ring`

The replacement dataset uses three qualitatively distinct, bounded flow fields
connected in a uni-directional ring (0 -> 1 -> 2 -> 0):

| Region | Flow type | Description |
|--------|-----------|-------------|
| R0 | **Limit cycle** (Hopf normal form) | Stable orbit at radius 1. Persistent oscillatory activity provides rich, time-varying message source. |
| R1 | **Line attractor** | Strong attraction to z2=0, memory along z1 axis. Integrates accumulated input over time. |
| R2 | **Double well** | Two stable states at z1 = +/-1 with a saddle at the origin. Models binary commitment dynamics. |

### Key improvements over the original dataset

| Parameter | Old dataset | Ring dataset |
|-----------|------------|--------------|
| Latent ranges | R1: [-30, 0.8], R2: [-47, 0.3] | R0: [-1.2, 1.2], R1: [-7, 7], R2: [-1.2, 1.2] |
| Task variance fraction | R0: 5.8% | R0: 35%, R1: 81%, R2: 36% |
| Mean firing rate | 0.7 Hz | 1.6 Hz |
| Message source | `softplus(z @ readout)` (unbounded) | `tanh(z)` (bounded in [-1, 1]) |
| Active edges | 4 of 6 (messy) | 3 ring edges (clean) |
| Readout scale | 1.1 | 2.5 |
| Baseline within-trial weight std | 0.25 | 0.10 |
| Const offset mean | 0.0 | 1.5 (higher floor rate) |
| Neurons per region | 100 | 50 |

### Connectivity

```
R0 (limit cycle) --[0.4]--> R1 (line attractor) --[0.4]--> R2 (double well) --[0.4]--> R0
```

All three edges have equal coupling strength. Inactive edges (0->2, 1->0, 2->1)
have exactly zero coupling matrices, providing a clear test of whether the model
correctly identifies which edges carry information.

### Generation

```bash
python scripts/generate_synthetic_dataset.py \
    --scenario three_region_ring \
    --out data/three_region_ring.npz \
    --n-trials 1000 \
    --neurons-per-region 50
```
