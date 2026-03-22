# Trajectory KL Collapse: Diagnosis and Fix Plan

## The Problem

In neuro-voltron's sequential VAE architecture, the **trajectory KL** (measuring the squared posterior correction) collapses to the free-bits floor within ~110 epochs and stays there for the rest of training. This means the posterior encoder contributes nothing to the dynamics — the prior flow absorbs everything.

### Observed symptoms (v3–v5 runs on ring topology dataset)

| Metric | Value | Expected |
|--------|-------|----------|
| traj_kl | 0.1000 (free_bits floor) | > 0.1 (non-trivial correction) |
| hid_kl | 12.000 (free_bits floor) | above floor |
| msg_kl | ~88 | > 0 (healthy) |
| Effectome | Dense (~15 on all edges) | Sparse (3 ring edges only) |
| Message R2 (true edges) | 1.0, 1.0, 1.0 | 1.0 |
| Message R2 (non-edges) | 0.02, 0.65, 0.01 | 0.0 |
| Neural reconstruction R2 | ~0 (negative in some regions) | > 0.3 |
| Flow fields | Trivial fixed-point attractors | Limit cycle, line attractor, double well |

## Root Cause Analysis

There are **two interacting causes**: a model design flaw and a dataset SNR issue.

### Cause 1: Prior absorbs posterior (model design)

The architecture uses an additive parameterization:

```
prior_drift  = prior_flow(z) + known_inputs + messages
post_drift   = prior_drift + correction_scale * post_flow(z, encoder_output)
z[t]         = z[t-1] + α * (post_drift + hidden_inputs) + noise
```

The trajectory KL = `0.5 * α * |correction|² / σ²` sends gradients to:
- **θ_prior**: "match the posterior" → prior moves toward posterior
- **θ_post**: "make correction smaller" → posterior moves toward prior

Both gradients push correction → 0. The only counter-force is the NLL gradient on θ_post, which is weak at low firing rates.

**Why the prior CAN absorb the posterior:** The prior_flow sees z[t], which was updated using the posterior dynamics at all previous timesteps. The prior learns to predict the combined effect from z alone — it "reads the posterior's homework" through the shared z trajectory.

**Why messages and hidden inputs DON'T collapse:** They occupy structurally separate channels. The prior_flow cannot substitute for hidden_drive (it's added separately outside the flow) or message variables (they have their own KL on the latent message variables, not the drift).

### Cause 2: Insufficient observation signal (dataset)

All three datasets (v3, v4, v5) have firing rates of **1–2 Hz** (1–2% of bins have spikes). At these rates:

| Dataset | Observed rate | % bins with spikes | NLL improvement from dynamics |
|---------|--------------|-------------------|------------------------------|
| three_region_additive | 0.6–0.8 Hz | 0.6–0.8% | ~0.002 nats |
| three_region_ring | 1.5–1.6 Hz | 1.5–1.6% | ~0.005 nats |
| three_region_ring_v5 | 1.2–1.5 Hz | 1.2–1.5% | ~0.005 nats |

The Poisson observation model at these rates provides ~0.004 nats of information per bin about the underlying rate. This is insufficient for the NLL gradient to keep the posterior correction alive against the KL penalty.

**Key computation:** The NLL gap between a baseline-only model and the full ground-truth model is only 0.00023–0.00157 nats. Even perfect dynamics recovery barely improves NLL.

### Why the prior_flow learns trivial dynamics

The prior_flow uses a gated relaxation architecture: `output = gate * (-z + target(z))`. At initialization, this is approximately `gate * (-z)` — a linear attractor toward z=0. With insufficient NLL gradient, the prior never escapes this trivial fixed point.

### Causal chain

```
Low firing rates → weak NLL gradient → posterior correction dies →
prior stays at trivial fixed point → z collapses → rates become constant →
flow fields are trivial → messages carry no temporal structure
```

## Expert Review

An external ML expert confirmed the diagnosis and provided additional context:

> "Your diagnosis is basically right. In latent SDE/ODE-style VAEs, the path KL is a control-cost between posterior and prior drifts, and standard training jointly optimizes both drifts under that objective. Recent DSSM work makes the broader point explicitly: maximizing the ELBO does **not** by itself ensure that the model learns the true underlying dynamics."

Key recommendations:
1. **Separate the two goals**: (a) good approximate inference and (b) correct autonomous dynamics identification
2. The additive parameterization `post = prior + correction` mixes these goals in one vector field — that is the core problem
3. A strong smoother (bidirectional encoder) will always make reconstruction good, but the prior is rewarded for matching smoothed trajectories that include future information unavailable at generation time

## Fix Plan: Four Phases

### Phase 1: Stop-grad + rollout loss (IMPLEMENTED)

**Status: Code complete, ready for training**

Two changes in `compute_loss()` in `neuro_voltron/model.py`:

#### 1a. Detach prior from traj_kl (`detach_prior_kl: true`)

```python
# Before (θ_prior gets gradient from KL, chases posterior):
traj = 0.5 * α * (post_drift - prior_drift)² / σ²

# After (θ_prior gets ZERO KL gradient, learns from NLL only):
correction = post_drift - prior_drift.detach()
traj = 0.5 * α * correction² / σ²
```

**Why .detach() on prior_drift, not on the full (post - prior)?** Because `post_drift = prior_drift + correction`. If we wrote `(post - prior.detach())²`, θ_prior would still get gradient through the `prior` inside `post`. Using `correction = post - prior.detach()` ensures θ_prior gets truly zero gradient from this loss term.

θ_prior still learns from the NLL, because: `NLL → rates → z → (prior contribution) → θ_prior`.

#### 1b. Prior-only rollout loss (`rollout_loss_weight: 1.0`, `rollout_steps: 8`)

From detached posterior z states, roll the prior forward k steps **without** posterior correction, then measure MSE against the posterior z trajectory:

```python
L_rollout = Σ_{j=1}^{k} |z_prior[t+j] - stopgrad(z_post[t+j])|²
```

This forces `prior_flow` to be good at **free-running prediction**, not just living on smoother-produced states. The prior must learn dynamics that actually predict the future trajectory.

**Implementation:** 4 anchor points per trial, 8 steps each, using detached posterior messages for known drives.

**Config:** `configs/A2_v6.json`
**Dataset:** `data/three_region_ring_v6.npz` (20 Hz firing rates, 16.4% of bins have spikes)

### Phase 2: Alternating training (NOT YET IMPLEMENTED)

If Phase 1 is insufficient, add mild EM-style training:

- **Inference step:** Update posterior/messages/hidden with θ_prior frozen
- **Model step:** Update θ_prior and decoder with posterior samples detached

This removes the "everyone moves together to kill the correction" dynamic. Implementation: alternate every N epochs (e.g., N=5).

### Phase 3: Predict–update architecture (NOT YET IMPLEMENTED)

Move the posterior correction out of the generative drift into an explicit update operator:

```
Forecast:  z⁻[t] = z[t-1] + α*(prior_flow(z[t-1]) + known + messages + hidden) + noise
Update:    z[t]   = z⁻[t] + u_φ(z⁻[t], encoder_output)
```

The prior defines the generative dynamics. The update is explicitly inference-only — the prior is never asked to imitate it. This is the most principled fix (closest to filtering/smoothing literature) but requires a significant architecture change.

### Phase 4: Identifiability constraints (NOT YET IMPLEMENTED)

Even with proper training, `prior_flow` vs hidden-input effects is generally **not identifiable** without structural constraints. Options:

- Low-capacity or smooth hidden-input channel
- Sparse or temporally local posterior correction
- Orthogonality penalty between posterior correction and prior drift
- Correction/update only at observation times
- Ablation experiments where hidden inputs or messages are known absent

## Dataset Fix: Higher Firing Rates

The v6 dataset (`data/three_region_ring_v6.npz`) was generated with `offset_mean=20.0` to achieve ~20 Hz firing rates:

| Parameter | v5 | v6 |
|-----------|----|----|
| offset_mean | 1.0 | 20.0 |
| readout_scale | 4.0 | 4.0 |
| Observed rate | 1.2 Hz | 18.1 Hz |
| Bins with spikes | 1.2% | 16.4% |
| Modulation depth | — | ~40% (16–24 Hz) |

Generation command:
```bash
python scripts/generate_synthetic_dataset.py \
  --out data/three_region_ring_v6.npz \
  --scenario three_region_ring \
  --n-trials 1000 --neurons-per-region 100 \
  --no-temporal-baseline --readout-scale 4.0 \
  --offset-mean 20.0 --msg-coupling-scale 0.8
```

## How Variants A, B, C Differ on This Issue

| Feature | A (full) | B (simple) | C (gain) |
|---------|----------|------------|----------|
| Messages | Stochastic | Deterministic | Stochastic |
| Coupling | Additive | Additive | Additive + gain |
| Prior flow | [64, 64] | [48] (1 layer) | [64, 64] |
| Post flow | [64, 64] | [48] (1 layer) | [64, 64] |

**Variant B** has simpler prior/posterior flows ([48] single layer). This partially mitigates the absorption problem (the prior has less capacity to track the posterior), but also limits the quality of learned flow fields.

**Variant C** adds a multiplicative gain term: `prior += tanh(msg_gain) * z`. This couples messages into the prior more tightly. The gain channel may be less susceptible to absorption (since it modulates z multiplicatively, which the prior_flow's additive structure can't easily replicate).

**All variants** suffer from the fundamental issue: the prior and posterior share the same z trajectory, and the traj_kl gradient pushes them to converge.

## Config Fields Reference

```json
{
  "loss": {
    "detach_prior_kl": true,        // Phase 1a: θ_prior gets no KL gradient
    "rollout_loss_weight": 1.0,     // Phase 1b: weight for prior rollout MSE
    "rollout_steps": 8              // Phase 1b: steps to roll forward
  }
}
```

These are backward-compatible: defaults are `false`, `0.0`, and `8`, matching pre-Phase-1 behavior.

## Validation Plan

After training v6 with Phase 1 fixes:

1. **traj_kl should be ABOVE free_bits floor** (> 0.1) — posterior correction is alive
2. **rollout_mse should decrease** during training — prior is learning to predict
3. **Flow fields should match GT topology** — limit cycle, line attractor, double well
4. **Effectome should be SPARSE** — only 3 ring edges active
5. **Message R2 should be high on true edges, near zero on non-edges**
6. **Neural reconstruction R2 should be >> 0** — dynamics actually improve spike prediction

If Phase 1 succeeds on v6 (20 Hz), progressively test on lower firing rates to find the minimum viable SNR.

## Files Modified

- `neuro_voltron/config.py`: Added `detach_prior_kl`, `rollout_loss_weight`, `rollout_steps` to `LossConfig`
- `neuro_voltron/model.py`: Modified `compute_loss()` — detach logic for traj_kl, rollout MSE computation
- `neuro_voltron/train.py`: Added `rollout_mse` to history tracking and epoch logging
- `configs/A2_v6.json`: Config with Phase 1 enabled + v6 dataset
- `data/three_region_ring_v6.npz`: 20 Hz firing rate dataset
