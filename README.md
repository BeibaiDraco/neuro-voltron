# Neuro-Voltron

A multi-region latent dynamical-system framework for neural population analysis. Neuro-Voltron learns interpretable, region-wise vector fields while inferring directed inter-region communication from task-relevant reconstructed activity.

The framework builds on ideas from [FINDR](https://github.com/Brody-Lab/findr) (local latent vector fields) and [MR-LFADS](https://proceedings.mlr.press/v267/liu25bh.html) (multi-region sequential autoencoders), combining them into a unified model that explicitly decomposes neural variance into local dynamics, inter-region messages, known inputs, hidden inputs, and observation-side nuisance baselines.

## Key Features

- **Interpretable local dynamics**: each brain region gets its own low-dimensional nonlinear vector field, suitable for phase-portrait and fixed-point analysis
- **Communication from task-relevant activity**: messages between regions are derived from reconstructed task rates (not raw latents), preventing nuisance structure from leaking into communication estimates
- **Flexible model family**: 8 variants (A0/A1/A2, B1/B2/B3, C1/C2) spanning different trade-offs between identification, interpretability, and coupling expressiveness
- **GPU-optimized**: batched recurrence over regions and edges in a single tensorized step body

## Model Variants

| Family | Communication | Coupling | Use case |
|--------|--------------|----------|----------|
| **A** (A0, A1, A2) | Stochastic messages | Additive | Default for balanced identification and interpretation |
| **B** (B1, B2, B3) | Deterministic messages | Additive | Clean vector fields and geometric analysis |
| **C** (C1, C2) | Stochastic messages | Additive + gain | Systems with modulatory inter-region interactions |

The numeric suffix controls the nuisance baseline: **0** = none, **1** = constant bias, **2** = jointly learned smooth baseline, **3** = pre-fit FINDR-style baseline.

**Recommended starting point: A2.**

For full mathematical details, see [docs/Techreport.md](docs/Techreport.md).

## Repository Layout

```
neuro_voltron/          # Core library
    model.py            #   Model definition (all variant families)
    train.py            #   Training loop and loss computation
    data.py             #   Dataset loading
    synthetic.py        #   Synthetic data generation
    basis.py            #   Raised-cosine basis functions
    config.py           #   Configuration and variant registry
    evaluate.py         #   Evaluation utilities
configs/
    variants/           # JSON configs for each variant (A0..C2)
scripts/
    train_variant.py    # Train a model variant
    generate_synthetic_dataset.py
    run_smoke_tests.py
analysis/               # Plotting and post-hoc analysis
    plot_flow_fields.py
    plot_training_curves.py
    plot_effectome.py
    plot_latent_trajectories.py
docs/
    Techreport.md       # Full mathematical specification
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For GPU training, install a CUDA-enabled PyTorch build appropriate for your hardware.

## Quick Start

### Generate synthetic data

```bash
python scripts/generate_synthetic_dataset.py \
    --scenario three_region_ring \
    --out data/three_region_ring.npz \
    --n-trials 1000 \
    --neurons-per-region 50
```

Available scenarios: `three_region_ring`, `three_region_additive`, `three_region_modulatory`.

### Train a model

```bash
python scripts/train_variant.py \
    --variant A2 \
    --data data/three_region_ring.npz \
    --workdir runs/A2_ring
```

### Run smoke tests

```bash
python scripts/run_smoke_tests.py
```

Runs all 8 variants for one epoch each on small synthetic datasets.

## Dataset Format

The loader expects an `.npz` file containing region-wise arrays:

- `spikes_region{i}`: spike count tensors per region
- `known_inputs_region{i}`: exogenous input channels per region
- `history_region{i}`: pre-trial spike history per region
- `lengths`, `times`, `dt`: timing metadata
- (Optional) `gt_latent_region{i}`, `gt_messages_{j}to{i}`: ground-truth arrays for evaluation

## Training Artifacts

Each run saves to `workdir/`:

| File | Contents |
|------|----------|
| `config_resolved.json` | Full resolved configuration |
| `model.pt` | Model checkpoint |
| `history.json` | Per-epoch loss components |
| `metrics.json` | Final evaluation metrics |
| `posterior_means_full.npz` | Inferred latent trajectories |
| `effectome_full.npy` | Inferred communication matrix |

## License

All rights reserved. This code is provided for viewing and reference purposes only. No permission is granted to use, copy, modify, or distribute this software without explicit written consent from the author.
