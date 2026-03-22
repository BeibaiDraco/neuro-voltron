#!/usr/bin/env python3
"""Visualize learned latent flow fields and compare against ground truth.

Usage:
    python analysis/plot_flow_fields.py --workdir runs/A2_additive --data data/three_region_additive.npz
    python analysis/plot_flow_fields.py --workdir runs/A2_additive --data data/three_region_additive.npz --no-gt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

import json
from neuro_voltron.config import build_variant_config, ExperimentConfig
from neuro_voltron.model import NeuroVoltron


# ---------------------------------------------------------------------------
# Ground truth flow functions
# ---------------------------------------------------------------------------

# Old gated-attractor flows (three_region_additive / modulatory)
_GATED_FLOW_PARAMS = {
    'F': [
        np.array([[1.1, 0.4], [-0.1, 0.7]], dtype=np.float32),
        np.array([[0.5, -1.2], [1.1, 0.4]], dtype=np.float32),
        np.array([[0.8, 0.3], [0.2, 1.0]], dtype=np.float32),
    ],
    'G': [
        np.array([[1.0, -0.2], [0.3, 0.8]], dtype=np.float32),
        np.array([[0.6, 0.4], [-0.5, 1.1]], dtype=np.float32),
        np.array([[0.7, 0.1], [0.2, 0.9]], dtype=np.float32),
    ],
    'bF': [np.array([0.05, -0.02]), np.array([0.02, 0.04]), np.array([-0.03, 0.01])],
    'bG': [np.array([0.1, -0.1]), np.array([0.0, 0.05]), np.array([-0.05, 0.0])],
}


def _gated_attractor_flow(z: np.ndarray, r: int) -> np.ndarray:
    F, G = _GATED_FLOW_PARAMS['F'][r], _GATED_FLOW_PARAMS['G'][r]
    bF, bG = _GATED_FLOW_PARAMS['bF'][r], _GATED_FLOW_PARAMS['bG'][r]
    gate = 1.0 / (1.0 + np.exp(-(z @ G.T + bG)))
    target = np.tanh(z @ F.T + bF)
    return gate * (-z + target)


# Ring flows (three_region_ring)
def _limit_cycle_flow(z: np.ndarray, radius: float = 1.0, speed: float = 1.5) -> np.ndarray:
    x, y = z[..., 0], z[..., 1]
    r2 = x ** 2 + y ** 2
    dx = speed * (-y + x * (radius ** 2 - r2))
    dy = speed * (x + y * (radius ** 2 - r2))
    return np.stack([dx, dy], axis=-1)


def _line_attractor_flow(z: np.ndarray, decay_perp: float = 2.0) -> np.ndarray:
    x, y = z[..., 0], z[..., 1]
    dx = np.zeros_like(x)
    dy = -decay_perp * y
    return np.stack([dx, dy], axis=-1)


def _double_well_flow(z: np.ndarray, depth: float = 2.0, sep: float = 1.0, y_decay: float = 2.0) -> np.ndarray:
    x, y = z[..., 0], z[..., 1]
    dx = -depth * x * (x ** 2 - sep ** 2) / sep ** 4
    dy = -y_decay * y
    return np.stack([dx, dy], axis=-1)


_RING_FLOWS = [_limit_cycle_flow, _line_attractor_flow, _double_well_flow]


def _detect_scenario(workdir, data_path) -> str:
    """Detect scenario from config_resolved.json or data metadata."""
    cfg_path = Path(workdir) / 'config_resolved.json'
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        name = cfg.get('name', '')
        if 'ring' in name.lower():
            return 'three_region_ring'
    meta_path = Path(data_path).with_suffix('.json')
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get('scenario') == 'three_region_ring':
            return 'three_region_ring'
        if 'ring_edges' in meta:
            return 'three_region_ring'
    return 'three_region_additive'


def gt_intrinsic_flow(z: np.ndarray, r: int, scenario: str = 'three_region_additive') -> np.ndarray:
    if scenario == 'three_region_ring':
        return _RING_FLOWS[r](z)
    return _gated_attractor_flow(z, r)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

COLORS = ['#f59e0b', '#06b6d4', '#a78bfa']
TRAJ_RGB = [(0.96, 0.62, 0.04), (0.02, 0.71, 0.83), (0.65, 0.55, 0.98)]
REGION_NAMES = ['Region 0', 'Region 1', 'Region 2']


def _grid_and_flow(z_flat: np.ndarray, flow_fn, grid_n: int = 22, pct: float = 1.0):
    lo = np.percentile(z_flat, pct, axis=0)
    hi = np.percentile(z_flat, 100 - pct, axis=0)
    margin = (hi - lo) * 0.25
    lo -= margin
    hi += margin
    g0 = np.linspace(lo[0], hi[0], grid_n)
    g1 = np.linspace(lo[1], hi[1], grid_n)
    G0, G1 = np.meshgrid(g0, g1)
    pts = np.stack([G0.ravel(), G1.ravel()], axis=-1)
    flows = flow_fn(pts)
    fu = flows[:, 0].reshape(grid_n, grid_n)
    fv = flows[:, 1].reshape(grid_n, grid_n)
    return g0, g1, fu, fv


def _plot_panel(ax, g0, g1, fu, fv, trajs, r, title):
    ax.set_facecolor('#0f1629')
    mag = np.sqrt(fu ** 2 + fv ** 2)
    max_mag = max(mag.max(), 1e-6)
    lw = 0.6 + 2.0 * mag / max_mag

    for traj in trajs:
        ax.plot(traj[:, 0], traj[:, 1],
                color=(*TRAJ_RGB[r], 0.1), linewidth=0.5, zorder=1)
        ax.plot(traj[0, 0], traj[0, 1], 'o',
                color=(*TRAJ_RGB[r], 0.35), markersize=1.5, zorder=3)

    ax.streamplot(g0, g1, fu, fv,
                  color=COLORS[r], linewidth=lw, density=1.3,
                  arrowsize=0.8, arrowstyle='->', zorder=2)

    ax.set_title(title, color=COLORS[r], fontsize=12, fontweight='bold', pad=8)
    ax.set_xlabel('z₁', color='#64748b', fontsize=9)
    ax.set_ylabel('z₂', color='#64748b', fontsize=9)
    ax.tick_params(colors='#475569', labelsize=7)
    for spine in ax.spines.values():
        spine.set_color('#1e293b')
    ax.grid(True, alpha=0.06, color='white')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Plot latent flow fields.')
    parser.add_argument('--workdir', type=str, required=True, help='Training run directory')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset .npz')
    parser.add_argument('--variant', type=str, default='A2', help='Model variant name')
    parser.add_argument('--n-trials', type=int, default=50, help='Number of trajectories to overlay')
    parser.add_argument('--grid-n', type=int, default=22, help='Flow field grid resolution')
    parser.add_argument('--no-gt', action='store_true', help='Skip ground truth (for real data)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    workdir = Path(args.workdir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = np.random.default_rng(args.seed)

    # Load model config (prefer saved config_resolved.json, fall back to variant)
    resolved_cfg_path = workdir / 'config_resolved.json'
    if resolved_cfg_path.exists():
        cfg = ExperimentConfig.from_dict(json.loads(resolved_cfg_path.read_text()))
    else:
        cfg = build_variant_config(args.variant)
    with np.load(args.data, allow_pickle=True) as f:
        num_neurons = [int(f[k].shape[2]) for k in sorted(f.files) if k.startswith('spikes_region')]
        input_dims_keys = sorted([k for k in f.files if k.startswith('known_inputs_region')])
        input_dims = [int(f[k].shape[2]) for k in input_dims_keys] if input_dims_keys else [0] * len(num_neurons)

    R = len(num_neurons)
    latent_sizes = list(cfg.model.latent_sizes[:R])
    model = NeuroVoltron(num_neurons, input_dims, latent_sizes, cfg.model).to(device)
    model.load_state_dict(torch.load(workdir / 'model.pt', map_location=device))
    model.eval()

    posterior = np.load(workdir / 'posterior_means_full.npz')
    inf_z = posterior['z']

    scenario = _detect_scenario(workdir, args.data)

    has_gt = not args.no_gt
    if has_gt:
        with np.load(args.data, allow_pickle=True) as f:
            gt_keys = [f'gt_latent_region{r}' for r in range(R)]
            has_gt = all(k in f.files for k in gt_keys)
            if has_gt:
                gt_z = [np.array(f[k]) for k in gt_keys]

    trial_idx = rng.choice(inf_z.shape[0], size=min(args.n_trials, inf_z.shape[0]), replace=False)

    def inferred_flow_fn(pts, r):
        batch_z = torch.zeros(len(pts), R, model.max_latent, device=device)
        batch_z[:, r, :latent_sizes[r]] = torch.tensor(pts[:, :latent_sizes[r]], dtype=torch.float32, device=device)
        with torch.no_grad():
            out = model.prior_flow(batch_z)
        return out[:, r, :latent_sizes[r]].cpu().numpy()

    if has_gt:
        # Two-row figure: GT on top, inferred on bottom
        fig, axes = plt.subplots(2, R, figsize=(6 * R, 12), facecolor='#0a0e17')
        fig.suptitle('Ground Truth vs Inferred Latent Flow Fields',
                     fontsize=18, color='white', fontweight='bold', y=0.97)

        for r in range(R):
            Lr = latent_sizes[r]
            gt_trajs = gt_z[r][trial_idx, :, :Lr]
            inf_trajs = inf_z[trial_idx, r, :, :Lr]

            g0, g1, fu, fv = _grid_and_flow(
                gt_z[r].reshape(-1, Lr),
                lambda pts, _r=r, _s=scenario: gt_intrinsic_flow(pts, _r, _s),
                args.grid_n)
            _plot_panel(axes[0, r], g0, g1, fu, fv, gt_trajs,
                        r, f'{REGION_NAMES[r]}  —  GROUND TRUTH')

            g0i, g1i, fui, fvi = _grid_and_flow(
                inf_z[:, r, :, :Lr].reshape(-1, Lr),
                lambda pts, _r=r: inferred_flow_fn(pts, _r),
                args.grid_n)
            _plot_panel(axes[1, r], g0i, g1i, fui, fvi, inf_trajs,
                        r, f'{REGION_NAMES[r]}  —  INFERRED')

        fig.text(0.5, 0.49,
                 'Top: ground truth intrinsic dynamics  |  Bottom: inferred prior flow  |  '
                 'Faint lines: latent trajectories',
                 fontsize=9, color='#64748b', ha='center')
        plt.tight_layout(rect=[0, 0.01, 1, 0.95])
        plt.subplots_adjust(hspace=0.3)
    else:
        # Single row: inferred only
        fig, axes = plt.subplots(1, R, figsize=(6 * R, 6), facecolor='#0a0e17')
        fig.suptitle('Inferred Latent Flow Fields  —  Prior Dynamics f(z)',
                     fontsize=16, color='white', fontweight='bold', y=0.98)
        if R == 1:
            axes = [axes]

        for r in range(R):
            Lr = latent_sizes[r]
            inf_trajs = inf_z[trial_idx, r, :, :Lr]
            g0, g1, fu, fv = _grid_and_flow(
                inf_z[:, r, :, :Lr].reshape(-1, Lr),
                lambda pts, _r=r: inferred_flow_fn(pts, _r),
                args.grid_n)
            _plot_panel(axes[r], g0, g1, fu, fv, inf_trajs,
                        r, REGION_NAMES[r])

        plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    out_path = workdir / 'flow_fields.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor='#0a0e17', edgecolor='none')
    print(f'Saved {out_path}')
    plt.close(fig)


if __name__ == '__main__':
    main()
