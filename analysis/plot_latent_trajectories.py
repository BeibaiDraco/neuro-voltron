#!/usr/bin/env python3
"""Plot inferred latent trajectories, optionally overlaid with ground truth.

Usage:
    python analysis/plot_latent_trajectories.py --workdir runs/A2_additive --data data/three_region_additive.npz
    python analysis/plot_latent_trajectories.py --workdir runs/A2_additive --data data/three_region_additive.npz --n-trials 20
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


COLORS = ['#f59e0b', '#06b6d4', '#a78bfa']
GT_COLORS = ['#b45309', '#0891b2', '#7c3aed']
NAMES = ['Region 0', 'Region 1', 'Region 2']


def main():
    parser = argparse.ArgumentParser(description='Plot latent trajectories.')
    parser.add_argument('--workdir', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--n-trials', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    workdir = Path(args.workdir)
    rng = np.random.default_rng(args.seed)

    posterior = np.load(workdir / 'posterior_means_full.npz')
    inf_z = posterior['z']  # [B, R, T, Lmax]
    B, R, T, Lmax = inf_z.shape

    with np.load(args.data, allow_pickle=True) as f:
        gt_keys = [f'gt_latent_region{r}' for r in range(R)]
        has_gt = all(k in f.files for k in gt_keys)
        if has_gt:
            gt_z = [np.array(f[k]) for k in gt_keys]

    trial_idx = rng.choice(B, size=min(args.n_trials, B), replace=False)

    fig, axes = plt.subplots(R, 2, figsize=(14, 4 * R), facecolor='#0a0e17')
    fig.suptitle('Latent Trajectories per Region',
                 fontsize=16, color='white', fontweight='bold', y=0.98)
    if R == 1:
        axes = axes[np.newaxis, :]

    for r in range(R):
        # Left: z1 and z2 vs time
        ax = axes[r, 0]
        ax.set_facecolor('#0f1629')
        for i, trial in enumerate(trial_idx):
            alpha = 0.7
            ax.plot(inf_z[trial, r, :, 0], color=COLORS[r], alpha=alpha, linewidth=0.8,
                    label='Inferred z₁' if i == 0 else None)
            ax.plot(inf_z[trial, r, :, 1], color=COLORS[r], alpha=alpha, linewidth=0.8,
                    linestyle='--', label='Inferred z₂' if i == 0 else None)
            if has_gt:
                ax.plot(gt_z[r][trial, :, 0], color=GT_COLORS[r], alpha=0.5, linewidth=0.8,
                        label='GT z₁' if i == 0 else None)
                ax.plot(gt_z[r][trial, :, 1], color=GT_COLORS[r], alpha=0.5, linewidth=0.8,
                        linestyle='--', label='GT z₂' if i == 0 else None)
        ax.set_xlabel('Time bin', color='#64748b')
        ax.set_ylabel('Latent value', color='#64748b')
        ax.set_title(f'{NAMES[r]}  —  Time traces', color=COLORS[r], fontsize=11, fontweight='bold')
        ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='white', fontsize=8, loc='upper right')
        ax.tick_params(colors='#475569', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#1e293b')
        ax.grid(True, alpha=0.06, color='white')

        # Right: z1 vs z2 phase portrait
        ax = axes[r, 1]
        ax.set_facecolor('#0f1629')
        for trial in trial_idx:
            ax.plot(inf_z[trial, r, :, 0], inf_z[trial, r, :, 1],
                    color=COLORS[r], alpha=0.5, linewidth=0.8)
            ax.plot(inf_z[trial, r, 0, 0], inf_z[trial, r, 0, 1], 'o',
                    color=COLORS[r], markersize=4, alpha=0.7)
            if has_gt:
                ax.plot(gt_z[r][trial, :, 0], gt_z[r][trial, :, 1],
                        color=GT_COLORS[r], alpha=0.35, linewidth=0.8)
        ax.set_xlabel('z₁', color='#64748b')
        ax.set_ylabel('z₂', color='#64748b')
        ax.set_title(f'{NAMES[r]}  —  Phase portrait', color=COLORS[r], fontsize=11, fontweight='bold')
        ax.tick_params(colors='#475569', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#1e293b')
        ax.grid(True, alpha=0.06, color='white')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = workdir / 'latent_trajectories.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor='#0a0e17', edgecolor='none')
    print(f'Saved {out_path}')
    plt.close(fig)


if __name__ == '__main__':
    main()
