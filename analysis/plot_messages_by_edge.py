#!/usr/bin/env python3
"""Compare inferred vs ground-truth per-edge message drives over time.

For each directed edge (src -> tgt), plots the additive contribution to the
target region's latent state: GT message projected through GT matrix vs
inferred message sample projected through learned add_w.

Usage:
    python analysis/plot_messages_by_edge.py \
        --workdir runs/A2_improved \
        --data data/three_region_additive.npz
"""
from __future__ import annotations

import argparse
import json
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

from neuro_voltron.config import ExperimentConfig, build_variant_config
from neuro_voltron.model import NeuroVoltron


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workdir', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--n-trials', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    workdir = Path(args.workdir)
    rng = np.random.default_rng(args.seed)

    # Load config
    resolved_path = workdir / 'config_resolved.json'
    if resolved_path.exists():
        cfg = ExperimentConfig.from_dict(json.loads(resolved_path.read_text()))
    else:
        cfg = build_variant_config('A2')

    # Load data
    data = np.load(args.data, allow_pickle=True)
    R = int(data['latent_dims_per_region'].shape[0])
    L = int(data['latent_dims_per_region'].max())

    edge_pairs = [(s, t) for s in range(R) for t in range(R) if s != t]

    # Load GT messages and projection matrices
    gt_drives = {}
    for s, t in edge_pairs:
        key = f'{s}to{t}'
        gt_msg = np.array(data[f'gt_messages_{key}'])            # (N, T, M_gt)
        gt_mat = np.array(data[f'gt_message_matrix_{key}'])      # (L, M_gt)
        gt_drive = np.einsum('ntm,lm->ntl', gt_msg, gt_mat)     # (N, T, L)
        gt_drives[key] = gt_drive

    # Load inferred message samples and model weights for add_w projection
    posterior = np.load(workdir / 'posterior_means_full.npz')
    inf_msg_sample = posterior['msg_sample']  # (N, R, R, T, M_inf)

    state_dict = torch.load(workdir / 'model.pt', map_location='cpu')
    add_w = state_dict['batched_edges.add_w'].numpy()  # (E, L_max, M_inf)

    # Reconstruct edge ordering (same as model __init__)
    edge_mask = cfg.model.edge_mask
    if edge_mask is None:
        edge_mask = [[0.0 if i == j else 1.0 for j in range(R)] for i in range(R)]
    edge_list = []
    for s in range(R):
        for t in range(R):
            if s != t and edge_mask[s][t] > 0:
                edge_list.append((s, t))

    # Compute inferred per-edge additive drives
    inf_drives = {}
    for e_idx, (s, t) in enumerate(edge_list):
        key = f'{s}to{t}'
        msg = inf_msg_sample[:, s, t, :, :]      # (N, T, M_inf)
        w = add_w[e_idx, :L, :]                   # (L, M_inf)
        drive = np.einsum('ntm,lm->ntl', msg, w)  # (N, T, L)
        inf_drives[key] = drive

    # Pick trials
    N = gt_drives[list(gt_drives.keys())[0]].shape[0]
    trial_idx = rng.choice(N, size=min(args.n_trials, N), replace=False)
    trial_idx.sort()

    T = gt_drives[list(gt_drives.keys())[0]].shape[1]
    time_axis = np.arange(T)

    n_edges = len(edge_pairs)
    n_trials = len(trial_idx)

    # --- Plot 1: Per-edge, trial-averaged message drive ---
    fig1, axes1 = plt.subplots(n_edges, L, figsize=(5 * L, 3.5 * n_edges),
                                squeeze=False)
    fig1.suptitle('Trial-Averaged Message Drive (additive contribution to latent)',
                  fontsize=14, fontweight='bold', color='white')
    fig1.patch.set_facecolor('#1a1a2e')

    for row, (s, t) in enumerate(edge_pairs):
        key = f'{s}to{t}'
        gt = gt_drives[key]      # (N, T, L)
        inf = inf_drives[key]    # (N, T, L)

        gt_mean = gt.mean(axis=0)    # (T, L)
        gt_std = gt.std(axis=0)
        inf_mean = inf.mean(axis=0)
        inf_std = inf.std(axis=0)

        for d in range(L):
            ax = axes1[row, d]
            ax.set_facecolor('#16213e')

            ax.fill_between(time_axis,
                            gt_mean[:, d] - gt_std[:, d],
                            gt_mean[:, d] + gt_std[:, d],
                            alpha=0.2, color='#e94560')
            ax.plot(time_axis, gt_mean[:, d], color='#e94560',
                    linewidth=1.5, label='GT')

            ax.fill_between(time_axis,
                            inf_mean[:, d] - inf_std[:, d],
                            inf_mean[:, d] + inf_std[:, d],
                            alpha=0.2, color='#0f3460')
            ax.plot(time_axis, inf_mean[:, d], color='#00d2ff',
                    linewidth=1.5, label='Inferred')

            ax.set_title(f'R{s}→R{t}  latent dim {d+1}',
                         color='white', fontsize=11)
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('#444')
            if row == 0 and d == 0:
                ax.legend(fontsize=8, facecolor='#16213e',
                          edgecolor='#444', labelcolor='white')
            if row == n_edges - 1:
                ax.set_xlabel('Time bin', color='white')
            if d == 0:
                ax.set_ylabel('Additive drive', color='white')

    fig1.tight_layout(rect=[0, 0, 1, 0.96])
    out1 = workdir / 'message_drives_averaged.png'
    fig1.savefig(out1, dpi=150, facecolor=fig1.get_facecolor())
    plt.close(fig1)
    print(f'Saved {out1}')

    # --- Plot 2: Example single trials (overlay GT vs inferred) ---
    fig2, axes2 = plt.subplots(n_edges, n_trials, figsize=(4 * n_trials, 3 * n_edges),
                                squeeze=False)
    fig2.suptitle(f'Per-Edge Message Drive — Example Trials (L1 norm over latent dims)',
                  fontsize=13, fontweight='bold', color='white')
    fig2.patch.set_facecolor('#1a1a2e')

    for row, (s, t) in enumerate(edge_pairs):
        key = f'{s}to{t}'
        gt = gt_drives[key]
        inf = inf_drives[key]

        for col, tr in enumerate(trial_idx):
            ax = axes2[row, col]
            ax.set_facecolor('#16213e')

            gt_norm = np.linalg.norm(gt[tr], axis=-1)    # (T,)
            inf_norm = np.linalg.norm(inf[tr], axis=-1)

            ax.plot(time_axis, gt_norm, color='#e94560', linewidth=1.2,
                    label='GT' if col == 0 else None)
            ax.plot(time_axis, inf_norm, color='#00d2ff', linewidth=1.2,
                    label='Inf' if col == 0 else None)

            if row == 0:
                ax.set_title(f'Trial {tr}', color='white', fontsize=10)
            if col == 0:
                ax.set_ylabel(f'R{s}→R{t}', color='white', fontsize=10)

            ax.tick_params(colors='white', labelsize=7)
            for spine in ax.spines.values():
                spine.set_color('#444')

            if row == 0 and col == 0:
                ax.legend(fontsize=7, facecolor='#16213e',
                          edgecolor='#444', labelcolor='white')

    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    out2 = workdir / 'message_drives_trials.png'
    fig2.savefig(out2, dpi=150, facecolor=fig2.get_facecolor())
    plt.close(fig2)
    print(f'Saved {out2}')

    # --- Plot 3: Per-edge correlation heatmap + R² ---
    from sklearn.metrics import r2_score

    fig3, axes3 = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
    fig3.suptitle('Per-Edge Message Drive: Inferred vs GT (all trials flattened)',
                  fontsize=14, fontweight='bold', color='white')
    fig3.patch.set_facecolor('#1a1a2e')

    for idx, (s, t) in enumerate(edge_pairs):
        key = f'{s}to{t}'
        gt_flat = gt_drives[key].reshape(-1, L)
        inf_flat = inf_drives[key].reshape(-1, L)

        gt_norm = np.linalg.norm(gt_flat, axis=-1)
        inf_norm = np.linalg.norm(inf_flat, axis=-1)

        r2 = r2_score(gt_norm, inf_norm)
        corr = np.corrcoef(gt_norm, inf_norm)[0, 1]

        row, col = divmod(idx, 3)
        ax = axes3[row, col]
        ax.set_facecolor('#16213e')

        subsample = rng.choice(len(gt_norm), size=min(5000, len(gt_norm)), replace=False)
        ax.scatter(gt_norm[subsample], inf_norm[subsample],
                   s=1, alpha=0.15, color='#00d2ff')

        lims = [min(gt_norm.min(), inf_norm.min()),
                max(gt_norm.max(), inf_norm.max())]
        ax.plot(lims, lims, '--', color='#e94560', linewidth=1, alpha=0.7)

        ax.set_title(f'R{s}→R{t}   R²={r2:.3f}  ρ={corr:.3f}',
                     color='white', fontsize=11)
        ax.set_xlabel('GT |drive|', color='white', fontsize=9)
        ax.set_ylabel('Inferred |drive|', color='white', fontsize=9)
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#444')

    fig3.tight_layout(rect=[0, 0, 1, 0.95])
    out3 = workdir / 'message_drives_scatter.png'
    fig3.savefig(out3, dpi=150, facecolor=fig3.get_facecolor())
    plt.close(fig3)
    print(f'Saved {out3}')


if __name__ == '__main__':
    main()
