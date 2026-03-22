#!/usr/bin/env python3
"""Visualize inferred vs ground truth effectome (connectivity matrix).

Usage:
    python analysis/plot_effectome.py --workdir runs/A2_additive
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Plot effectome comparison.')
    parser.add_argument('--workdir', type=str, required=True)
    args = parser.parse_args()

    workdir = Path(args.workdir)
    with open(workdir / 'metrics.json') as f:
        metrics = json.load(f)

    inferred = np.array(metrics['effectome'])
    gt = np.array(metrics['ground_truth_effectome'])
    cos_sim = metrics['effectome_cosine_similarity']
    msg_r2 = metrics.get('message_r2', {})
    R = inferred.shape[0]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor='#0a0e17')
    fig.suptitle(f'Effectome Recovery  (cosine similarity = {cos_sim:.3f})',
                 fontsize=16, color='white', fontweight='bold', y=0.98)

    region_labels = [f'R{i}' for i in range(R)]
    cmap = 'RdBu_r'
    vmax = max(abs(gt).max(), abs(inferred).max(), 1e-6)

    for idx, (mat, title) in enumerate([(gt, 'Ground Truth'), (inferred, 'Inferred')]):
        ax = axes[idx]
        ax.set_facecolor('#0f1629')
        im = ax.imshow(mat, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='equal')
        ax.set_xticks(range(R))
        ax.set_yticks(range(R))
        ax.set_xticklabels(region_labels, color='#94a3b8', fontsize=10)
        ax.set_yticklabels(region_labels, color='#94a3b8', fontsize=10)
        ax.set_xlabel('Target', color='#64748b')
        ax.set_ylabel('Source', color='#64748b')
        ax.set_title(title, color='white', fontsize=13, pad=10)
        for i in range(R):
            for j in range(R):
                ax.text(j, i, f'{mat[i, j]:.2f}', ha='center', va='center',
                        color='white' if abs(mat[i, j]) > vmax * 0.5 else '#94a3b8',
                        fontsize=9, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel 3: Message R2 bar chart
    ax = axes[2]
    ax.set_facecolor('#0f1629')
    if msg_r2:
        edges = sorted(msg_r2.keys())
        r2_vals = [msg_r2[e] for e in edges]
        bar_colors = ['#f59e0b' if v > 0.5 else '#06b6d4' if v > 0.1 else '#475569' for v in r2_vals]
        bars = ax.barh(range(len(edges)), r2_vals, color=bar_colors, edgecolor='#1e293b')
        ax.set_yticks(range(len(edges)))
        ax.set_yticklabels([e.replace('to', ' → ') for e in edges], color='#94a3b8', fontsize=10)
        ax.set_xlabel('R²', color='#64748b')
        ax.set_xlim(0, 1.05)
        for i, v in enumerate(r2_vals):
            ax.text(v + 0.02, i, f'{v:.2f}', va='center', color='#e2e8f0', fontsize=9)
    ax.set_title('Message R²', color='white', fontsize=13, pad=10)
    ax.tick_params(colors='#475569', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#1e293b')
    ax.grid(True, axis='x', alpha=0.08, color='white')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = workdir / 'effectome.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor='#0a0e17', edgecolor='none')
    print(f'Saved {out_path}')
    plt.close(fig)


if __name__ == '__main__':
    main()
