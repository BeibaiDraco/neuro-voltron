#!/usr/bin/env python3
"""Plot training loss curves and KL component breakdown from a training run.

Usage:
    python analysis/plot_training_curves.py --workdir runs/A2_additive
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    'loss': '#f8fafc',
    'nll': '#f59e0b',
    'traj_kl': '#06b6d4',
    'hidden_kl': '#a78bfa',
    'message_kl': '#f472b6',
    'z0_kl': '#34d399',
    'baseline_reg': '#94a3b8',
}


def main():
    parser = argparse.ArgumentParser(description='Plot training curves.')
    parser.add_argument('--workdir', type=str, required=True)
    args = parser.parse_args()

    workdir = Path(args.workdir)
    with open(workdir / 'history.json') as f:
        history = json.load(f)

    epochs = np.arange(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='#0a0e17')
    fig.suptitle(f'Training Curves  —  {workdir.name}',
                 fontsize=16, color='white', fontweight='bold', y=0.97)

    for ax in axes.flat:
        ax.set_facecolor('#0f1629')
        ax.tick_params(colors='#475569', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#1e293b')
        ax.grid(True, alpha=0.08, color='white')

    # Panel 1: Total loss
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], color='#f59e0b', linewidth=1.5, label='Train')
    ax.plot(epochs, history['val_loss'], color='#06b6d4', linewidth=1.5, label='Val')
    ax.set_xlabel('Epoch', color='#64748b')
    ax.set_ylabel('Total Loss', color='#64748b')
    ax.set_title('Total Loss', color='white', fontsize=12)
    ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='white', fontsize=9)

    # Panel 2: NLL only
    ax = axes[0, 1]
    ax.plot(epochs, history['train_nll'], color='#f59e0b', linewidth=1.5, label='Train NLL')
    ax.plot(epochs, history['val_nll'], color='#06b6d4', linewidth=1.5, label='Val NLL')
    ax.set_xlabel('Epoch', color='#64748b')
    ax.set_ylabel('NLL (Poisson)', color='#64748b')
    ax.set_title('Reconstruction Loss', color='white', fontsize=12)
    ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='white', fontsize=9)

    # Panel 3: KL components (train)
    ax = axes[1, 0]
    for key, color in [('traj_kl', '#06b6d4'), ('hidden_kl', '#a78bfa'),
                        ('message_kl', '#f472b6'), ('z0_kl', '#34d399')]:
        label = key.replace('_', ' ').replace('kl', 'KL')
        ax.plot(epochs, history[f'train_{key}'], color=color, linewidth=1.2, label=label)
    ax.set_xlabel('Epoch', color='#64748b')
    ax.set_ylabel('Raw KL (before beta weighting)', color='#64748b')
    ax.set_title('KL Components (Train)', color='white', fontsize=12)
    ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='white', fontsize=9)

    # Panel 4: Beta schedule
    ax = axes[1, 1]
    for key, color in [('beta_traj', '#06b6d4'), ('beta_hidden', '#a78bfa'),
                        ('beta_message', '#f472b6'), ('beta_z0', '#34d399')]:
        ax.plot(epochs, history[key], color=color, linewidth=1.5, label=key.replace('_', ' '))
    ax.set_xlabel('Epoch', color='#64748b')
    ax.set_ylabel('Beta weight', color='#64748b')
    ax.set_title('KL Annealing Schedule', color='white', fontsize=12)
    ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='white', fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = workdir / 'training_curves.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor='#0a0e17', edgecolor='none')
    print(f'Saved {out_path}')
    plt.close(fig)


if __name__ == '__main__':
    main()
