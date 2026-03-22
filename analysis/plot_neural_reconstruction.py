#!/usr/bin/env python3
"""Visualize neural activity reconstruction: predicted rates vs observed spikes.

Usage:
    python analysis/plot_neural_reconstruction.py \
        --workdir runs/A2_improved \
        --data data/three_region_additive.npz
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
import matplotlib.gridspec as gridspec
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workdir', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--n-trials', type=int, default=3)
    parser.add_argument('--n-neurons', type=int, default=6,
                        help='neurons per region to show')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    workdir = Path(args.workdir)
    rng = np.random.default_rng(args.seed)

    data = np.load(args.data, allow_pickle=True)
    posterior = np.load(workdir / 'posterior_means_full.npz')

    R = int(data['latent_dims_per_region'].shape[0])
    dt = float(data['dt'])
    rates = posterior['rates'] * dt     # convert Hz to expected counts/bin
    task_rates = posterior['task_rates'] * dt

    spikes_list = [np.array(data[f'spikes_region{r}']) for r in range(R)]
    num_neurons = [s.shape[2] for s in spikes_list]

    N, T_max = spikes_list[0].shape[0], spikes_list[0].shape[1]
    trial_idx = rng.choice(N, size=min(args.n_trials, N), replace=False)
    trial_idx.sort()
    n_trials = len(trial_idx)

    time_axis = np.arange(T_max)
    region_colors = ['#e2a04f', '#00d2ff', '#c77dff']
    region_names = [f'Region {r}' for r in range(R)]

    # ---- Plot 1: Raster + rate overlay for example trials ----
    n_show = min(args.n_neurons, min(num_neurons))
    fig1, axes1 = plt.subplots(R, n_trials, figsize=(5 * n_trials, 3.5 * R),
                                squeeze=False)
    fig1.suptitle('Neural Reconstruction: Spikes (dots) vs Predicted Rate (lines)',
                  fontsize=14, fontweight='bold', color='white')
    fig1.patch.set_facecolor('#1a1a2e')

    for row in range(R):
        nn = num_neurons[row]
        neuron_idx = rng.choice(nn, size=min(n_show, nn), replace=False)
        neuron_idx.sort()

        for col, tr in enumerate(trial_idx):
            ax = axes1[row, col]
            ax.set_facecolor('#16213e')

            for i, nid in enumerate(neuron_idx):
                offset = i * 1.0
                spike_times = np.where(spikes_list[row][tr, :, nid] > 0)[0]
                spike_heights = spikes_list[row][tr, spike_times, nid]
                ax.scatter(spike_times, np.full_like(spike_times, offset, dtype=float),
                           s=spike_heights * 8, alpha=0.5,
                           color=region_colors[row], marker='|', linewidths=0.5)

                rate_trace = rates[tr, row, :, nid]
                rate_scaled = rate_trace / max(rate_trace.max(), 1e-6) * 0.8 + offset - 0.1
                ax.plot(time_axis, rate_scaled, color='white', alpha=0.7,
                        linewidth=0.8)

            ax.set_ylim(-0.5, n_show)
            ax.set_yticks([i for i in range(n_show)])
            ax.set_yticklabels([f'n{nid}' for nid in neuron_idx], fontsize=7)

            if row == 0:
                ax.set_title(f'Trial {tr}', color='white', fontsize=11)
            if col == 0:
                ax.set_ylabel(region_names[row], color=region_colors[row],
                              fontsize=11, fontweight='bold')
            if row == R - 1:
                ax.set_xlabel('Time bin', color='white')

            ax.tick_params(colors='white', labelsize=7)
            for spine in ax.spines.values():
                spine.set_color('#444')

    fig1.tight_layout(rect=[0, 0, 1, 0.96])
    out1 = workdir / 'neural_reconstruction_raster.png'
    fig1.savefig(out1, dpi=150, facecolor=fig1.get_facecolor())
    plt.close(fig1)
    print(f'Saved {out1}')

    # ---- Plot 2: PSTH-style — trial-averaged rate vs trial-averaged spikes ----
    n_psth = 8
    fig2, axes2 = plt.subplots(R, n_psth, figsize=(2.5 * n_psth, 3 * R),
                                squeeze=False)
    fig2.suptitle('Trial-Averaged Firing Rate: Observed (smoothed spikes) vs Predicted',
                  fontsize=13, fontweight='bold', color='white')
    fig2.patch.set_facecolor('#1a1a2e')

    from scipy.ndimage import uniform_filter1d
    smooth_win = 5

    for row in range(R):
        nn = num_neurons[row]
        active_rates = np.array(data[f'spikes_region{row}']).mean(axis=0).mean(axis=0)
        top_neurons = np.argsort(active_rates)[-n_psth:][::-1]

        for col, nid in enumerate(top_neurons):
            ax = axes2[row, col]
            ax.set_facecolor('#16213e')

            obs_mean = spikes_list[row][:, :, nid].mean(axis=0)
            obs_smooth = uniform_filter1d(obs_mean.astype(float), smooth_win)

            pred_mean = rates[:, row, :, nid].mean(axis=0)

            ax.fill_between(time_axis, obs_smooth, alpha=0.3,
                            color=region_colors[row])
            ax.plot(time_axis, obs_smooth, color=region_colors[row],
                    linewidth=1.0, label='Observed' if col == 0 else None)
            ax.plot(time_axis, pred_mean, color='white',
                    linewidth=1.0, label='Predicted' if col == 0 else None)

            if row == 0:
                ax.set_title(f'Neuron {nid}', color='white', fontsize=9)
            if col == 0:
                ax.set_ylabel(region_names[row], color=region_colors[row],
                              fontsize=10, fontweight='bold')

            ax.tick_params(colors='white', labelsize=6)
            for spine in ax.spines.values():
                spine.set_color('#444')
            if row == 0 and col == 0:
                ax.legend(fontsize=6, facecolor='#16213e',
                          edgecolor='#444', labelcolor='white')

    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    out2 = workdir / 'neural_reconstruction_psth.png'
    fig2.savefig(out2, dpi=150, facecolor=fig2.get_facecolor())
    plt.close(fig2)
    print(f'Saved {out2}')

    # ---- Plot 3: Per-region R² and variance explained summary ----
    fig3, axes3 = plt.subplots(1, R + 1, figsize=(4 * (R + 1), 4),
                                squeeze=False,
                                gridspec_kw={'width_ratios': [1]*R + [1.3]})
    fig3.suptitle('Reconstruction Quality per Region',
                  fontsize=14, fontweight='bold', color='white')
    fig3.patch.set_facecolor('#1a1a2e')

    region_r2s = []
    for r in range(R):
        nn = num_neurons[r]
        obs = spikes_list[r]          # (N, T, nn)
        pred = rates[:, r, :, :nn]    # (N, T, nn)

        # R² on trial-averaged traces (proper metric for Poisson observations)
        obs_avg = obs.mean(axis=0)    # (T, nn)
        pred_avg = pred.mean(axis=0)

        neuron_r2 = np.zeros(nn)
        for n in range(nn):
            ss_res = np.sum((obs_avg[:, n] - pred_avg[:, n]) ** 2)
            ss_tot = np.sum((obs_avg[:, n] - obs_avg[:, n].mean()) ** 2)
            neuron_r2[n] = 1.0 - ss_res / max(ss_tot, 1e-12)

        region_r2s.append(neuron_r2)

        ax = axes3[0, r]
        ax.set_facecolor('#16213e')
        sorted_r2 = np.sort(neuron_r2)[::-1]
        ax.bar(range(nn), sorted_r2, color=region_colors[r], alpha=0.8, width=1.0)
        ax.axhline(np.median(neuron_r2), color='white', linestyle='--',
                    linewidth=1, alpha=0.7)
        ax.set_title(f'{region_names[r]}\nmedian R²={np.median(neuron_r2):.3f}',
                     color=region_colors[r], fontsize=11)
        ax.set_xlabel('Neuron (sorted)', color='white', fontsize=9)
        ax.set_ylabel('R²', color='white', fontsize=9)
        ax.set_ylim(-0.05, max(1.0, sorted_r2.max() * 1.1))
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#444')

    ax_summary = axes3[0, R]
    ax_summary.set_facecolor('#16213e')
    all_r2 = [np.median(r) for r in region_r2s]
    bars = ax_summary.bar(range(R), all_r2,
                          color=region_colors[:R], alpha=0.9, width=0.6)
    ax_summary.set_xticks(range(R))
    ax_summary.set_xticklabels(region_names, fontsize=9)
    ax_summary.set_ylabel('Median Neuron R²', color='white', fontsize=10)
    ax_summary.set_title('Summary', color='white', fontsize=12)
    for i, v in enumerate(all_r2):
        ax_summary.text(i, v + 0.01, f'{v:.3f}', ha='center',
                        color='white', fontsize=10, fontweight='bold')
    ax_summary.tick_params(colors='white')
    for spine in ax_summary.spines.values():
        spine.set_color('#444')

    fig3.tight_layout(rect=[0, 0, 1, 0.94])
    out3 = workdir / 'neural_reconstruction_r2.png'
    fig3.savefig(out3, dpi=150, facecolor=fig3.get_facecolor())
    plt.close(fig3)
    print(f'Saved {out3}')


if __name__ == '__main__':
    main()
