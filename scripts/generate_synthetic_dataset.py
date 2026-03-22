#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse

from neuro_voltron.synthetic import SyntheticConfig, generate_synthetic_dataset, generate_ring_dataset, save_synthetic_dataset


def main() -> None:
    p = argparse.ArgumentParser(description='Generate a 3-region synthetic dataset.')
    p.add_argument('--out', required=True)
    p.add_argument('--scenario', default='three_region_additive',
                   choices=['three_region_additive', 'three_region_modulatory', 'three_region_ring'])
    p.add_argument('--n-trials', type=int, default=512)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--history-steps', type=int, default=20)
    p.add_argument('--main-steps', type=int, default=160)
    p.add_argument('--neurons-per-region', type=int, default=0,
                   help='If > 0, overrides the default neuron counts to this number for all regions.')
    p.add_argument('--no-temporal-baseline', action='store_true',
                   help='Remove temporal/trial baseline (const offset only).')
    p.add_argument('--readout-scale', type=float, default=2.5)
    p.add_argument('--offset-mean', type=float, default=1.5)
    p.add_argument('--msg-coupling-scale', type=float, default=1.2)
    args = p.parse_args()

    cfg = SyntheticConfig(
        scenario=args.scenario,
        n_trials=args.n_trials,
        seed=args.seed,
        history_steps=args.history_steps,
        main_steps=args.main_steps,
        max_length=args.main_steps,
        min_length=max(20, args.main_steps - 30),
    )
    if args.neurons_per_region > 0:
        cfg.neuron_counts = (args.neurons_per_region, args.neurons_per_region, args.neurons_per_region)
    if args.no_temporal_baseline:
        cfg.use_temporal_baseline = False
    cfg.readout_scale = args.readout_scale
    cfg.offset_mean = args.offset_mean
    cfg.msg_coupling_scale = args.msg_coupling_scale

    if cfg.scenario == 'three_region_ring':
        payload, meta = generate_ring_dataset(cfg)
    else:
        payload, meta = generate_synthetic_dataset(cfg)
    save_synthetic_dataset(args.out, payload, meta)
    print(f'Saved dataset to {args.out}')


if __name__ == '__main__':
    main()
