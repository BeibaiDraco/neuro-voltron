#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json

from neuro_voltron.config import ALL_VARIANTS, ExperimentConfig, build_variant_config
from neuro_voltron.data import apply_baseline_mode, load_dataset, split_dataset
from neuro_voltron.train import fit_model


def main() -> None:
    p = argparse.ArgumentParser(description='Train a hybrid MR-FINDR variant.')
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--variant', choices=ALL_VARIANTS)
    group.add_argument('--config-json')
    p.add_argument('--data', required=True)
    p.add_argument('--workdir', required=True)
    p.add_argument('--epochs', type=int)
    p.add_argument('--batch-size', type=int)
    p.add_argument('--seed', type=int)
    p.add_argument('--small', action='store_true', help='Use reduced hidden sizes and fast smoke-test schedules.')
    args = p.parse_args()

    if args.config_json is not None:
        cfg = ExperimentConfig.from_dict(json.loads(Path(args.config_json).read_text()))
    else:
        cfg = build_variant_config(args.variant, small=args.small)
    if args.epochs is not None:
        cfg.optim.epochs = args.epochs
    if args.batch_size is not None:
        cfg.optim.batch_size = args.batch_size
    if args.seed is not None:
        cfg.optim.seed = args.seed
        cfg.data.shuffle_seed = args.seed

    dataset = load_dataset(args.data, cfg.model, cfg.data)
    splits = split_dataset(dataset, cfg.data)
    splits = apply_baseline_mode(dataset, splits, cfg.model)
    fit_model(cfg, splits, workdir=args.workdir)
    print(f'Training complete. Artifacts saved to {args.workdir}')


if __name__ == '__main__':
    main()
