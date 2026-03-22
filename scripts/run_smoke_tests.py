#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tempfile
from pathlib import Path

from neuro_voltron.config import ALL_VARIANTS, build_variant_config
from neuro_voltron.data import apply_baseline_mode, load_dataset, split_dataset
from neuro_voltron.synthetic import SyntheticConfig, generate_synthetic_dataset, save_synthetic_dataset
from neuro_voltron.train import fit_model


def main() -> None:
    tmpdir = Path(tempfile.mkdtemp(prefix='neuro_voltron_smoke_'))
    print(f'[smoke] tempdir={tmpdir}')
    datasets = {
        'three_region_additive': tmpdir / 'additive_small.npz',
        'three_region_modulatory': tmpdir / 'modulatory_small.npz',
    }
    for scenario, path in datasets.items():
        cfg = SyntheticConfig(
            scenario=scenario,
            n_trials=24,
            history_steps=8,
            main_steps=40,
            min_length=30,
            max_length=40,
            neuron_counts=(6, 7, 8),
            seed=0 if 'additive' in scenario else 1,
        )
        payload, meta = generate_synthetic_dataset(cfg)
        save_synthetic_dataset(path, payload, meta)
        print(f'[smoke] wrote dataset {path.name}')

    for variant in ALL_VARIANTS:
        scenario = 'three_region_modulatory' if variant.startswith('C') else 'three_region_additive'
        data_path = datasets[scenario]
        cfg = build_variant_config(variant, small=True)
        cfg.optim.epochs = 1
        cfg.optim.batch_size = 8
        cfg.optim.log_every = 1
        cfg.data.synthetic_history_bins = 8
        dataset = load_dataset(data_path, cfg.model, cfg.data)
        splits = split_dataset(dataset, cfg.data)
        splits = apply_baseline_mode(dataset, splits, cfg.model)
        outdir = tmpdir / variant
        fit_model(cfg, splits, workdir=outdir)
        metrics_path = outdir / 'metrics.json'
        assert metrics_path.exists(), f'metrics missing for {variant}'
        print(f'[smoke] PASS {variant}')

    print(f'[smoke] leaving artifacts in {tmpdir}')


if __name__ == '__main__':
    main()
