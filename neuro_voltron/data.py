from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .basis import radial_basis_over_trials, raised_cosine_basis, softplus_inv
from .config import DataConfig, ModelConfig

Array = Any


@dataclass
class HybridDataset:
    spikes: np.ndarray  # [B, R, T, Nmax]
    known_inputs: np.ndarray  # [B, R, T, Kmax]
    history: np.ndarray  # [B, R, H, Nmax]
    lengths: np.ndarray  # [B]
    times: np.ndarray  # [B]
    dt: float
    region_names: list[str]
    num_neurons: np.ndarray  # [R]
    input_dims: np.ndarray  # [R]
    latent_sizes: np.ndarray  # [R]
    neuron_mask: np.ndarray  # [R, Nmax]
    input_mask: np.ndarray  # [R, Kmax]
    latent_mask: np.ndarray  # [R, Lmax]
    history_mask: np.ndarray  # [R, Nmax]
    history_lengths: np.ndarray  # [B]
    baseline_prefit: np.ndarray  # [B, R, T, Nmax]
    metadata: dict[str, Any]

    def select(self, indices: np.ndarray) -> 'HybridDataset':
        idx = np.asarray(indices)
        return replace(
            self,
            spikes=self.spikes[idx],
            known_inputs=self.known_inputs[idx],
            history=self.history[idx],
            lengths=self.lengths[idx],
            times=self.times[idx],
            history_lengths=self.history_lengths[idx],
            baseline_prefit=self.baseline_prefit[idx],
        )

    def to_jax(self):
        import jax.numpy as jnp
        return {
            'spikes': jnp.asarray(self.spikes, dtype=jnp.float32),
            'known_inputs': jnp.asarray(self.known_inputs, dtype=jnp.float32),
            'history': jnp.asarray(self.history, dtype=jnp.float32),
            'lengths': jnp.asarray(self.lengths, dtype=jnp.int32),
            'times': jnp.asarray(self.times, dtype=jnp.float32),
            'history_lengths': jnp.asarray(self.history_lengths, dtype=jnp.int32),
            'neuron_mask': jnp.asarray(self.neuron_mask, dtype=jnp.float32),
            'input_mask': jnp.asarray(self.input_mask, dtype=jnp.float32),
            'latent_mask': jnp.asarray(self.latent_mask, dtype=jnp.float32),
            'history_mask': jnp.asarray(self.history_mask, dtype=jnp.float32),
            'baseline_prefit': jnp.asarray(self.baseline_prefit, dtype=jnp.float32),
            'dt': jnp.asarray(self.dt, dtype=jnp.float32),
        }

    def to_torch(self, device: str | None = None):
        import torch

        def maybe(x, dtype):
            t = torch.as_tensor(x, dtype=dtype)
            return t.to(device) if device is not None else t

        return {
            'spikes': maybe(self.spikes, torch.float32),
            'known_inputs': maybe(self.known_inputs, torch.float32),
            'history': maybe(self.history, torch.float32),
            'lengths': maybe(self.lengths, torch.long),
            'times': maybe(self.times, torch.float32),
            'history_lengths': maybe(self.history_lengths, torch.long),
            'neuron_mask': maybe(self.neuron_mask, torch.float32),
            'input_mask': maybe(self.input_mask, torch.float32),
            'latent_mask': maybe(self.latent_mask, torch.float32),
            'history_mask': maybe(self.history_mask, torch.float32),
            'baseline_prefit': maybe(self.baseline_prefit, torch.float32),
            'dt': torch.tensor(float(self.dt), dtype=torch.float32, device=device),
        }


@dataclass
class DatasetSplits:
    train: HybridDataset
    val: HybridDataset
    test: HybridDataset
    full: HybridDataset
    split_indices: dict[str, np.ndarray]


def _find_region_spike_keys(files: list[str]) -> list[str]:
    keys = []
    idx = 0
    while f'spikes_region{idx}' in files:
        keys.append(f'spikes_region{idx}')
        idx += 1
    if keys:
        return keys
    keys = []
    idx = 0
    while f'region{idx}' in files:
        keys.append(f'region{idx}')
        idx += 1
    if keys:
        return keys
    raise ValueError('Could not find region spike arrays: expected spikes_region0 or region0 style keys.')


def _find_region_known_input_keys(files: list[str], num_regions: int) -> list[str] | None:
    prefixes = ['known_inputs_region', 'known_region', 'externalinputs_region']
    for prefix in prefixes:
        keys = [f'{prefix}{r}' for r in range(num_regions)]
        if all(k in files for k in keys):
            return keys
    return None


def _find_region_history_keys(files: list[str], num_regions: int) -> list[str] | None:
    prefixes = ['history_region', 'pretrial_region']
    for prefix in prefixes:
        keys = [f'{prefix}{r}' for r in range(num_regions)]
        if all(k in files for k in keys):
            return keys
    return None


def _build_region_inputs_from_global(
    global_inputs: np.ndarray,
    num_regions: int,
    region_input_dims: list[int] | None,
    broadcast_single_input: bool,
) -> tuple[np.ndarray, np.ndarray]:
    trials, time, total_dim = global_inputs.shape
    if region_input_dims is not None:
        if len(region_input_dims) != num_regions:
            raise ValueError('region_input_dims must have one entry per region')
        if sum(region_input_dims) > total_dim:
            raise ValueError('sum(region_input_dims) cannot exceed input dimension')
        slices = []
        start = 0
        for d in region_input_dims:
            if d > 0:
                slices.append(global_inputs[:, :, start : start + d])
            else:
                slices.append(np.zeros((trials, time, 0), dtype=np.float32))
            start += d
        input_dims = np.asarray(region_input_dims, dtype=np.int32)
    else:
        if total_dim == num_regions:
            slices = [global_inputs[:, :, r : r + 1] for r in range(num_regions)]
            input_dims = np.ones(num_regions, dtype=np.int32)
        elif total_dim == 1 and not broadcast_single_input:
            slices = [global_inputs[:, :, :1]] + [np.zeros((trials, time, 0), dtype=np.float32) for _ in range(num_regions - 1)]
            input_dims = np.asarray([1] + [0] * (num_regions - 1), dtype=np.int32)
        else:
            slices = [global_inputs for _ in range(num_regions)]
            input_dims = np.full(num_regions, total_dim, dtype=np.int32)
    max_dim = int(input_dims.max()) if len(input_dims) else 0
    packed = np.zeros((trials, num_regions, time, max_dim), dtype=np.float32)
    for r in range(num_regions):
        d = int(input_dims[r])
        if d > 0:
            packed[:, r, :, :d] = slices[r]
    return packed, input_dims


def _pad_regions(raw_regions: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, int, int]:
    num_trials = raw_regions[0].shape[0]
    time_max = max(arr.shape[1] for arr in raw_regions)
    neuron_counts = np.asarray([arr.shape[2] for arr in raw_regions], dtype=np.int32)
    neuron_max = int(neuron_counts.max())
    packed = np.zeros((num_trials, len(raw_regions), time_max, neuron_max), dtype=np.float32)
    for r, arr in enumerate(raw_regions):
        packed[:, r, : arr.shape[1], : arr.shape[2]] = arr
    return packed, neuron_counts, time_max, neuron_max


def load_dataset(path: str | Path, model_config: ModelConfig, data_config: DataConfig) -> HybridDataset:
    path = Path(path)
    with np.load(path, allow_pickle=True) as npz:
        files = list(npz.files)
        region_keys = _find_region_spike_keys(files)
        raw_regions = [np.asarray(npz[k], dtype=np.float32) for k in region_keys]
        num_regions = len(raw_regions)
        spikes, num_neurons, time_max, neuron_max = _pad_regions(raw_regions)
        lengths = np.asarray(npz['lengths'], dtype=np.int32) if 'lengths' in files else np.full(spikes.shape[0], time_max, dtype=np.int32)
        times = np.asarray(npz['times'], dtype=np.float32) if 'times' in files else np.arange(spikes.shape[0], dtype=np.float32)
        dt = float(np.asarray(npz['dt']).item()) if 'dt' in files else 0.01
        if model_config.latent_sizes:
            latent_sizes = np.asarray(model_config.latent_sizes, dtype=np.int32)
        elif 'latent_dims_per_region' in files:
            latent_sizes = np.asarray(npz['latent_dims_per_region'], dtype=np.int32)
        else:
            latent_sizes = np.full(num_regions, 2, dtype=np.int32)
        if len(latent_sizes) != num_regions:
            raise ValueError('latent_sizes must match num_regions')

        known_region_keys = _find_region_known_input_keys(files, num_regions)
        if known_region_keys is not None:
            raw_inputs = [np.asarray(npz[k], dtype=np.float32) for k in known_region_keys]
            max_input_dim = max(arr.shape[-1] for arr in raw_inputs)
            known_inputs = np.zeros((spikes.shape[0], num_regions, time_max, max_input_dim), dtype=np.float32)
            input_dims = np.asarray([arr.shape[-1] for arr in raw_inputs], dtype=np.int32)
            for r, arr in enumerate(raw_inputs):
                known_inputs[:, r, : arr.shape[1], : arr.shape[2]] = arr
        elif 'externalinputs' in files:
            global_inputs = np.asarray(npz['externalinputs'], dtype=np.float32)
            region_dims = data_config.region_input_dims
            if region_dims is None and 'region_input_dims' in files:
                region_dims = [int(x) for x in np.asarray(npz['region_input_dims']).tolist()]
            known_inputs, input_dims = _build_region_inputs_from_global(global_inputs, num_regions, region_dims, data_config.broadcast_single_input)
        else:
            known_inputs = np.zeros((spikes.shape[0], num_regions, time_max, 0), dtype=np.float32)
            input_dims = np.zeros((num_regions,), dtype=np.int32)

        history_keys = _find_region_history_keys(files, num_regions)
        if history_keys is not None:
            raw_history = [np.asarray(npz[k], dtype=np.float32) for k in history_keys]
            history_time = max(arr.shape[1] for arr in raw_history)
            history = np.zeros((spikes.shape[0], num_regions, history_time, neuron_max), dtype=np.float32)
            for r, arr in enumerate(raw_history):
                history[:, r, : arr.shape[1], : arr.shape[2]] = arr
            history_lengths = np.asarray(npz['history_lengths'], dtype=np.int32) if 'history_lengths' in files else np.full(spikes.shape[0], history_time, dtype=np.int32)
        else:
            history_time = max(1, data_config.synthetic_history_bins)
            history = np.zeros((spikes.shape[0], num_regions, history_time, neuron_max), dtype=np.float32)
            history_lengths = np.full(spikes.shape[0], history_time, dtype=np.int32)

        latent_max = int(latent_sizes.max())
        neuron_mask = np.zeros((num_regions, neuron_max), dtype=np.float32)
        history_mask = np.zeros((num_regions, neuron_max), dtype=np.float32)
        input_mask = np.zeros((num_regions, known_inputs.shape[-1]), dtype=np.float32)
        latent_mask = np.zeros((num_regions, latent_max), dtype=np.float32)
        for r in range(num_regions):
            neuron_mask[r, : num_neurons[r]] = 1.0
            history_mask[r, : num_neurons[r]] = 1.0
            input_mask[r, : input_dims[r]] = 1.0
            latent_mask[r, : latent_sizes[r]] = 1.0

        baseline_prefit = np.zeros_like(spikes, dtype=np.float32)
        region_names = [f'region{r}' for r in range(num_regions)]
        metadata = {
            k: np.asarray(npz[k])
            for k in files
            if k not in set(region_keys)
            and (known_region_keys is None or k not in set(known_region_keys))
            and (history_keys is None or k not in set(history_keys))
            and k not in {'externalinputs', 'lengths', 'times', 'dt', 'latent_dims_per_region', 'region_input_dims', 'history_lengths'}
        }
    return HybridDataset(
        spikes=spikes,
        known_inputs=known_inputs,
        history=history,
        lengths=lengths,
        times=times,
        dt=dt,
        region_names=region_names,
        num_neurons=num_neurons,
        input_dims=input_dims,
        latent_sizes=latent_sizes,
        neuron_mask=neuron_mask,
        input_mask=input_mask,
        latent_mask=latent_mask,
        history_mask=history_mask,
        history_lengths=history_lengths,
        baseline_prefit=baseline_prefit,
        metadata=metadata,
    )


def split_dataset(dataset: HybridDataset, data_config: DataConfig) -> DatasetSplits:
    num_trials = dataset.spikes.shape[0]
    rng = np.random.default_rng(data_config.shuffle_seed)
    perm = rng.permutation(num_trials)
    n_test = max(1, int(round(num_trials * data_config.test_fraction)))
    n_val = max(1, int(round(num_trials * data_config.val_fraction)))
    n_test = min(n_test, num_trials - 2)
    n_val = min(n_val, num_trials - n_test - 1)
    test_idx = perm[:n_test]
    val_idx = perm[n_test : n_test + n_val]
    train_idx = perm[n_test + n_val :]
    return DatasetSplits(
        train=dataset.select(train_idx),
        val=dataset.select(val_idx),
        test=dataset.select(test_idx),
        full=dataset,
        split_indices={'train': train_idx, 'val': val_idx, 'test': test_idx},
    )


def _fit_across_trial_rate(
    spikes: np.ndarray,
    lengths: np.ndarray,
    trial_times: np.ndarray,
    dt: float,
    train_indices: np.ndarray,
    num_basis: int,
    ridge: float,
) -> np.ndarray:
    B, _, N = spikes.shape
    basis = radial_basis_over_trials(num_basis, trial_times)
    valid_seconds = np.maximum(lengths.astype(np.float32) * dt, 1.0e-6)
    mean_rate = spikes.sum(axis=1) / valid_seconds[:, None]
    X_train = basis[train_indices]
    y_train = mean_rate[train_indices]
    gram = X_train.T @ X_train + ridge * np.eye(X_train.shape[1], dtype=np.float32)
    coef = np.linalg.solve(gram, X_train.T @ y_train)
    pred = basis @ coef
    pred = np.clip(pred, 1.0e-4, None)
    return pred.astype(np.float32)


def _fit_within_trial_rate(
    spikes: np.ndarray,
    lengths: np.ndarray,
    dt: float,
    train_indices: np.ndarray,
    across_rate: np.ndarray,
    num_basis: int,
    ridge: float,
) -> np.ndarray:
    B, T, N = spikes.shape
    basis = raised_cosine_basis(num_basis, T, warp='asinh')
    X_chunks = []
    y_chunks = []
    for idx in train_indices:
        L = int(lengths[idx])
        X_chunks.append(basis[:L])
        base_hz = np.tile(across_rate[idx : idx + 1], (L, 1))
        y_chunks.append(spikes[idx, :L] / max(dt, 1.0e-8) - base_hz)
    X = np.concatenate(X_chunks, axis=0) if X_chunks else basis
    y = np.concatenate(y_chunks, axis=0) if y_chunks else np.zeros((T, N), dtype=np.float32)
    gram = X.T @ X + ridge * np.eye(num_basis, dtype=np.float32)
    coef = np.linalg.solve(gram, X.T @ y)
    within = basis @ coef
    return within.astype(np.float32)


def compute_prefit_baseline(
    dataset: HybridDataset,
    train_indices: np.ndarray,
    model_config: ModelConfig,
) -> np.ndarray:
    B, R, T, Nmax = dataset.spikes.shape
    baseline = np.zeros((B, R, T, Nmax), dtype=np.float32)
    for r in range(R):
        N = int(dataset.num_neurons[r])
        spikes_r = dataset.spikes[:, r, :, :N]
        across = _fit_across_trial_rate(
            spikes_r,
            dataset.lengths,
            dataset.times,
            dataset.dt,
            train_indices,
            model_config.baseline_prefit_num_basis_across,
            model_config.baseline_prefit_ridge,
        )
        within = _fit_within_trial_rate(
            spikes_r,
            dataset.lengths,
            dataset.dt,
            train_indices,
            across,
            model_config.baseline_prefit_num_basis_within,
            model_config.baseline_prefit_ridge,
        )
        rate = across[:, None, :] + within[None, :, :]
        rate = np.clip(rate, 1.0e-4, model_config.baseline_prefit_clip_hz)
        lograte = softplus_inv(rate)
        valid = (np.arange(T)[None, :] < dataset.lengths[:, None]).astype(np.float32)
        lograte = lograte * valid[:, :, None]
        baseline[:, r, :, :N] = lograte
    return baseline.astype(np.float32)


def attach_prefit_baseline(dataset: HybridDataset, train_indices: np.ndarray, model_config: ModelConfig) -> HybridDataset:
    baseline = compute_prefit_baseline(dataset, train_indices, model_config)
    return replace(dataset, baseline_prefit=baseline)


def apply_baseline_mode(dataset: HybridDataset, splits: DatasetSplits, model_config: ModelConfig) -> DatasetSplits:
    if model_config.baseline_mode != 3:
        return splits
    full = attach_prefit_baseline(dataset, splits.split_indices['train'], model_config)
    return DatasetSplits(
        train=full.select(splits.split_indices['train']),
        val=full.select(splits.split_indices['val']),
        test=full.select(splits.split_indices['test']),
        full=full,
        split_indices=splits.split_indices,
    )
