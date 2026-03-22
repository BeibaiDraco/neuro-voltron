from __future__ import annotations

import numpy as np


def softplus_inv(y: np.ndarray | float, eps: float = 1.0e-8) -> np.ndarray:
    y_arr = np.asarray(y, dtype=np.float32)
    y_arr = np.maximum(y_arr, eps)
    return np.log(np.expm1(y_arr)).astype(np.float32)


def _time_warp(x: np.ndarray, warp: str) -> np.ndarray:
    if warp == 'linear':
        return x
    if warp == 'log':
        return np.log1p(x)
    if warp == 'asinh':
        return np.arcsinh(x)
    raise ValueError(f'Unknown warp {warp}')


def raised_cosine_basis(num_basis: int, T: int, warp: str = 'linear') -> np.ndarray:
    if num_basis <= 0:
        raise ValueError('num_basis must be positive')
    if T <= 0:
        raise ValueError('T must be positive')
    x = np.linspace(0.0, 1.0, T, dtype=np.float32)
    xw = _time_warp(x, warp)
    if num_basis == 1:
        return np.ones((T, 1), dtype=np.float32)
    centers = np.linspace(xw.min(), xw.max(), num_basis, dtype=np.float32)
    step = centers[1] - centers[0]
    width = max(step, 1.0e-6)
    basis = np.zeros((T, num_basis), dtype=np.float32)
    for k, c in enumerate(centers):
        d = np.clip((xw - c) * np.pi / width, -np.pi, np.pi)
        vals = 0.5 * (np.cos(d) + 1.0)
        mask = np.abs(xw - c) <= width
        basis[:, k] = np.where(mask, vals, 0.0)
    return basis.astype(np.float32)


def radial_basis_over_trials(num_basis: int, trial_times: np.ndarray) -> np.ndarray:
    times = np.asarray(trial_times, dtype=np.float32).reshape(-1)
    if num_basis <= 0:
        raise ValueError('num_basis must be positive')
    if times.size == 0:
        raise ValueError('trial_times must be non-empty')
    if num_basis == 1:
        return np.ones((times.shape[0], 1), dtype=np.float32)
    t0 = float(times.min())
    t1 = float(times.max())
    if abs(t1 - t0) < 1.0e-8:
        return np.ones((times.shape[0], 1), dtype=np.float32) if num_basis == 1 else np.pad(np.ones((times.shape[0], 1), dtype=np.float32), ((0,0),(0,num_basis-1)))
    centers = np.linspace(t0, t1, num_basis, dtype=np.float32)
    width = max((t1 - t0) / max(num_basis - 1, 1), 1.0e-6)
    Phi = np.exp(-0.5 * ((times[:, None] - centers[None, :]) / width) ** 2).astype(np.float32)
    return Phi


def second_difference_penalty(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float32)
    if arr.shape[0] < 3:
        return 0.0
    diff2 = arr[2:] - 2.0 * arr[1:-1] + arr[:-2]
    return float(np.mean(diff2 ** 2))
