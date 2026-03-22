from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

Array = Any


def effectome_from_messages(messages: np.ndarray) -> np.ndarray:
    if messages.ndim != 5:
        raise ValueError('messages must have shape [trials, source, target, time, dim]')
    norms = np.linalg.norm(messages, axis=(-1, -2))
    return norms.mean(axis=0).T


def effectome_cosine_similarity(inferred: np.ndarray, ground_truth: np.ndarray) -> float:
    a = inferred.reshape(-1)
    b = ground_truth.reshape(-1)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 1.0e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def extract_ground_truth_effectome(metadata: dict[str, Any], num_regions: int) -> np.ndarray:
    effectome = np.zeros((num_regions, num_regions), dtype=np.float32)
    for s in range(num_regions):
        for t in range(num_regions):
            if s == t:
                continue
            key = f'gt_messages_{s}to{t}'
            if key in metadata:
                arr = np.asarray(metadata[key])
                effectome[t, s] = np.linalg.norm(arr.reshape(-1, arr.shape[-1]), axis=-1).mean()
            key = f'gt_message_matrix_{s}to{t}'
            if key in metadata:
                effectome[t, s] = np.linalg.norm(np.asarray(metadata[key]))
    return effectome


def cross_validated_linear_r2(x: np.ndarray, y: np.ndarray, n_splits: int = 5, seed: int = 0) -> float:
    if x.shape[0] != y.shape[0]:
        raise ValueError('x and y must have same samples')
    if x.shape[0] < n_splits:
        n_splits = max(2, x.shape[0] // 2)
    if n_splits < 2:
        return float('nan')
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for train_idx, test_idx in kf.split(x):
        reg = LinearRegression()
        reg.fit(x[train_idx], y[train_idx])
        scores.append(reg.score(x[test_idx], y[test_idx]))
    return float(np.mean(scores))


def message_r2_scores(inferred_messages: np.ndarray, metadata: dict[str, Any], num_regions: int, seed: int = 0) -> dict[str, float]:
    scores: dict[str, float] = {}
    for s in range(num_regions):
        for t in range(num_regions):
            if s == t:
                continue
            key = f'gt_messages_{s}to{t}'
            if key not in metadata:
                continue
            gt = np.asarray(metadata[key])
            inf = inferred_messages[:, s, t]
            x = inf.reshape(-1, inf.shape[-1])
            y = gt.reshape(-1, gt.shape[-1])
            scores[f'{s}to{t}'] = cross_validated_linear_r2(x, y, seed=seed)
    return scores
