from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np

from .basis import radial_basis_over_trials, raised_cosine_basis


@dataclass
class SyntheticConfig:
    scenario: str = 'three_region_additive'
    seed: int = 0
    n_trials: int = 512
    dt: float = 0.01
    history_steps: int = 20
    main_steps: int = 160
    min_length: int = 130
    max_length: int = 160
    latent_sizes: tuple[int, int, int] = (2, 2, 2)
    neuron_counts: tuple[int, int, int] = (16, 20, 18)
    known_input_dims: tuple[int, int, int] = (3, 2, 1)
    hidden_input_dim: int = 2
    message_dim: int = 4
    process_noise_std: float = 0.05
    hidden_ar: float = 0.95
    hidden_noise_std: float = 0.12
    click_rate_hz: float = 20.0
    stim_onset_range: tuple[int, int] = (15, 30)
    stim_duration_range: tuple[int, int] = (35, 55)
    go_delay_range: tuple[int, int] = (85, 115)
    trial_interval_mean: float = 3.0
    # v5 data generation controls
    use_temporal_baseline: bool = True   # False = const offset only (no within/across nuisance)
    readout_scale: float = 2.5          # scale for semi-orthogonal readout matrices
    offset_mean: float = 1.5            # mean of per-neuron constant offset
    msg_coupling_scale: float = 1.2     # scale for ring edge coupling matrices


def _semi_orthogonal(rng: np.random.Generator, out_dim: int, in_dim: int, scale: float = 1.0) -> np.ndarray:
    a = rng.normal(size=(out_dim, in_dim)).astype(np.float32)
    q, _ = np.linalg.qr(a)
    q = q[:, :in_dim]
    return (scale * q).astype(np.float32)


def _intrinsic_flow(z: np.ndarray, F: np.ndarray, G: np.ndarray, b_f: np.ndarray, b_g: np.ndarray) -> np.ndarray:
    gate = 1.0 / (1.0 + np.exp(-(z @ G.T + b_g)))
    target = np.tanh(z @ F.T + b_f)
    return gate * (-z + target)


def _make_known_inputs(cfg: SyntheticConfig, rng: np.random.Generator) -> tuple[list[np.ndarray], dict[str, Any]]:
    T = cfg.main_steps
    H = cfg.history_steps
    total = H + T
    B = cfg.n_trials
    region_inputs = [np.zeros((B, total, d), dtype=np.float32) for d in cfg.known_input_dims]
    meta: dict[str, Any] = {'channel_meta': {}, 'events': {}}

    stim_on = rng.integers(cfg.stim_onset_range[0], cfg.stim_onset_range[1] + 1, size=B)
    stim_dur = rng.integers(cfg.stim_duration_range[0], cfg.stim_duration_range[1] + 1, size=B)
    go_on = rng.integers(cfg.go_delay_range[0], cfg.go_delay_range[1] + 1, size=B)
    rule = rng.integers(0, 2, size=B)
    evidence_sign = rng.choice([-1, 1], size=B)

    for b in range(B):
        region_inputs[1][b, :, rule[b]] = 1.0

    p_click = cfg.click_rate_hz * cfg.dt
    for b in range(B):
        start = H + stim_on[b]
        stop = min(H + stim_on[b] + stim_dur[b], total)
        region_inputs[0][b, start:stop, 2] = 1.0
        for t in range(start, stop):
            if rng.random() < p_click:
                if evidence_sign[b] > 0:
                    region_inputs[0][b, t, 0] += 1.0
                else:
                    region_inputs[0][b, t, 1] += 1.0
            if rng.random() < 0.25 * p_click:
                if evidence_sign[b] > 0:
                    region_inputs[0][b, t, 1] += 1.0
                else:
                    region_inputs[0][b, t, 0] += 1.0

    for b in range(B):
        start = H + go_on[b]
        region_inputs[2][b, start:, 0] = 1.0

    meta['events']['stim_onset'] = stim_on.tolist()
    meta['events']['stim_duration'] = stim_dur.tolist()
    meta['events']['go_onset'] = go_on.tolist()
    meta['events']['rule'] = rule.tolist()
    meta['events']['evidence_sign'] = evidence_sign.tolist()
    meta['channel_meta']['region0'] = [
        {'name': 'click_left', 'type': 'event'},
        {'name': 'click_right', 'type': 'event'},
        {'name': 'stim_on', 'type': 'step'},
    ]
    meta['channel_meta']['region1'] = [
        {'name': 'rule_A', 'type': 'step'},
        {'name': 'rule_B', 'type': 'step'},
    ]
    meta['channel_meta']['region2'] = [
        {'name': 'go_cue', 'type': 'step'},
    ]
    return region_inputs, meta


def _make_hidden_inputs(cfg: SyntheticConfig, rng: np.random.Generator) -> list[np.ndarray]:
    total = cfg.history_steps + cfg.main_steps
    B = cfg.n_trials
    D = cfg.hidden_input_dim
    common = np.zeros((B, total, D), dtype=np.float32)
    private = [np.zeros((B, total, D), dtype=np.float32) for _ in range(3)]
    common[:, 0] = rng.normal(scale=0.5, size=(B, D))
    for r in range(3):
        private[r][:, 0] = rng.normal(scale=0.5, size=(B, D))
    for t in range(1, total):
        common[:, t] = cfg.hidden_ar * common[:, t - 1] + cfg.hidden_noise_std * rng.normal(size=(B, D))
        for r in range(3):
            private[r][:, t] = 0.85 * private[r][:, t - 1] + 0.6 * cfg.hidden_noise_std * rng.normal(size=(B, D))
    mixes = [
        np.asarray([[0.9, -0.2], [0.1, 0.5]], dtype=np.float32),
        np.asarray([[0.3, 0.4], [-0.6, 0.2]], dtype=np.float32),
        np.asarray([[0.5, 0.2], [0.2, -0.4]], dtype=np.float32),
    ]
    out = []
    for r in range(3):
        out.append(common @ mixes[r].T + 0.7 * private[r])
    return out


def _make_baseline(cfg: SyntheticConfig, trial_times: np.ndarray, rng: np.random.Generator) -> tuple[list[np.ndarray], list[np.ndarray]]:
    total = cfg.history_steps + cfg.main_steps
    within_basis = raised_cosine_basis(4, total, warp='asinh')
    across_basis = radial_basis_over_trials(4, trial_times)
    within_all = []
    across_all = []
    for n in cfg.neuron_counts:
        w_within = rng.normal(scale=0.25, size=(within_basis.shape[1], n)).astype(np.float32)
        within = within_basis @ w_within
        w_across = rng.normal(scale=0.20, size=(across_basis.shape[1], n)).astype(np.float32)
        across = across_basis @ w_across
        within_all.append(within.astype(np.float32))
        across_all.append(across.astype(np.float32))
    return within_all, across_all


def generate_synthetic_dataset(cfg: SyntheticConfig) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if cfg.scenario not in {'three_region_additive', 'three_region_modulatory'}:
        raise ValueError(f'Unknown scenario {cfg.scenario}')
    rng = np.random.default_rng(cfg.seed)
    B = cfg.n_trials
    R = 3
    Ls = cfg.latent_sizes
    Ns = cfg.neuron_counts
    total = cfg.history_steps + cfg.main_steps

    isi = rng.exponential(cfg.trial_interval_mean, size=B).astype(np.float32)
    trial_times = np.cumsum(isi).astype(np.float32)

    known_inputs, input_meta = _make_known_inputs(cfg, rng)
    hidden_inputs = _make_hidden_inputs(cfg, rng)
    within_base, across_base = _make_baseline(cfg, trial_times, rng)

    tau = np.asarray([0.12, 0.10, 0.11], dtype=np.float32)
    F = [
        np.asarray([[1.1, 0.4], [-0.1, 0.7]], dtype=np.float32),
        np.asarray([[0.5, -1.2], [1.1, 0.4]], dtype=np.float32),
        np.asarray([[0.8, 0.3], [0.2, 1.0]], dtype=np.float32),
    ]
    G = [
        np.asarray([[1.0, -0.2], [0.3, 0.8]], dtype=np.float32),
        np.asarray([[0.6, 0.4], [-0.5, 1.1]], dtype=np.float32),
        np.asarray([[0.7, 0.1], [0.2, 0.9]], dtype=np.float32),
    ]
    bF = [np.asarray([0.05, -0.02], dtype=np.float32), np.asarray([0.02, 0.04], dtype=np.float32), np.asarray([-0.03, 0.01], dtype=np.float32)]
    bG = [np.asarray([0.1, -0.1], dtype=np.float32), np.asarray([0.0, 0.05], dtype=np.float32), np.asarray([-0.05, 0.0], dtype=np.float32)]

    B_known = [
        np.asarray([[0.9, -0.9, 0.25], [0.2, 0.2, 0.05]], dtype=np.float32),
        np.asarray([[0.7, -0.7], [0.2, 0.15]], dtype=np.float32),
        np.asarray([[0.6], [0.2]], dtype=np.float32),
    ]
    H_hidden = [
        np.asarray([[0.3, -0.1], [0.1, 0.25]], dtype=np.float32),
        np.asarray([[0.15, 0.25], [-0.2, 0.15]], dtype=np.float32),
        np.asarray([[0.2, 0.1], [0.05, -0.15]], dtype=np.float32),
    ]

    readouts = [_semi_orthogonal(rng, n, 2, scale=1.1) for n in Ns]
    const_offsets = [rng.normal(scale=0.15, size=(n,)).astype(np.float32) for n in Ns]

    msg_src = [[None for _ in range(R)] for _ in range(R)]
    msg_add = [[None for _ in range(R)] for _ in range(R)]
    msg_gain = [[None for _ in range(R)] for _ in range(R)]
    active_edges = {(0, 1): 1.0, (1, 2): 1.0, (0, 2): 0.7, (2, 1): 0.2}
    for s in range(R):
        for t in range(R):
            if s == t:
                msg_src[s][t] = np.zeros((cfg.message_dim, Ns[s]), dtype=np.float32)
                msg_add[s][t] = np.zeros((Ls[t], cfg.message_dim), dtype=np.float32)
                msg_gain[s][t] = np.zeros((Ls[t], cfg.message_dim), dtype=np.float32)
                continue
            scale = active_edges.get((s, t), 0.0)
            msg_src[s][t] = (scale * rng.normal(scale=0.25, size=(cfg.message_dim, Ns[s]))).astype(np.float32)
            msg_add[s][t] = (scale * rng.normal(scale=0.18, size=(Ls[t], cfg.message_dim))).astype(np.float32)
            if cfg.scenario == 'three_region_modulatory':
                gain_scale = 0.25 if (s, t) in {(0, 2), (1, 2)} else 0.05 * scale
                msg_gain[s][t] = (gain_scale * rng.normal(scale=0.15, size=(Ls[t], cfg.message_dim))).astype(np.float32)
            else:
                msg_gain[s][t] = np.zeros((Ls[t], cfg.message_dim), dtype=np.float32)

    z = [np.zeros((B, total, Ls[r]), dtype=np.float32) for r in range(R)]
    messages = [[np.zeros((B, total, cfg.message_dim), dtype=np.float32) for _ in range(R)] for _ in range(R)]
    task_logrates = [np.zeros((B, total, Ns[r]), dtype=np.float32) for r in range(R)]
    task_rates = [np.zeros((B, total, Ns[r]), dtype=np.float32) for r in range(R)]
    full_logrates = [np.zeros((B, total, Ns[r]), dtype=np.float32) for r in range(R)]
    spikes = [np.zeros((B, total, Ns[r]), dtype=np.float32) for r in range(R)]

    for r in range(R):
        context = np.stack([
            known_inputs[1][:, 0, 0] - known_inputs[1][:, 0, 1],
            hidden_inputs[r][:, 0, 0],
        ], axis=1).astype(np.float32)
        init_w = np.asarray([[0.4 + 0.1 * r, 0.2], [0.1, -0.3 - 0.05 * r]], dtype=np.float32)
        z[r][:, 0] = context @ init_w.T + 0.1 * rng.normal(size=(B, 2))

    for r in range(R):
        task_logrates[r][:, 0] = z[r][:, 0] @ readouts[r].T
        task_rates[r][:, 0] = np.log1p(np.exp(task_logrates[r][:, 0]))

    for t in range(1, total):
        for s in range(R):
            for tgt in range(R):
                if s == tgt:
                    continue
                messages[s][tgt][:, t] = task_rates[s][:, t - 1] @ msg_src[s][tgt].T
        for r in range(R):
            intrinsic = _intrinsic_flow(z[r][:, t - 1], F[r], G[r], bF[r], bG[r])
            known_drive = known_inputs[r][:, t] @ B_known[r].T
            hidden_drive = hidden_inputs[r][:, t] @ H_hidden[r].T
            msg_add_drive = np.zeros((B, Ls[r]), dtype=np.float32)
            msg_gain_drive = np.zeros((B, Ls[r]), dtype=np.float32)
            for s in range(R):
                if s == r:
                    continue
                msg_add_drive += messages[s][r][:, t] @ msg_add[s][r].T
                msg_gain_drive += messages[s][r][:, t] @ msg_gain[s][r].T
            drift = intrinsic + known_drive + hidden_drive + msg_add_drive
            if cfg.scenario == 'three_region_modulatory':
                drift = drift + np.tanh(msg_gain_drive) * z[r][:, t - 1]
            noise = cfg.process_noise_std * rng.normal(size=(B, Ls[r])).astype(np.float32)
            z[r][:, t] = z[r][:, t - 1] + (cfg.dt / tau[r]) * drift + np.sqrt(cfg.dt / tau[r]) * noise
            task_logrates[r][:, t] = z[r][:, t] @ readouts[r].T
            task_rates[r][:, t] = np.log1p(np.exp(task_logrates[r][:, t]))

    for r in range(R):
        baseline = within_base[r][None, :, :] + across_base[r][:, None, :] + const_offsets[r][None, None, :]
        full_logrates[r] = task_logrates[r] + baseline
        rate_hz = np.log1p(np.exp(full_logrates[r]))
        lam = np.clip(cfg.dt * rate_hz, 1.0e-5, 5.0)
        spikes[r] = rng.poisson(lam).astype(np.float32)

    lengths = rng.integers(cfg.min_length, cfg.max_length + 1, size=B, dtype=np.int32)
    for b in range(B):
        L = int(lengths[b])
        for r in range(R):
            spikes[r][b, cfg.history_steps + L :] = 0.0
            known_inputs[r][b, cfg.history_steps + L :] = 0.0
            task_logrates[r][b, cfg.history_steps + L :] = 0.0
            task_rates[r][b, cfg.history_steps + L :] = 0.0
            full_logrates[r][b, cfg.history_steps + L :] = 0.0
            z[r][b, cfg.history_steps + L :] = 0.0
            hidden_inputs[r][b, cfg.history_steps + L :] = 0.0
            for s in range(R):
                messages[s][r][b, cfg.history_steps + L :] = 0.0

    payload: dict[str, np.ndarray] = {
        'dt': np.asarray(cfg.dt, dtype=np.float32),
        'times': trial_times.astype(np.float32),
        'lengths': lengths.astype(np.int32),
        'latent_dims_per_region': np.asarray(cfg.latent_sizes, dtype=np.int32),
        'region_input_dims': np.asarray(cfg.known_input_dims, dtype=np.int32),
        'history_lengths': np.full(B, cfg.history_steps, dtype=np.int32),
    }
    for r in range(R):
        payload[f'spikes_region{r}'] = spikes[r][:, cfg.history_steps :].astype(np.float32)
        payload[f'history_region{r}'] = spikes[r][:, : cfg.history_steps].astype(np.float32)
        payload[f'known_inputs_region{r}'] = known_inputs[r][:, cfg.history_steps :].astype(np.float32)
        payload[f'gt_hidden_input_region{r}'] = hidden_inputs[r][:, cfg.history_steps :].astype(np.float32)
        payload[f'gt_latent_region{r}'] = z[r][:, cfg.history_steps :].astype(np.float32)
        payload[f'gt_task_lograte_region{r}'] = task_logrates[r][:, cfg.history_steps :].astype(np.float32)
        payload[f'gt_full_lograte_region{r}'] = full_logrates[r][:, cfg.history_steps :].astype(np.float32)
        payload[f'gt_baseline_region{r}'] = (full_logrates[r] - task_logrates[r])[:, cfg.history_steps :].astype(np.float32)
    for s in range(R):
        for t in range(R):
            if s == t:
                continue
            payload[f'gt_messages_{s}to{t}'] = messages[s][t][:, cfg.history_steps :].astype(np.float32)
            payload[f'gt_message_matrix_{s}to{t}'] = msg_add[s][t].astype(np.float32)
            payload[f'gt_message_gain_matrix_{s}to{t}'] = msg_gain[s][t].astype(np.float32)
    meta = {
        'config': asdict(cfg),
        'scenario': cfg.scenario,
        'channel_meta': input_meta['channel_meta'],
        'events': input_meta['events'],
        'notes': {
            'known_inputs': 'All known inputs are generic per-bin time-series channels. Event-like clicks are sparse pulse channels. Sustained onsets are step channels that stay high after onset.',
            'hidden_inputs': 'Hidden inputs are low-dimensional AR(1) processes from unobserved sources, mixed differently into each region.',
            'communication': 'Messages are emitted from source task-relevant rates with one-bin lag.',
            'baseline': 'Baseline nuisance combines a smooth within-trial PSTH component, a slow across-trial drift, and neuron-specific offsets.',
        },
    }
    return payload, meta


# ---------------------------------------------------------------------------
# Three-region ring scenario: limit cycle + line attractor + double well
# ---------------------------------------------------------------------------

def _limit_cycle_flow(z: np.ndarray, radius: float = 1.0, speed: float = 1.5) -> np.ndarray:
    x, y = z[..., 0], z[..., 1]
    r2 = x ** 2 + y ** 2
    dx = speed * (-y + x * (radius ** 2 - r2))
    dy = speed * (x + y * (radius ** 2 - r2))
    return np.stack([dx, dy], axis=-1)


def _line_attractor_flow(z: np.ndarray, decay_perp: float = 2.0) -> np.ndarray:
    x, y = z[..., 0], z[..., 1]
    dx = np.zeros_like(x)
    dy = -decay_perp * y
    return np.stack([dx, dy], axis=-1)


def _double_well_flow(z: np.ndarray, depth: float = 2.0, sep: float = 1.0, y_decay: float = 2.0) -> np.ndarray:
    x, y = z[..., 0], z[..., 1]
    dx = -depth * x * (x ** 2 - sep ** 2) / sep ** 4
    dy = -y_decay * y
    return np.stack([dx, dy], axis=-1)


RING_FLOW_FNS = [_limit_cycle_flow, _line_attractor_flow, _double_well_flow]


def _make_ring_known_inputs(cfg: SyntheticConfig, rng: np.random.Generator) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Known inputs for the ring scenario.

    R0 (limit cycle): 2 channels -- smooth ramp (evidence accumulation) + pulse (perturbation).
    R1 (line attractor): 1 channel -- step onset (context/bias).
    R2 (double well): 1 channel -- brief pulse (kick toward one well).
    """
    T = cfg.main_steps
    H = cfg.history_steps
    total = H + T
    B = cfg.n_trials
    input_dims = (2, 1, 1)
    region_inputs = [np.zeros((B, total, d), dtype=np.float32) for d in input_dims]
    meta: dict[str, Any] = {'channel_meta': {}, 'events': {}}

    stim_on = rng.integers(10, 25, size=B)
    kick_on = rng.integers(60, 90, size=B)
    direction = rng.choice([-1.0, 1.0], size=B)

    for b in range(B):
        start = H + int(stim_on[b])
        end = min(start + 60, total)
        ramp = np.linspace(0.0, float(direction[b]), end - start, dtype=np.float32)
        region_inputs[0][b, start:end, 0] = ramp
        n_pulses = rng.integers(3, 8)
        pulse_times = rng.integers(start, min(end, total - 1), size=n_pulses)
        for pt in pulse_times:
            region_inputs[0][b, pt, 1] = float(direction[b]) * 0.5

    for b in range(B):
        onset = H + int(stim_on[b])
        region_inputs[1][b, onset:, 0] = float(direction[b])

    for b in range(B):
        k = H + int(kick_on[b])
        dur = min(5, total - k)
        region_inputs[2][b, k:k + dur, 0] = float(direction[b]) * 1.0

    meta['events']['stim_onset'] = stim_on.tolist()
    meta['events']['kick_onset'] = kick_on.tolist()
    meta['events']['direction'] = direction.tolist()
    meta['channel_meta']['region0'] = [
        {'name': 'evidence_ramp', 'type': 'ramp'},
        {'name': 'perturbation', 'type': 'event'},
    ]
    meta['channel_meta']['region1'] = [
        {'name': 'context_step', 'type': 'step'},
    ]
    meta['channel_meta']['region2'] = [
        {'name': 'kick_pulse', 'type': 'pulse'},
    ]
    return region_inputs, meta, input_dims


def _make_ring_hidden_inputs(cfg: SyntheticConfig, rng: np.random.Generator) -> list[np.ndarray]:
    """Hidden inputs -- non-trivial confounders (5-10% of drift variance target)."""
    total = cfg.history_steps + cfg.main_steps
    B = cfg.n_trials
    D = cfg.hidden_input_dim
    # v5: stronger AR(1) processes for detectable hidden inputs
    hidden_noise_common = getattr(cfg, 'hidden_noise_std', 0.12)
    hidden_noise_private = hidden_noise_common * 0.75
    common = np.zeros((B, total, D), dtype=np.float32)
    private = [np.zeros((B, total, D), dtype=np.float32) for _ in range(3)]
    common[:, 0] = rng.normal(scale=0.3, size=(B, D))
    for r in range(3):
        private[r][:, 0] = rng.normal(scale=0.3, size=(B, D))
    for t in range(1, total):
        common[:, t] = 0.95 * common[:, t - 1] + hidden_noise_common * rng.normal(size=(B, D))
        for r in range(3):
            private[r][:, t] = 0.90 * private[r][:, t - 1] + hidden_noise_private * rng.normal(size=(B, D))
    mixes = [
        np.asarray([[0.3, -0.08], [0.05, 0.2]], dtype=np.float32),
        np.asarray([[0.1, 0.15], [-0.2, 0.08]], dtype=np.float32),
        np.asarray([[0.18, 0.05], [0.08, -0.12]], dtype=np.float32),
    ]
    out = []
    for r in range(3):
        out.append(common @ mixes[r].T + 0.3 * private[r])
    return out


def _make_ring_baseline(cfg: SyntheticConfig, trial_times: np.ndarray, neuron_counts: tuple, rng: np.random.Generator) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Reduced baseline variance so task signal dominates."""
    total = cfg.history_steps + cfg.main_steps
    within_basis = raised_cosine_basis(4, total, warp='asinh')
    across_basis = radial_basis_over_trials(4, trial_times)
    within_all, across_all = [], []
    for n in neuron_counts:
        w_within = rng.normal(scale=0.10, size=(within_basis.shape[1], n)).astype(np.float32)
        within_all.append((within_basis @ w_within).astype(np.float32))
        w_across = rng.normal(scale=0.08, size=(across_basis.shape[1], n)).astype(np.float32)
        across_all.append((across_basis @ w_across).astype(np.float32))
    return within_all, across_all


def generate_ring_dataset(cfg: SyntheticConfig) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Three-region ring: limit cycle -> line attractor -> double well -> (back to limit cycle)."""
    rng = np.random.default_rng(cfg.seed)
    B = cfg.n_trials
    R = 3
    Ls = list(cfg.latent_sizes)
    Ns = list(cfg.neuron_counts)
    total = cfg.history_steps + cfg.main_steps

    isi = rng.exponential(cfg.trial_interval_mean, size=B).astype(np.float32)
    trial_times = np.cumsum(isi).astype(np.float32)

    known_inputs, input_meta, input_dims = _make_ring_known_inputs(cfg, rng)
    hidden_inputs = _make_ring_hidden_inputs(cfg, rng)
    # v5: no temporal/trial baseline -- only constant neuron offsets
    # This ensures all temporal variance must come from latent dynamics
    use_temporal_baseline = getattr(cfg, 'use_temporal_baseline', True)
    if use_temporal_baseline:
        within_base, across_base = _make_ring_baseline(cfg, trial_times, tuple(Ns), rng)

    tau = np.asarray([0.10, 0.12, 0.11], dtype=np.float32)

    # v5: Known input coupling ~2x stronger (target: 10-20% of drift variance)
    B_known = [
        np.asarray([[0.8, 0.30], [0.30, -0.6]], dtype=np.float32),   # R0: 2 inputs -> 2 latent
        np.asarray([[0.9], [0.2]], dtype=np.float32),                 # R1: 1 input -> 2 latent
        np.asarray([[1.0], [0.30]], dtype=np.float32),                # R2: 1 input -> 2 latent
    ]
    # v5: Hidden input coupling ~10x stronger (target: 5-10% of drift variance)
    H_hidden = [
        np.asarray([[0.30, -0.20], [0.20, 0.30]], dtype=np.float32),
        np.asarray([[0.30, 0.40], [-0.20, 0.30]], dtype=np.float32),
        np.asarray([[0.40, 0.20], [0.10, -0.30]], dtype=np.float32),
    ]

    # v5: Larger readout for stronger observation signal; lower offset for higher modulation depth
    readout_scale = getattr(cfg, 'readout_scale', 2.5)
    offset_mean = getattr(cfg, 'offset_mean', 1.5)
    readouts = [_semi_orthogonal(rng, n, 2, scale=readout_scale) for n in Ns]
    const_offsets = [rng.normal(loc=offset_mean, scale=0.10, size=(n,)).astype(np.float32) for n in Ns]

    # v5: Ring connectivity reduced from 1.2 to 0.8 (messages already 46-52% of drift)
    # With baseline removed, even moderate messages will be observable
    msg_coupling_scale = getattr(cfg, 'msg_coupling_scale', 1.2)
    ring_edges = {(0, 1): msg_coupling_scale, (1, 2): msg_coupling_scale, (2, 0): msg_coupling_scale}
    M = cfg.message_dim
    msg_src = [[np.zeros((M, 2), dtype=np.float32) for _ in range(R)] for _ in range(R)]
    msg_add = [[np.zeros((2, M), dtype=np.float32) for _ in range(R)] for _ in range(R)]
    msg_gain = [[np.zeros((2, M), dtype=np.float32) for _ in range(R)] for _ in range(R)]
    for (s, t), scale in ring_edges.items():
        msg_src[s][t] = (scale * rng.normal(scale=0.30, size=(M, 2))).astype(np.float32)
        msg_add[s][t] = (scale * rng.normal(scale=0.25, size=(2, M))).astype(np.float32)

    z = [np.zeros((B, total, 2), dtype=np.float32) for _ in range(R)]
    messages = [[np.zeros((B, total, M), dtype=np.float32) for _ in range(R)] for _ in range(R)]
    task_logrates = [np.zeros((B, total, Ns[r]), dtype=np.float32) for r in range(R)]
    task_rates = [np.zeros((B, total, Ns[r]), dtype=np.float32) for r in range(R)]
    full_logrates = [np.zeros((B, total, Ns[r]), dtype=np.float32) for r in range(R)]
    spikes = [np.zeros((B, total, Ns[r]), dtype=np.float32) for r in range(R)]

    # Initial conditions
    for r in range(R):
        if r == 0:
            theta = rng.uniform(0, 2 * np.pi, size=B)
            rad = 0.8 + 0.2 * rng.normal(size=B)
            z[r][:, 0, 0] = rad * np.cos(theta)
            z[r][:, 0, 1] = rad * np.sin(theta)
        elif r == 1:
            z[r][:, 0, 0] = rng.normal(scale=0.3, size=B)
            z[r][:, 0, 1] = rng.normal(scale=0.3, size=B)
        else:
            z[r][:, 0, 0] = rng.normal(scale=0.15, size=B)
            z[r][:, 0, 1] = rng.normal(scale=0.3, size=B)

    for r in range(R):
        task_logrates[r][:, 0] = z[r][:, 0] @ readouts[r].T
        task_rates[r][:, 0] = np.log1p(np.exp(task_logrates[r][:, 0]))

    flow_fns = RING_FLOW_FNS
    for t in range(1, total):
        # Messages: source is tanh(z) for boundedness
        for s in range(R):
            for tgt in range(R):
                if s == tgt or (s, tgt) not in ring_edges:
                    continue
                src_signal = np.tanh(z[s][:, t - 1])  # bounded [-1, 1]
                messages[s][tgt][:, t] = src_signal @ msg_src[s][tgt].T

        for r in range(R):
            intrinsic = flow_fns[r](z[r][:, t - 1])
            known_drive = known_inputs[r][:, t, :input_dims[r]] @ B_known[r].T
            hidden_drive = hidden_inputs[r][:, t] @ H_hidden[r].T
            msg_add_drive = np.zeros((B, 2), dtype=np.float32)
            for s in range(R):
                if s == r or (s, r) not in ring_edges:
                    continue
                msg_add_drive += messages[s][r][:, t] @ msg_add[s][r].T
            drift = intrinsic + known_drive + hidden_drive + msg_add_drive
            noise = cfg.process_noise_std * rng.normal(size=(B, 2)).astype(np.float32)
            z[r][:, t] = z[r][:, t - 1] + (cfg.dt / tau[r]) * drift + np.sqrt(cfg.dt / tau[r]) * noise
            task_logrates[r][:, t] = z[r][:, t] @ readouts[r].T
            task_rates[r][:, t] = np.log1p(np.exp(task_logrates[r][:, t]))

    for r in range(R):
        if use_temporal_baseline:
            baseline = within_base[r][None, :, :] + across_base[r][:, None, :] + const_offsets[r][None, None, :]
        else:
            baseline = const_offsets[r][None, None, :]
        full_logrates[r] = task_logrates[r] + baseline
        rate_hz = np.log1p(np.exp(full_logrates[r]))
        lam = np.clip(cfg.dt * rate_hz, 1.0e-5, 5.0)
        spikes[r] = rng.poisson(lam).astype(np.float32)

    lengths = rng.integers(cfg.min_length, cfg.max_length + 1, size=B, dtype=np.int32)
    for b in range(B):
        L = int(lengths[b])
        for r in range(R):
            spikes[r][b, cfg.history_steps + L:] = 0.0
            known_inputs[r][b, cfg.history_steps + L:] = 0.0
            task_logrates[r][b, cfg.history_steps + L:] = 0.0
            task_rates[r][b, cfg.history_steps + L:] = 0.0
            full_logrates[r][b, cfg.history_steps + L:] = 0.0
            z[r][b, cfg.history_steps + L:] = 0.0
            hidden_inputs[r][b, cfg.history_steps + L:] = 0.0
            for s in range(R):
                messages[s][r][b, cfg.history_steps + L:] = 0.0

    H = cfg.history_steps
    payload: dict[str, np.ndarray] = {
        'dt': np.asarray(cfg.dt, dtype=np.float32),
        'times': trial_times.astype(np.float32),
        'lengths': lengths.astype(np.int32),
        'latent_dims_per_region': np.asarray(Ls, dtype=np.int32),
        'region_input_dims': np.asarray(input_dims, dtype=np.int32),
        'history_lengths': np.full(B, H, dtype=np.int32),
    }
    for r in range(R):
        payload[f'spikes_region{r}'] = spikes[r][:, H:].astype(np.float32)
        payload[f'history_region{r}'] = spikes[r][:, :H].astype(np.float32)
        payload[f'known_inputs_region{r}'] = known_inputs[r][:, H:, :input_dims[r]].astype(np.float32)
        payload[f'gt_hidden_input_region{r}'] = hidden_inputs[r][:, H:].astype(np.float32)
        payload[f'gt_latent_region{r}'] = z[r][:, H:].astype(np.float32)
        payload[f'gt_task_lograte_region{r}'] = task_logrates[r][:, H:].astype(np.float32)
        payload[f'gt_full_lograte_region{r}'] = full_logrates[r][:, H:].astype(np.float32)
        payload[f'gt_baseline_region{r}'] = (full_logrates[r] - task_logrates[r])[:, H:].astype(np.float32)
    for s in range(R):
        for t in range(R):
            if s == t:
                continue
            payload[f'gt_messages_{s}to{t}'] = messages[s][t][:, H:].astype(np.float32)
            payload[f'gt_message_matrix_{s}to{t}'] = msg_add[s][t].astype(np.float32)
            payload[f'gt_message_gain_matrix_{s}to{t}'] = msg_gain[s][t].astype(np.float32)

    meta = {
        'config': {**asdict(cfg), 'known_input_dims': list(input_dims)},
        'scenario': cfg.scenario,
        'channel_meta': input_meta['channel_meta'],
        'events': input_meta['events'],
        'ring_edges': {f'{s}to{t}': scale for (s, t), scale in ring_edges.items()},
        'flow_types': ['limit_cycle', 'line_attractor', 'double_well'],
        'notes': {
            'connectivity': 'Ring: 0->1->2->0. Message source uses tanh(z) for boundedness.',
            'flows': 'R0=limit cycle (Hopf), R1=line attractor, R2=double well.',
            'baseline': 'Reduced variance (within=0.10, across=0.08) so task signal dominates.',
            'offset': 'Mean offset ~1.5 for informative firing rates (~5-15 Hz).',
        },
    }
    return payload, meta


def save_synthetic_dataset(out_path: str | Path, payload: dict[str, np.ndarray], meta: dict[str, Any]) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)
    with open(out_path.with_suffix('.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
