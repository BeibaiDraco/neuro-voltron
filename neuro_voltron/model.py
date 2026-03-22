from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .basis import raised_cosine_basis
from .config import ModelConfig

Array = Any


# ---------------------------------------------------------------------------
# Batched helper modules: operate on packed [B, R, ...] region tensors
# or [B, E, ...] edge tensors — one fused kernel per call, not per-region.
# ---------------------------------------------------------------------------

def _init_batched_xavier(param: nn.Parameter) -> None:
    for i in range(param.shape[0]):
        nn.init.xavier_uniform_(param.data[i : i + 1])


def _batched_linear(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None) -> torch.Tensor:
    y = torch.einsum('bri,roi->bro', x, w)
    if b is not None:
        y = y + b
    return y


def _batched_gru_step(
    x: torch.Tensor, h: torch.Tensor,
    wih: torch.Tensor, whh: torch.Tensor,
    bih: torch.Tensor, bhh: torch.Tensor,
) -> torch.Tensor:
    H = h.shape[-1]
    gi = torch.einsum('bri,rji->brj', x, wih) + bih
    gh = torch.einsum('brh,rjh->brj', h, whh) + bhh
    i_r, i_z, i_n = gi[..., :H], gi[..., H:2*H], gi[..., 2*H:]
    h_r, h_z, h_n = gh[..., :H], gh[..., H:2*H], gh[..., 2*H:]
    reset = torch.sigmoid(i_r + h_r)
    update = torch.sigmoid(i_z + h_z)
    new = torch.tanh(i_n + reset * h_n)
    return (1 - update) * new + update * h


class BatchedFlowMLP(nn.Module):
    """R parallel gated-relaxation flow networks (FINDR-style)."""

    def __init__(self, n: int, state_dim: int, cond_dim: int, hidden_sizes: list[int]) -> None:
        super().__init__()
        if not hidden_sizes:
            raise ValueError('hidden_sizes must be non-empty')
        self.state_dim = state_dim
        self.cond_dim = cond_dim
        self.has_cond = cond_dim > 0

        self.gate_state_w = nn.Parameter(torch.empty(n, state_dim, state_dim))
        self.gate_state_b = nn.Parameter(torch.zeros(n, state_dim))
        _init_batched_xavier(self.gate_state_w)

        if self.has_cond:
            self.gate_cond_w = nn.Parameter(torch.empty(n, state_dim, cond_dim))
            _init_batched_xavier(self.gate_cond_w)

        dims_in = [state_dim + cond_dim] + list(hidden_sizes[:-1])
        dims_out = list(hidden_sizes)
        self.layer_w = nn.ParameterList()
        self.layer_b = nn.ParameterList()
        for d_in, d_out in zip(dims_in, dims_out):
            w = nn.Parameter(torch.empty(n, d_out, d_in))
            _init_batched_xavier(w)
            self.layer_w.append(w)
            self.layer_b.append(nn.Parameter(torch.zeros(n, d_out)))

        self.out_w = nn.Parameter(torch.empty(n, state_dim, hidden_sizes[-1]))
        self.out_b = nn.Parameter(torch.zeros(n, state_dim))
        _init_batched_xavier(self.out_w)

    def forward(self, state: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        gate = torch.einsum('bri,roi->bro', state, self.gate_state_w) + self.gate_state_b
        if self.has_cond and cond is not None:
            gate = gate + torch.einsum('bri,roi->bro', cond, self.gate_cond_w)
        gate = torch.sigmoid(gate)

        h = torch.cat([state, cond], dim=-1) if (self.has_cond and cond is not None) else state
        for w, b in zip(self.layer_w, self.layer_b):
            h = F.silu(torch.einsum('bri,roi->bro', h, w) + b)
        target = torch.tanh(torch.einsum('bri,roi->bro', h, self.out_w) + self.out_b)
        return gate * (-state + target)


class BatchedEdges(nn.Module):
    """All directed inter-region message edges computed in one batched call."""

    def __init__(
        self, edge_indices: list[tuple[int, int]],
        n_regions: int, src_dim: int, max_latent: int, config: ModelConfig,
    ) -> None:
        super().__init__()
        E = len(edge_indices)
        M = config.message_dim
        self.n_edges = E
        self.n_regions = n_regions
        self.msg_dim = M
        self.max_latent = max_latent
        self.stochastic = config.message_stochastic
        self.use_gain = config.coupling_mode == 'additive_gain'

        src_idx = torch.tensor([s for s, _ in edge_indices], dtype=torch.long)
        tgt_idx = torch.tensor([t for _, t in edge_indices], dtype=torch.long)
        self.register_buffer('src_idx', src_idx)
        self.register_buffer('tgt_idx', tgt_idx)
        self.register_buffer('flat_idx', src_idx * n_regions + tgt_idx)

        self.mean_w = nn.Parameter(torch.empty(E, M, src_dim))
        _init_batched_xavier(self.mean_w)
        self.mean_b = nn.Parameter(torch.zeros(E, M)) if config.use_message_bias else None

        if self.stochastic:
            self.logstd_w = nn.Parameter(torch.empty(E, M, src_dim))
            _init_batched_xavier(self.logstd_w)
            self.logstd_b = nn.Parameter(torch.zeros(E, M)) if config.use_message_bias else None

        self.add_w = nn.Parameter(torch.zeros(E, max_latent, M))
        if self.use_gain:
            self.gain_w = nn.Parameter(torch.zeros(E, max_latent, M))

    def forward(self, source: torch.Tensor, deterministic: bool):
        B = source.shape[0]
        E, R, M, L = self.n_edges, self.n_regions, self.msg_dim, self.max_latent

        src = source[:, self.src_idx, :]                              # [B, E, S]
        mean = torch.einsum('bes,ems->bem', src, self.mean_w)        # [B, E, M]
        if self.mean_b is not None:
            mean = mean + self.mean_b

        if self.stochastic:
            logstd = torch.einsum('bes,ems->bem', src, self.logstd_w)
            if self.logstd_b is not None:
                logstd = logstd + self.logstd_b
            std = F.softplus(logstd)
            msg = mean if deterministic else mean + std * torch.randn_like(mean)
        else:
            logstd = torch.zeros_like(mean)
            msg = mean

        add = torch.einsum('bem,elm->bel', msg, self.add_w)          # [B, E, L]
        tgt_exp = self.tgt_idx[None, :, None].expand(B, E, L)
        inc_add = source.new_zeros(B, R, L).scatter_add(1, tgt_exp, add)

        if self.use_gain:
            gain = torch.einsum('bem,elm->bel', msg, self.gain_w)
            inc_gain = source.new_zeros(B, R, L).scatter_add(1, tgt_exp, gain)
        else:
            inc_gain = source.new_zeros(B, R, L)

        flat_exp = self.flat_idx[None, :, None].expand(B, E, M)
        mm = source.new_zeros(B, R * R, M).scatter_(1, flat_exp, mean).view(B, R, R, M)
        ml = source.new_zeros(B, R * R, M).scatter_(1, flat_exp, logstd).view(B, R, R, M)
        ms = source.new_zeros(B, R * R, M).scatter_(1, flat_exp, msg).view(B, R, R, M)

        return inc_add, inc_gain, mm, ml, ms


# ---------------------------------------------------------------------------
# Per-region encoder (pre-loop; already fast with fused bidirectional GRU)
# ---------------------------------------------------------------------------

class RegionEncoder(nn.Module):
    def __init__(self, region_index: int, n_neurons: int, n_inputs: int, latent_dim: int, config: ModelConfig) -> None:
        super().__init__()
        self.region_index = int(region_index)
        self.n_neurons = int(n_neurons)
        self.n_inputs = int(n_inputs)
        self.latent_dim = int(latent_dim)
        enc_h = int(config.encoder_hidden_size)
        hist_h = int(config.history_hidden_size)
        self.enc_dim = 2 * enc_h
        self.hist_dim = 2 * hist_h
        self.config = config

        self.encoder_rnn = nn.GRU(n_neurons + n_inputs, enc_h, batch_first=True, bidirectional=True)
        self.history_rnn = nn.GRU(n_neurons, hist_h, batch_first=True, bidirectional=True)
        self.z0_mean = nn.Linear(self.hist_dim, latent_dim)
        self.z0_logstd = nn.Linear(self.hist_dim, latent_dim)

        self.const_bias = nn.Parameter(torch.zeros(n_neurons))
        if config.baseline_mode == 2:
            self.baseline_within = nn.Parameter(torch.zeros(config.baseline_num_basis_within, n_neurons))
            if config.baseline_trial_rank > 0:
                self.baseline_trial_proj = nn.Linear(self.hist_dim, config.baseline_trial_rank)
                self.baseline_trial_to_neuron = nn.Linear(config.baseline_trial_rank, n_neurons)
            else:
                self.baseline_trial_proj = None
                self.baseline_trial_to_neuron = None
        else:
            self.baseline_within = None
            self.baseline_trial_proj = None
            self.baseline_trial_to_neuron = None

    def encode_sequence(self, spikes: torch.Tensor, known: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([spikes, known], dim=-1)
        packed = pack_padded_sequence(inp, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False)
        output, _ = self.encoder_rnn(packed)
        output, _ = pad_packed_sequence(output, batch_first=True, total_length=spikes.shape[1])
        return output

    def encode_history(self, history: torch.Tensor, history_lengths: torch.Tensor) -> torch.Tensor:
        B, T, _ = history.shape
        packed = pack_padded_sequence(history, history_lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False)
        output, _ = self.history_rnn(packed)
        output, _ = pad_packed_sequence(output, batch_first=True, total_length=T)
        valid = (torch.arange(T, device=history.device).unsqueeze(0) < history_lengths.unsqueeze(1)).float().unsqueeze(-1)
        return (output * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

    def initial_state(self, hist_sum: torch.Tensor, deterministic: bool):
        mean = self.z0_mean(hist_sum)
        logstd_raw = self.z0_logstd(hist_sum)
        std = F.softplus(logstd_raw)
        sample = mean if deterministic else mean + std * torch.randn_like(mean)
        return mean, logstd_raw, sample

    def baseline(self, hist_sum: torch.Tensor, baseline_prefit: torch.Tensor, time_basis: torch.Tensor) -> torch.Tensor:
        B, T = hist_sum.shape[0], baseline_prefit.shape[1]
        if self.config.baseline_mode == 0:
            return baseline_prefit.new_zeros(B, T, self.n_neurons)
        if self.config.baseline_mode == 1:
            return self.const_bias.view(1, 1, -1).expand(B, T, -1)
        if self.config.baseline_mode == 3:
            return baseline_prefit
        assert self.baseline_within is not None
        within = time_basis @ self.baseline_within
        bl = within.unsqueeze(0).expand(B, T, self.n_neurons)
        if self.baseline_trial_proj is not None and self.baseline_trial_to_neuron is not None:
            bl = bl + self.baseline_trial_to_neuron(torch.tanh(self.baseline_trial_proj(hist_sum))).unsqueeze(1)
        return bl


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class ForwardOutputs:
    z: torch.Tensor              # [B, R, T, Lmax]
    task_logrates: torch.Tensor  # [B, R, T, Nmax]
    full_logrates: torch.Tensor  # [B, R, T, Nmax]
    rates: torch.Tensor          # [B, R, T, Nmax]
    task_rates: torch.Tensor     # [B, R, T, Nmax]
    baseline: torch.Tensor       # [B, R, T, Nmax]
    prior_drift: torch.Tensor    # [B, R, T, Lmax]
    post_drift: torch.Tensor     # [B, R, T, Lmax]
    u_mean: torch.Tensor         # [B, R, T, U]
    u_logstd: torch.Tensor       # [B, R, T, U]
    u_sample: torch.Tensor       # [B, R, T, U]
    hidden_drive: torch.Tensor   # [B, R, T, Lmax]
    msg_mean: torch.Tensor       # [B, S, TGT, T, M]
    msg_logstd: torch.Tensor     # [B, S, TGT, T, M]
    msg_sample: torch.Tensor     # [B, S, TGT, T, M]
    msg_add: torch.Tensor        # [B, R, T, Lmax]
    msg_gain: torch.Tensor       # [B, R, T, Lmax]
    z0_mean: torch.Tensor        # [B, R, Lmax]
    z0_logstd: torch.Tensor      # [B, R, Lmax]
    z0_sample: torch.Tensor      # [B, R, Lmax]
    history_summary: torch.Tensor  # [B, R, Hmax]
    process_std: torch.Tensor    # [R, Lmax]


# ---------------------------------------------------------------------------
# Main model — tensorized recurrence
# ---------------------------------------------------------------------------

class NeuroVoltron(nn.Module):
    """Hybrid multi-region FINDR / MR-LFADS with tensorized recurrence."""

    def __init__(self, num_neurons: list[int], input_dims: list[int], latent_sizes: list[int], config: ModelConfig) -> None:
        super().__init__()
        self.num_neurons = [int(x) for x in num_neurons]
        self.input_dims = [int(x) for x in input_dims]
        self.latent_sizes = [int(x) for x in latent_sizes]
        self.config = config
        R = len(num_neurons)
        self.n_regions = R
        N = max(num_neurons)
        self.max_neurons = N
        L = max(latent_sizes)
        self.max_latent = L
        K = max(input_dims) if input_dims and max(input_dims) > 0 else 0
        self.max_input = K
        self.max_hist = 2 * int(config.history_hidden_size)
        self.max_hidden_input = int(config.hidden_input_dim)
        H_enc = 2 * int(config.encoder_hidden_size)
        H_ctrl = int(config.controller_hidden_size)
        U = int(config.hidden_input_dim)
        M = int(config.message_dim)

        edge_mask = config.edge_mask if config.edge_mask is not None else \
            [[0.0 if i == j else 1.0 for j in range(R)] for i in range(R)]
        self.register_buffer('edge_mask', torch.tensor(edge_mask, dtype=torch.float32))

        lat_mask = torch.zeros(R, L)
        neu_mask = torch.zeros(R, N)
        for r in range(R):
            lat_mask[r, :latent_sizes[r]] = 1.0
            neu_mask[r, :num_neurons[r]] = 1.0
        self.register_buffer('lat_dim_mask', lat_mask)
        self.register_buffer('neu_dim_mask', neu_mask)

        # Per-region encoders (pre-loop, already fast with fused GRU)
        self.regions = nn.ModuleList([
            RegionEncoder(i, num_neurons[i], input_dims[i], latent_sizes[i], config)
            for i in range(R)
        ])

        # Readout: z → task_lograte  (no bias; const_bias lives in encoder for baseline)
        self.readout_w = nn.Parameter(torch.zeros(R, N, L))
        for r in range(R):
            nn.init.xavier_uniform_(self.readout_w.data[r : r + 1, :num_neurons[r], :latent_sizes[r]])

        # Known-input → latent
        if K > 0:
            self.known_w = nn.Parameter(torch.zeros(R, L, K))
            for r in range(R):
                if input_dims[r] > 0:
                    nn.init.xavier_uniform_(self.known_w.data[r : r + 1, :latent_sizes[r], :input_dims[r]])
            self.known_b = nn.Parameter(torch.zeros(R, L)) if config.use_known_input_bias else None
        else:
            self.known_w = None
            self.known_b = None

        # Controller GRU (batched over regions)
        ctrl_in = H_enc + L
        k_gru = 1.0 / (H_ctrl ** 0.5)
        self.ctrl_wih = nn.Parameter(torch.empty(R, 3 * H_ctrl, ctrl_in).uniform_(-k_gru, k_gru))
        self.ctrl_whh = nn.Parameter(torch.empty(R, 3 * H_ctrl, H_ctrl).uniform_(-k_gru, k_gru))
        self.ctrl_bih = nn.Parameter(torch.empty(R, 3 * H_ctrl).uniform_(-k_gru, k_gru))
        self.ctrl_bhh = nn.Parameter(torch.empty(R, 3 * H_ctrl).uniform_(-k_gru, k_gru))

        # Hidden-input projections (batched)
        self.u_mean_w = nn.Parameter(torch.empty(R, U, H_ctrl))
        self.u_mean_b = nn.Parameter(torch.zeros(R, U))
        self.u_logstd_w = nn.Parameter(torch.empty(R, U, H_ctrl))
        self.u_logstd_b = nn.Parameter(torch.zeros(R, U))
        self.u_to_lat_w = nn.Parameter(torch.empty(R, L, U))
        self.u_to_lat_b = nn.Parameter(torch.zeros(R, L)) if config.use_hidden_input_map else None
        _init_batched_xavier(self.u_mean_w)
        _init_batched_xavier(self.u_logstd_w)
        _init_batched_xavier(self.u_to_lat_w)

        # Flow MLPs (batched over regions)
        self.prior_flow = BatchedFlowMLP(R, L, 0, list(config.prior_hidden_sizes))
        post_cond_dim = H_enc + K + L
        self.post_flow = BatchedFlowMLP(R, L, post_cond_dim, list(config.posterior_hidden_sizes))

        # Process noise
        self.log_process_std = nn.Parameter(torch.log(torch.full((R, L), float(config.process_noise_init))))

        # Batched edges
        active = [(s, t) for s in range(R) for t in range(R) if s != t and edge_mask[s][t] != 0.0]
        self.batched_edges = BatchedEdges(active, R, N, L, config)

    @classmethod
    def from_dataset(cls, dataset, config: ModelConfig) -> 'NeuroVoltron':
        return cls(dataset.num_neurons.tolist(), dataset.input_dims.tolist(), dataset.latent_sizes.tolist(), config)

    def _time_basis(self, T: int, device: torch.device) -> torch.Tensor:
        if not hasattr(self, '_tb_cache'):
            self._tb_cache: dict[tuple[int, str], torch.Tensor] = {}
        key = (T, str(device))
        if key not in self._tb_cache:
            basis = raised_cosine_basis(self.config.baseline_num_basis_within, T, warp='asinh')
            self._tb_cache[key] = torch.as_tensor(basis, dtype=torch.float32, device=device)
        return self._tb_cache[key]

    def _pack_region(self, vals: list[torch.Tensor], max_dim: int) -> torch.Tensor:
        B = vals[0].shape[0]
        out = vals[0].new_zeros(B, self.n_regions, max_dim)
        for r, v in enumerate(vals):
            out[:, r, : v.shape[-1]] = v
        return out

    def forward(self, batch: dict[str, torch.Tensor], deterministic: bool = False) -> ForwardOutputs:
        spikes = batch['spikes']
        known_inputs = batch['known_inputs']
        history = batch['history']
        lengths = batch['lengths']
        history_lengths = batch['history_lengths']
        baseline_prefit = batch['baseline_prefit']
        B, _, T, _ = spikes.shape
        device = spikes.device
        R, L, N = self.n_regions, self.max_latent, self.max_neurons
        K, U, M = self.max_input, self.max_hidden_input, self.config.message_dim
        H_enc = 2 * self.config.encoder_hidden_size
        H_ctrl = self.config.controller_hidden_size

        time_basis = self._time_basis(T, device)
        tau = torch.as_tensor(self.config.tau, dtype=torch.float32, device=device).clamp_min(1e-6)
        alpha = (batch['dt'] / tau).view(1, R, 1)
        sqrt_alpha = torch.sqrt(alpha)

        # === Pre-loop: per-region encoding (fused bidir GRU, fast) ===
        enc_list, hist_list, base_list = [], [], []
        z0_mean_list, z0_logstd_list, z0_sample_list, z_init_list = [], [], [], []

        for r, region in enumerate(self.regions):
            n, k = self.num_neurons[r], self.input_dims[r]
            obs_r = spikes[:, r, :, :n]
            known_r = known_inputs[:, r, :, :k] if k > 0 else obs_r.new_zeros(B, T, 0)
            enc_r = region.encode_sequence(obs_r, known_r, lengths)
            hist_r_sum = region.encode_history(history[:, r, :, :n], history_lengths)
            z0_m, z0_ls, z0_s = region.initial_state(
                hist_r_sum, deterministic=(deterministic or not self.config.use_initial_condition))
            z_init = z0_s.new_zeros(B, self.latent_sizes[r]) if not self.config.use_initial_condition else z0_s
            base_r = region.baseline(hist_r_sum, baseline_prefit[:, r, :, :n], time_basis)
            enc_list.append(enc_r)
            hist_list.append(hist_r_sum)
            base_list.append(base_r)
            z0_mean_list.append(z0_m)
            z0_logstd_list.append(z0_ls)
            z0_sample_list.append(z0_s)
            z_init_list.append(z_init)

        # Stack to packed region tensors
        enc_all = torch.stack(enc_list, dim=1)                          # [B, R, T, H_enc]
        base_all = spikes.new_zeros(B, R, T, N)
        z = spikes.new_zeros(B, R, L)
        for r in range(R):
            base_all[:, r, :, :self.num_neurons[r]] = base_list[r]
            z[:, r, :self.latent_sizes[r]] = z_init_list[r]

        source = torch.einsum('brl,rnl->brn', z, self.readout_w) * self.neu_dim_mask
        if self.config.message_source == 'task_rate':
            source = F.softplus(source)
        ctrl_h = spikes.new_zeros(B, R, H_ctrl)
        pstd = (self.config.process_noise_floor + torch.exp(self.log_process_std)).unsqueeze(0)  # [1, R, L]

        known_all = known_inputs[:, :, :, :K] if K > 0 else spikes.new_zeros(B, R, T, 0)

        # Pre-allocate outputs
        z_out = spikes.new_zeros(B, R, T, L)
        tl_out = spikes.new_zeros(B, R, T, N)
        fl_out = spikes.new_zeros(B, R, T, N)
        rt_out = spikes.new_zeros(B, R, T, N)
        tr_out = spikes.new_zeros(B, R, T, N)
        pr_out = spikes.new_zeros(B, R, T, L)
        po_out = spikes.new_zeros(B, R, T, L)
        um_out = spikes.new_zeros(B, R, T, U)
        ul_out = spikes.new_zeros(B, R, T, U)
        us_out = spikes.new_zeros(B, R, T, U)
        hd_out = spikes.new_zeros(B, R, T, L)
        ma_out = spikes.new_zeros(B, R, T, L)
        mg_out = spikes.new_zeros(B, R, T, L)
        mm_out = spikes.new_zeros(B, R, R, T, M)
        ml_out = spikes.new_zeros(B, R, R, T, M)
        ms_out = spikes.new_zeros(B, R, R, T, M)

        lat_mask = self.lat_dim_mask                                    # [R, L]
        neu_mask = self.neu_dim_mask                                    # [R, N]

        # === Tensorized time loop ===
        for t in range(T):
            valid = (t < lengths).float().view(B, 1, 1)

            # 1. All edge messages in one call
            inc_add, inc_gain, msg_m, msg_ls, msg_s = self.batched_edges(source, deterministic)
            mm_out[:, :, :, t] = msg_m
            ml_out[:, :, :, t] = msg_ls
            ms_out[:, :, :, t] = msg_s

            enc_t = enc_all[:, :, t, :]                                 # [B, R, H_enc]

            # 2. Batched controller + hidden input (all regions at once)
            if self.config.use_hidden_inputs:
                ctrl_in = torch.cat([enc_t, z], dim=-1)
                ctrl_prop = _batched_gru_step(ctrl_in, ctrl_h, self.ctrl_wih, self.ctrl_whh, self.ctrl_bih, self.ctrl_bhh)
                ctrl_h = valid * ctrl_prop + (1.0 - valid) * ctrl_h
                u_mean = torch.einsum('brh,ruh->bru', ctrl_prop, self.u_mean_w) + self.u_mean_b
                u_logstd = torch.einsum('brh,ruh->bru', ctrl_prop, self.u_logstd_w) + self.u_logstd_b
                u_samp = u_mean if deterministic else u_mean + F.softplus(u_logstd) * torch.randn_like(u_mean)
                hidden_drive = torch.einsum('bru,rlu->brl', u_samp, self.u_to_lat_w)
                if self.u_to_lat_b is not None:
                    hidden_drive = hidden_drive + self.u_to_lat_b
            else:
                u_mean = spikes.new_zeros(B, R, U)
                u_logstd = spikes.new_zeros(B, R, U)
                u_samp = spikes.new_zeros(B, R, U)
                hidden_drive = spikes.new_zeros(B, R, L)

            # 3. Batched known-input drive (all regions at once)
            known_t = known_all[:, :, t, :]
            if self.known_w is not None and K > 0:
                kd = torch.einsum('brk,rlk->brl', known_t, self.known_w)
                if self.known_b is not None:
                    kd = kd + self.known_b
            else:
                kd = spikes.new_zeros(B, R, L)

            # 4. Batched prior + posterior flow (all regions at once)
            prior = self.prior_flow(z) + kd + inc_add
            if self.config.coupling_mode == 'additive_gain':
                prior = prior + torch.tanh(inc_gain) * z

            post_cond = torch.cat([enc_t, known_t, inc_add], dim=-1)
            post = prior + self.config.posterior_correction_scale * self.post_flow(z, post_cond)

            # 5. State update (all regions at once, one kernel)
            eps = torch.randn_like(z) if not deterministic else torch.zeros_like(z)
            z_prop = z + alpha * (post + hidden_drive) + sqrt_alpha * pstd * eps
            z = (valid * z_prop + (1.0 - valid) * z) * lat_mask

            # 6. Batched readout (all regions at once, one einsum)
            task_log = torch.einsum('brl,rnl->brn', z, self.readout_w) * neu_mask
            base_t = base_all[:, :, t, :]
            full_log = task_log + base_t
            rate = F.softplus(full_log)
            task_rate = F.softplus(task_log)
            source = task_rate if self.config.message_source == 'task_rate' else task_log

            # 7. Store
            z_out[:, :, t] = z
            tl_out[:, :, t] = task_log
            fl_out[:, :, t] = full_log
            rt_out[:, :, t] = rate
            tr_out[:, :, t] = task_rate
            pr_out[:, :, t] = prior
            po_out[:, :, t] = post
            um_out[:, :, t] = u_mean
            ul_out[:, :, t] = u_logstd
            us_out[:, :, t] = u_samp
            hd_out[:, :, t] = hidden_drive
            ma_out[:, :, t] = inc_add
            mg_out[:, :, t] = inc_gain

        return ForwardOutputs(
            z=z_out, task_logrates=tl_out, full_logrates=fl_out,
            rates=rt_out, task_rates=tr_out, baseline=base_all,
            prior_drift=pr_out, post_drift=po_out,
            u_mean=um_out, u_logstd=ul_out, u_sample=us_out,
            hidden_drive=hd_out,
            msg_mean=mm_out, msg_logstd=ml_out, msg_sample=ms_out,
            msg_add=ma_out, msg_gain=mg_out,
            z0_mean=self._pack_region(z0_mean_list, L),
            z0_logstd=self._pack_region(z0_logstd_list, L),
            z0_sample=self._pack_region(z0_sample_list, L),
            history_summary=self._pack_region(hist_list, self.max_hist),
            process_std=self.config.process_noise_floor + torch.exp(self.log_process_std),
        )


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

_cosine_basis_cache: dict[tuple[int, int], np.ndarray] = {}


def _cached_cosine_basis(num_basis: int, T: int) -> np.ndarray:
    key = (num_basis, T)
    if key not in _cosine_basis_cache:
        _cosine_basis_cache[key] = raised_cosine_basis(num_basis, T, warp='asinh')
    return _cosine_basis_cache[key]


def diag_gaussian_kl(mean: torch.Tensor, logstd_raw: torch.Tensor, prior_std: float) -> torch.Tensor:
    std = F.softplus(logstd_raw)
    sigma2 = float(prior_std) ** 2
    return 0.5 * (std.pow(2) / sigma2 + mean.pow(2) / sigma2 - 1.0 + np.log(sigma2) - torch.log(std.pow(2) + 1.0e-8))


def regularization_schedule(epoch: int, loss_cfg) -> dict[str, float]:
    def linear_ramp(start: int, span: int) -> float:
        if span <= 0:
            return 1.0
        return float(np.clip((epoch + 1 - start) / span, 0.0, 1.0))

    def exp_ramp(start: int, rate: float) -> float:
        if rate <= 0:
            return 1.0
        t = max(0, epoch - start)
        return float(1.0 - rate ** t)

    def _start(explicit: int, fallback: int) -> int:
        return explicit if explicit >= 0 else fallback

    floor = float(loss_cfg.kl_floor) if loss_cfg.use_kl_floor else 0.0
    w = loss_cfg.warmup_epochs
    n_cyc = getattr(loss_cfg, 'n_cycles', 0)
    inc_rate = getattr(loss_cfg, 'beta_inc_rate', 0.0)

    if n_cyc > 0:
        # --- Cyclical mode for message + hidden ---
        cyc_len = getattr(loss_cfg, 'cycle_length', 200)
        ramp_frac = getattr(loss_cfg, 'ramp_fraction', 0.5)
        ramp_len = int(cyc_len * ramp_frac)

        cycle_idx = epoch // cyc_len
        epoch_in_cycle = epoch % cyc_len

        if epoch_in_cycle < ramp_len:
            cyc_ramp = float(epoch_in_cycle + 1) / ramp_len
        else:
            cyc_ramp = 1.0

        msg_r = max(floor, cyc_ramp)
        hidden_r = max(floor, cyc_ramp)

        # --- Monotonic exponential for traj + z0 ---
        t_start = _start(getattr(loss_cfg, 'traj_ramp_start', -1), w + loss_cfg.message_ramp_epochs)
        z_start = _start(getattr(loss_cfg, 'z0_ramp_start', -1), w)

        if inc_rate > 0:
            traj_r = max(floor, exp_ramp(t_start, inc_rate))
            z0_r = max(floor, exp_ramp(z_start, inc_rate))
        else:
            traj_r = max(floor, linear_ramp(t_start, loss_cfg.traj_ramp_epochs))
            z0_r = max(floor, linear_ramp(z_start, loss_cfg.z0_ramp_epochs))

        return {
            'beta_hidden': loss_cfg.beta_hidden_final * hidden_r,
            'beta_message': loss_cfg.beta_message_final * msg_r,
            'beta_traj': loss_cfg.beta_traj_final * traj_r,
            'beta_z0': loss_cfg.beta_z0_final * z0_r,
            'beta_baseline': loss_cfg.beta_baseline,
            '_cycle': cycle_idx,
            '_phase': 'ramp' if epoch_in_cycle < ramp_len else 'hold',
        }

    # --- Legacy linear/exponential mode (no cycling) ---
    h_start = _start(getattr(loss_cfg, 'hidden_ramp_start', -1), w)
    m_start = _start(getattr(loss_cfg, 'message_ramp_start', -1), w)
    t_start = _start(getattr(loss_cfg, 'traj_ramp_start', -1), w + loss_cfg.message_ramp_epochs)
    z_start = _start(getattr(loss_cfg, 'z0_ramp_start', -1), w)

    if inc_rate > 0:
        hidden_r = max(floor, exp_ramp(h_start, inc_rate))
        msg_r = max(floor, exp_ramp(m_start, inc_rate))
        traj_r = max(floor, exp_ramp(t_start, inc_rate))
        z0_r = max(floor, exp_ramp(z_start, inc_rate))
    else:
        hidden_r = max(floor, linear_ramp(h_start, loss_cfg.hidden_ramp_epochs))
        msg_r = max(floor, linear_ramp(m_start, loss_cfg.message_ramp_epochs))
        traj_r = max(floor, linear_ramp(t_start, loss_cfg.traj_ramp_epochs))
        z0_r = max(floor, linear_ramp(z_start, loss_cfg.z0_ramp_epochs))

    return {
        'beta_hidden': loss_cfg.beta_hidden_final * hidden_r,
        'beta_message': loss_cfg.beta_message_final * msg_r,
        'beta_traj': loss_cfg.beta_traj_final * traj_r,
        'beta_z0': loss_cfg.beta_z0_final * z0_r,
        'beta_baseline': loss_cfg.beta_baseline,
    }


def _apply_free_bits(kl_per_element: torch.Tensor, mask: torch.Tensor, free_bits: float) -> torch.Tensor:
    """Apply free-bits: per-dimension KL below threshold incurs no gradient."""
    if free_bits <= 0.0:
        return (kl_per_element * mask).sum() / mask.sum().clamp_min(1.0)
    # Clamp each element from below so dims using < free_bits nats get zero gradient
    clamped = torch.clamp(kl_per_element, min=free_bits)
    return (clamped * mask).sum() / mask.sum().clamp_min(1.0)


def compute_loss(
    outputs: ForwardOutputs,
    batch: dict[str, torch.Tensor],
    model: NeuroVoltron,
    loss_cfg,
    epoch: int,
) -> tuple[torch.Tensor, dict[str, float | torch.Tensor]]:
    spikes = batch['spikes']
    B, R, T, _ = spikes.shape
    device = spikes.device
    valid = (torch.arange(T, device=device).unsqueeze(0) < batch['lengths'].unsqueeze(1)).float()
    valid_brt = valid[:, None, :, None]
    neuron_mask = batch['neuron_mask'][None, :, None, :]
    latent_mask = batch['latent_mask'][None, :, None, :]
    dt = batch['dt']
    free_bits = getattr(loss_cfg, 'free_bits', 0.0)

    rates = torch.clamp(outputs.rates, min=1.0e-6)
    nll = dt * rates - spikes * torch.log(dt * rates) + torch.lgamma(spikes + 1.0)
    nll = (nll * valid_brt * neuron_mask).sum() / (valid_brt * neuron_mask).sum().clamp_min(1.0)

    alpha = dt / torch.as_tensor(model.config.tau, dtype=torch.float32, device=device).view(1, R, 1, 1)
    sigma2 = outputs.process_std.clamp_min(1.0e-6).pow(2).view(1, R, 1, model.max_latent)
    # Phase 1 fix: when detach_prior_kl=True, compute traj KL from |correction|^2 only.
    # This ensures θ_prior gets ZERO gradient from the traj KL term —
    # the prior inside post_drift is also stopped (since correction = post - prior,
    # and we detach the entire prior contribution).
    if getattr(loss_cfg, 'detach_prior_kl', False):
        correction = outputs.post_drift - outputs.prior_drift.detach()
        traj = 0.5 * alpha * correction.pow(2) / sigma2
    else:
        traj = 0.5 * alpha * (outputs.post_drift - outputs.prior_drift).pow(2) / sigma2
    traj = _apply_free_bits(traj, valid_brt * latent_mask, free_bits)

    if model.config.use_hidden_inputs:
        hkl = diag_gaussian_kl(outputs.u_mean, outputs.u_logstd, model.config.sigma_input_prior)
        hidden_mask = valid[:, None, :, None]
        hidden_kl = _apply_free_bits(hkl, hidden_mask, free_bits)
    else:
        hidden_kl = spikes.new_tensor(0.0)

    if model.config.use_initial_condition:
        z0kl = diag_gaussian_kl(outputs.z0_mean, outputs.z0_logstd, model.config.z0_prior_std)
        z0_mask = batch['latent_mask'][None, :, :]
        z0_kl = _apply_free_bits(z0kl, z0_mask, free_bits)
    else:
        z0_kl = spikes.new_tensor(0.0)

    edge_mask_buf = model.edge_mask.view(1, R, R, 1, 1)
    valid_msg = valid[:, None, None, :, None] * edge_mask_buf
    if model.config.message_stochastic:
        mkl = diag_gaussian_kl(outputs.msg_mean, outputs.msg_logstd, model.config.sigma_message_prior)
        message_kl = _apply_free_bits(mkl, valid_msg, free_bits)
    else:
        message_kl = (0.5 * outputs.msg_sample.pow(2) / (model.config.sigma_message_prior ** 2) * valid_msg).sum() / valid_msg.sum().clamp_min(1.0)

    if model.config.baseline_mode == 2:
        reg = spikes.new_tensor(0.0)
        basis = torch.as_tensor(_cached_cosine_basis(model.config.baseline_num_basis_within, T), dtype=torch.float32, device=device)
        for region in model.regions:
            assert region.baseline_within is not None
            reg = reg + model.config.baseline_l2 * region.baseline_within.pow(2).mean()
            curve = basis @ region.baseline_within
            if curve.shape[0] >= 3:
                reg = reg + 0.1 * (curve[2:] - 2 * curve[1:-1] + curve[:-2]).pow(2).mean()
            if region.baseline_trial_proj is not None and region.baseline_trial_to_neuron is not None:
                reg = reg + 0.1 * region.baseline_trial_proj.weight.pow(2).mean()
        baseline_reg = reg / len(model.regions)
    else:
        baseline_reg = spikes.new_tensor(0.0)

    # Auxiliary message loss: messages should predict posterior correction
    # (the part of the drift that the prior alone cannot explain)
    aux_w = getattr(loss_cfg, 'aux_message_weight', 0.0)
    if aux_w > 0.0:
        # posterior correction = post_drift - prior_drift; message drive = msg_add
        # If messages carry useful info, msg_add should correlate with posterior correction
        post_correction = outputs.post_drift - outputs.prior_drift  # [B, R, T, L]
        msg_drive = outputs.msg_add                                  # [B, R, T, L]
        # Cosine similarity loss: encourage alignment (maximize cosine sim → minimize 1-cos)
        # Compute per-region per-timestep
        dot = (post_correction * msg_drive).sum(dim=-1, keepdim=True)          # [B, R, T, 1]
        norm_pc = post_correction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        norm_md = msg_drive.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        cosine_sim = dot / (norm_pc * norm_md)                                 # [B, R, T, 1]
        # Only apply where both signals are non-trivial
        active = (norm_pc.squeeze(-1) > 1e-4) & (norm_md.squeeze(-1) > 1e-4)  # [B, R, T]
        active = active.float() * valid[:, None, :]
        if active.sum() > 0:
            aux_loss = (1.0 - cosine_sim.squeeze(-1)) * active
            aux_loss = aux_loss.sum() / active.sum().clamp_min(1.0)
        else:
            aux_loss = spikes.new_tensor(0.0)
    else:
        aux_loss = spikes.new_tensor(0.0)

    # Phase 1: Prior-only rollout loss
    # From detached posterior z, roll forward k steps using only prior dynamics,
    # then score against detached posterior z targets.
    rollout_w = getattr(loss_cfg, 'rollout_loss_weight', 0.0)
    rollout_k = getattr(loss_cfg, 'rollout_steps', 8)
    if rollout_w > 0.0 and rollout_k > 0:
        z_post = outputs.z.detach()                          # [B, R, T, L]
        known_all = batch.get('known_inputs')                # [B, R, T, K]
        lat_mask_2d = batch['latent_mask']                   # [R, L]
        alpha_r = alpha.squeeze(-1)                          # [1, R, 1]  (dt/tau per region)
        base_all = outputs.baseline.detach()                 # [B, R, T, N]
        hd_all = outputs.hidden_drive.detach()               # [B, R, T, L]
        rollout_nll = spikes.new_tensor(0.0)
        n_rollout = 0
        # Pick random start points spread across the sequence
        max_start = T - rollout_k - 1
        if max_start > 0:
            stride = max(1, max_start // 4)
            starts = list(range(0, max_start, stride))[:4]   # up to 4 anchor points
            for t0 in starts:
                z_r = z_post[:, :, t0]                       # [B, R, L]
                for j in range(rollout_k):
                    t_cur = t0 + j
                    # Known input drive
                    if model.known_w is not None and known_all is not None:
                        known_t = known_all[:, :, t_cur]
                        kd = torch.einsum('brk,rlk->brl', known_t, model.known_w)
                        if model.known_b is not None:
                            kd = kd + model.known_b
                    else:
                        kd = spikes.new_zeros(B, R, model.max_latent)
                    # Use detached posterior messages and hidden inputs
                    inc_add_t = outputs.msg_add[:, :, t_cur].detach()
                    hd_t = hd_all[:, :, t_cur]               # [B, R, L]
                    # Prior flow + hidden inputs (no posterior correction)
                    prior_drift = model.prior_flow(z_r) + kd + inc_add_t
                    if model.config.coupling_mode == 'additive_gain':
                        inc_gain_t = outputs.msg_gain[:, :, t_cur].detach()
                        prior_drift = prior_drift + torch.tanh(inc_gain_t) * z_r
                    # Euler step (include hidden drive so MSE tests dynamics only)
                    z_r = z_r + alpha_r * (prior_drift + hd_t)
                    z_r = z_r * lat_mask_2d.unsqueeze(0)
                    # Score against observed spikes via Poisson NLL
                    t_next = t_cur + 1
                    v = (t_next < batch['lengths']).float().view(B, 1, 1)   # [B, 1, 1]
                    task_log_r = torch.einsum('brl,rnl->brn', z_r, model.readout_w) * model.neu_dim_mask
                    base_t = base_all[:, :, t_next]                         # [B, R, N]
                    full_log_r = task_log_r + base_t
                    rate_r = torch.clamp(F.softplus(full_log_r), min=1e-6)
                    spk_t = spikes[:, :, t_next]                            # [B, R, N]
                    step_nll = dt * rate_r - spk_t * torch.log(dt * rate_r + 1e-8)
                    step_nll = (step_nll * v.unsqueeze(-1) * neuron_mask.squeeze(-2)).sum()
                    rollout_nll = rollout_nll + step_nll
                    n_rollout += (v.unsqueeze(-1) * neuron_mask.squeeze(-2)).sum().item()
            if n_rollout > 0:
                rollout_nll = rollout_nll / max(n_rollout, 1.0)
    else:
        rollout_nll = spikes.new_tensor(0.0)

    weights = regularization_schedule(epoch, loss_cfg)
    total = (nll + weights['beta_traj'] * traj + weights['beta_hidden'] * hidden_kl
             + weights['beta_z0'] * z0_kl + weights['beta_message'] * message_kl
             + weights['beta_baseline'] * baseline_reg + aux_w * aux_loss
             + rollout_w * rollout_nll)
    metrics = {
        'loss': total, 'nll': nll, 'traj_kl': traj, 'hidden_kl': hidden_kl,
        'z0_kl': z0_kl, 'message_kl': message_kl, 'baseline_reg': baseline_reg,
        'aux_loss': aux_loss, 'rollout_nll': rollout_nll,
        **weights,
    }
    return total, metrics
