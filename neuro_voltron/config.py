from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal
import json


VariantFamily = Literal['A', 'B', 'C']
MessageSource = Literal['task_rate', 'task_lograte']
MessageTiming = Literal['lagged']
CouplingMode = Literal['additive', 'additive_gain']
BaselineMode = Literal[0, 1, 2, 3]


@dataclass
class ModelConfig:
    variant_family: VariantFamily = 'A'
    baseline_mode: BaselineMode = 2
    latent_sizes: list[int] = field(default_factory=lambda: [2, 2, 2])
    tau: list[float] = field(default_factory=lambda: [0.12, 0.10, 0.11])
    encoder_hidden_size: int = 64
    history_hidden_size: int = 32
    controller_hidden_size: int = 64
    history_embed_size: int = 64
    prior_hidden_sizes: list[int] = field(default_factory=lambda: [64, 64])
    posterior_hidden_sizes: list[int] = field(default_factory=lambda: [64, 64])
    message_dim: int = 4
    hidden_input_dim: int = 4
    z0_prior_std: float = 1.0
    sigma_input_prior: float = 1.0
    sigma_message_prior: float = 1.0
    process_noise_floor: float = 1.0e-3
    process_noise_init: float = 0.03
    message_source: MessageSource = 'task_rate'
    message_timing: MessageTiming = 'lagged'
    message_lag: int = 1
    message_stochastic: bool = True
    use_hidden_inputs: bool = True
    use_initial_condition: bool = True
    coupling_mode: CouplingMode = 'additive'
    baseline_num_basis_within: int = 6
    baseline_trial_rank: int = 2
    baseline_l2: float = 1.0e-4
    baseline_prefit_num_basis_within: int = 6
    baseline_prefit_num_basis_across: int = 5
    baseline_prefit_ridge: float = 1.0e-3
    baseline_prefit_clip_hz: float = 150.0
    use_message_bias: bool = True
    use_known_input_bias: bool = True
    use_hidden_input_map: bool = True
    posterior_correction_scale: float = 1.0
    intrinsic_weight_scale: float = 1.0
    edge_mask: list[list[float]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LossConfig:
    beta_traj_final: float = 0.5
    beta_message_final: float = 0.02
    beta_hidden_final: float = 0.2
    beta_z0_final: float = 0.1
    beta_baseline: float = 1.0
    beta_message_energy: float = 1.0
    warmup_epochs: int = 5
    message_ramp_epochs: int = 20
    hidden_ramp_epochs: int = 20
    traj_ramp_epochs: int = 20
    z0_ramp_epochs: int = 20
    hidden_ramp_start: int = -1
    message_ramp_start: int = -1
    traj_ramp_start: int = -1
    z0_ramp_start: int = -1
    use_kl_floor: bool = True
    kl_floor: float = 1.0e-3
    # Cyclical annealing (for message + hidden betas)
    n_cycles: int = 0              # 0 = disabled (use linear/exp ramp); >0 = cyclical
    cycle_length: int = 200        # epochs per cycle
    ramp_fraction: float = 0.5     # fraction of cycle spent ramping (rest is hold)
    # Exponential ramp (for traj + z0 betas when cyclical is on)
    beta_inc_rate: float = 0.0     # 0 = linear ramp; >0 = exponential (e.g. 0.995)
    min_checkpoint_epoch: int = 0  # don't save best checkpoint before this epoch
    # Free bits: per-dimension KL floor (nats). KL below this value incurs no gradient.
    free_bits: float = 0.0         # 0 = disabled; typical values: 0.1-1.0
    # Auxiliary message reconstruction loss weight
    aux_message_weight: float = 0.0  # 0 = disabled; >0 = adds trajectory-predictability loss
    # Phase 1: Prevent prior from tracking posterior
    detach_prior_kl: bool = False    # True = θ_prior gets no gradient from traj_kl
    # Phase 1: Prior-only rollout loss (tests free-running prediction)
    rollout_loss_weight: float = 0.0  # 0 = disabled; >0 = adds prior rollout MSE
    rollout_steps: int = 8            # number of steps to roll prior forward

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OptimConfig:
    learning_rate: float = 2.0e-3
    weight_decay: float = 1.0e-5
    grad_clip_norm: float = 5.0
    batch_size: int = 32
    epochs: int = 80
    seed: int = 0
    log_every: int = 10
    eval_every: int = 10
    lr_schedule: str = 'constant'  # 'constant', 'cosine', 'plateau', or 'sgdr'
    lr_min_factor: float = 0.05   # minimum LR as fraction of initial (for cosine)
    lr_patience: int = 6          # epochs without improvement before LR drop (plateau)
    lr_decay: float = 0.95        # multiplicative LR decay factor (plateau)
    lr_min: float = 1.0e-5        # absolute minimum LR (plateau/sgdr)
    lr_warmup_epochs: int = 10    # linear warmup at start of each SGDR cycle
    lr_cycle_epochs: int = 200    # initial SGDR cycle length (T_0)
    lr_cycle_mult: int = 1        # SGDR cycle multiplier (T_mult)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DataConfig:
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    shuffle_seed: int = 0
    broadcast_single_input: bool = False
    region_input_dims: list[int] | None = None
    synthetic_history_bins: int = 20

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    data: DataConfig = field(default_factory=DataConfig)
    name: str = 'A2'

    def to_dict(self) -> dict[str, Any]:
        return {
            'name': self.name,
            'model': self.model.to_dict(),
            'loss': self.loss.to_dict(),
            'optim': self.optim.to_dict(),
            'data': self.data.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> 'ExperimentConfig':
        return cls(
            name=payload.get('name', 'unnamed'),
            model=ModelConfig(**payload.get('model', {})),
            loss=LossConfig(**payload.get('loss', {})),
            optim=OptimConfig(**payload.get('optim', {})),
            data=DataConfig(**payload.get('data', {})),
        )

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))


def _default_edge_mask(n_regions: int) -> list[list[float]]:
    return [[0.0 if i == j else 1.0 for j in range(n_regions)] for i in range(n_regions)]


_VARIANT_SPECS: dict[str, dict[str, Any]] = {
    'A0': {
        'model': {
            'variant_family': 'A',
            'baseline_mode': 0,
            'message_stochastic': True,
            'use_hidden_inputs': True,
            'use_initial_condition': True,
            'coupling_mode': 'additive',
            'message_source': 'task_rate',
        },
        'loss': {
            'beta_traj_final': 0.4,
            'beta_message_final': 0.02,
            'beta_hidden_final': 0.2,
            'beta_z0_final': 0.08,
        },
    },
    'A1': {
        'model': {
            'variant_family': 'A',
            'baseline_mode': 1,
            'message_stochastic': True,
            'use_hidden_inputs': True,
            'use_initial_condition': True,
            'coupling_mode': 'additive',
            'message_source': 'task_rate',
        },
        'loss': {
            'beta_traj_final': 0.45,
            'beta_message_final': 0.02,
            'beta_hidden_final': 0.2,
            'beta_z0_final': 0.08,
        },
    },
    'A2': {
        'model': {
            'variant_family': 'A',
            'baseline_mode': 2,
            'message_stochastic': True,
            'use_hidden_inputs': True,
            'use_initial_condition': True,
            'coupling_mode': 'additive',
            'message_source': 'task_rate',
            'baseline_num_basis_within': 6,
            'baseline_trial_rank': 2,
        },
        'loss': {
            'beta_traj_final': 0.5,
            'beta_message_final': 0.02,
            'beta_hidden_final': 0.2,
            'beta_z0_final': 0.1,
            'beta_baseline': 1.0,
        },
    },
    'B1': {
        'model': {
            'variant_family': 'B',
            'baseline_mode': 1,
            'message_stochastic': False,
            'use_hidden_inputs': True,
            'use_initial_condition': True,
            'coupling_mode': 'additive',
            'message_source': 'task_rate',
            'prior_hidden_sizes': [48],
            'posterior_hidden_sizes': [48],
        },
        'loss': {
            'beta_traj_final': 0.6,
            'beta_message_final': 0.02,
            'beta_hidden_final': 0.5,
            'beta_z0_final': 0.12,
        },
    },
    'B2': {
        'model': {
            'variant_family': 'B',
            'baseline_mode': 2,
            'message_stochastic': False,
            'use_hidden_inputs': True,
            'use_initial_condition': True,
            'coupling_mode': 'additive',
            'message_source': 'task_rate',
            'prior_hidden_sizes': [48],
            'posterior_hidden_sizes': [48],
            'baseline_num_basis_within': 6,
            'baseline_trial_rank': 1,
        },
        'loss': {
            'beta_traj_final': 0.65,
            'beta_message_final': 0.02,
            'beta_hidden_final': 0.6,
            'beta_z0_final': 0.12,
            'beta_baseline': 1.0,
        },
    },
    'B3': {
        'model': {
            'variant_family': 'B',
            'baseline_mode': 3,
            'message_stochastic': False,
            'use_hidden_inputs': True,
            'use_initial_condition': True,
            'coupling_mode': 'additive',
            'message_source': 'task_rate',
            'prior_hidden_sizes': [48],
            'posterior_hidden_sizes': [48],
            'baseline_prefit_num_basis_within': 6,
            'baseline_prefit_num_basis_across': 5,
        },
        'loss': {
            'beta_traj_final': 0.65,
            'beta_message_final': 0.02,
            'beta_hidden_final': 0.6,
            'beta_z0_final': 0.12,
            'beta_baseline': 0.0,
        },
    },
    'C1': {
        'model': {
            'variant_family': 'C',
            'baseline_mode': 1,
            'message_stochastic': True,
            'use_hidden_inputs': True,
            'use_initial_condition': True,
            'coupling_mode': 'additive_gain',
            'message_source': 'task_rate',
        },
        'loss': {
            'beta_traj_final': 0.5,
            'beta_message_final': 0.02,
            'beta_hidden_final': 0.2,
            'beta_z0_final': 0.1,
        },
    },
    'C2': {
        'model': {
            'variant_family': 'C',
            'baseline_mode': 2,
            'message_stochastic': True,
            'use_hidden_inputs': True,
            'use_initial_condition': True,
            'coupling_mode': 'additive_gain',
            'message_source': 'task_rate',
            'baseline_num_basis_within': 6,
            'baseline_trial_rank': 2,
        },
        'loss': {
            'beta_traj_final': 0.55,
            'beta_message_final': 0.02,
            'beta_hidden_final': 0.2,
            'beta_z0_final': 0.1,
            'beta_baseline': 1.0,
        },
    },
}


def build_variant_config(
    variant_name: str,
    *,
    latent_sizes: list[int] | None = None,
    edge_mask: list[list[float]] | None = None,
    small: bool = False,
) -> ExperimentConfig:
    if variant_name not in _VARIANT_SPECS:
        raise KeyError(f'Unknown variant {variant_name}. Available: {sorted(_VARIANT_SPECS)}')
    cfg = ExperimentConfig(name=variant_name)
    if latent_sizes is not None:
        cfg.model.latent_sizes = list(latent_sizes)
    if edge_mask is None:
        cfg.model.edge_mask = _default_edge_mask(len(cfg.model.latent_sizes))
    else:
        cfg.model.edge_mask = edge_mask
    spec = _VARIANT_SPECS[variant_name]
    for section_name, patch in spec.items():
        section = getattr(cfg, section_name)
        for key, value in patch.items():
            setattr(section, key, value)
    if small:
        cfg.model.encoder_hidden_size = 24
        cfg.model.history_hidden_size = 16
        cfg.model.controller_hidden_size = 24
        cfg.model.history_embed_size = 32
        cfg.model.prior_hidden_sizes = [24]
        cfg.model.posterior_hidden_sizes = [24]
        cfg.model.message_dim = 2
        cfg.model.hidden_input_dim = 2
        cfg.optim.batch_size = 8
        cfg.optim.epochs = 3
        cfg.optim.log_every = 1
        cfg.optim.eval_every = 1
        cfg.loss.warmup_epochs = 0
        cfg.loss.message_ramp_epochs = 1
        cfg.loss.hidden_ramp_epochs = 1
        cfg.loss.traj_ramp_epochs = 1
        cfg.loss.z0_ramp_epochs = 1
    return cfg


ALL_VARIANTS = tuple(_VARIANT_SPECS.keys())
