from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import copy
import json
import pickle
import sys

import numpy as np
import torch

from .config import ExperimentConfig
from .data import DatasetSplits
from .evaluate import effectome_cosine_similarity, effectome_from_messages, extract_ground_truth_effectome, message_r2_scores
from .model import ForwardOutputs, NeuroVoltron, compute_loss

Array = Any


@dataclass
class FitResults:
    state_dict: dict[str, Any]
    history: dict[str, list[float]]
    metrics: dict[str, Any]
    last_outputs: dict[str, np.ndarray]


_BATCH_KEYS = {'spikes', 'known_inputs', 'history', 'lengths', 'times', 'history_lengths', 'baseline_prefit'}


def _prepare_tensor_dataset(dataset, device: torch.device | None = None) -> dict[str, torch.Tensor]:
    tensors = dataset.to_torch(device=None)
    if device is not None:
        for key in tensors:
            if key not in _BATCH_KEYS:
                tensors[key] = tensors[key].to(device)
    return tensors


def _slice_batch(tensors: dict[str, torch.Tensor], idx: np.ndarray | list[int], device: torch.device) -> dict[str, torch.Tensor]:
    idx_t = torch.as_tensor(idx, dtype=torch.long)
    batch: dict[str, torch.Tensor] = {}
    for key, val in tensors.items():
        if key in _BATCH_KEYS:
            batch[key] = val.index_select(0, idx_t).to(device)
        else:
            batch[key] = val.to(device) if val.device != device else val
    return batch


def _epoch_iterator(n_items: int, batch_size: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_items)
    return [perm[i : i + batch_size] for i in range(0, n_items, batch_size)]


def _mean_metric_dict(metric_dicts: list[dict[str, float]]) -> dict[str, float]:
    if not metric_dicts:
        return {}
    keys = metric_dicts[0].keys()
    out = {}
    for k in keys:
        vals = [m[k] for m in metric_dicts]
        if isinstance(vals[0], str):
            out[k] = vals[-1]
        else:
            out[k] = float(np.mean(vals))
    return out


def _detach_metrics(metrics: dict[str, float | torch.Tensor]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, val in metrics.items():
        if isinstance(val, str):
            out[key] = val
        elif isinstance(val, torch.Tensor):
            out[key] = float(val.detach().cpu().item())
        else:
            out[key] = float(val)
    return out


def _outputs_to_numpy(outputs: ForwardOutputs) -> dict[str, np.ndarray]:
    return {
        'z': outputs.z.detach().cpu().numpy(),
        'task_logrates': outputs.task_logrates.detach().cpu().numpy(),
        'full_logrates': outputs.full_logrates.detach().cpu().numpy(),
        'rates': outputs.rates.detach().cpu().numpy(),
        'task_rates': outputs.task_rates.detach().cpu().numpy(),
        'baseline': outputs.baseline.detach().cpu().numpy(),
        'prior_drift': outputs.prior_drift.detach().cpu().numpy(),
        'post_drift': outputs.post_drift.detach().cpu().numpy(),
        'u_mean': outputs.u_mean.detach().cpu().numpy(),
        'u_logstd': outputs.u_logstd.detach().cpu().numpy(),
        'u_sample': outputs.u_sample.detach().cpu().numpy(),
        'hidden_drive': outputs.hidden_drive.detach().cpu().numpy(),
        'msg_mean': outputs.msg_mean.detach().cpu().numpy(),
        'msg_logstd': outputs.msg_logstd.detach().cpu().numpy(),
        'msg_sample': outputs.msg_sample.detach().cpu().numpy(),
        'msg_add': outputs.msg_add.detach().cpu().numpy(),
        'msg_gain': outputs.msg_gain.detach().cpu().numpy(),
        'z0_mean': outputs.z0_mean.detach().cpu().numpy(),
        'z0_logstd': outputs.z0_logstd.detach().cpu().numpy(),
        'z0_sample': outputs.z0_sample.detach().cpu().numpy(),
        'history_summary': outputs.history_summary.detach().cpu().numpy(),
        'process_std': outputs.process_std.detach().cpu().numpy(),
    }


def _evaluate_dataset(
    model: NeuroVoltron,
    tensors: dict[str, torch.Tensor],
    batch_size: int,
    epoch: int,
    loss_cfg,
    device: torch.device,
    *,
    collect_outputs: bool = False,
    forward_fn=None,
) -> tuple[dict[str, float], dict[str, np.ndarray] | None]:
    model.eval()
    n_items = int(tensors['spikes'].shape[0])
    metric_accum: dict[str, float] = {}
    total_weight = 0
    outputs_store: dict[str, list[np.ndarray]] | None = {} if collect_outputs else None

    _fwd = forward_fn if forward_fn is not None else model
    with torch.no_grad():
        for start in range(0, n_items, batch_size):
            idx = np.arange(start, min(start + batch_size, n_items), dtype=np.int64)
            batch = _slice_batch(tensors, idx, device)
            outputs = _fwd(batch, deterministic=True)
            _, metrics = compute_loss(outputs, batch, model, loss_cfg, epoch)
            metrics_np = _detach_metrics(metrics)
            weight = len(idx)
            total_weight += weight
            for key, value in metrics_np.items():
                if isinstance(value, str):
                    metric_accum[key] = value
                else:
                    metric_accum[key] = metric_accum.get(key, 0.0) + weight * value
            if collect_outputs:
                out_np = _outputs_to_numpy(outputs)
                assert outputs_store is not None
                for key, arr in out_np.items():
                    outputs_store.setdefault(key, []).append(arr)

    metric_mean = {}
    for k, v in metric_accum.items():
        if isinstance(v, str):
            metric_mean[k] = v
        else:
            metric_mean[k] = float(v / max(total_weight, 1))
    if not collect_outputs:
        return metric_mean, None

    assert outputs_store is not None
    concat = {k: np.concatenate(v, axis=0) for k, v in outputs_store.items()}
    return metric_mean, concat


def save_artifacts(workdir: str | Path, config: ExperimentConfig, results: FitResults, split_indices: dict[str, np.ndarray]) -> None:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    config.save_json(workdir / 'config_resolved.json')
    with open(workdir / 'history.json', 'w', encoding='utf-8') as f:
        json.dump(results.history, f, indent=2)
    with open(workdir / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(results.metrics, f, indent=2)
    with open(workdir / 'split_indices.json', 'w', encoding='utf-8') as f:
        json.dump({k: np.asarray(v).tolist() for k, v in split_indices.items()}, f, indent=2)
    try:
        torch.save(results.state_dict, workdir / 'model.pt')
    except Exception:
        torch.save(results.state_dict, workdir / 'model.pt', _use_new_zipfile_serialization=False)
    with open(workdir / 'state_dict.pkl', 'wb') as f:
        pickle.dump(results.state_dict, f)
    np.savez_compressed(workdir / 'posterior_means_full.npz', **results.last_outputs)
    if 'effectome' in results.metrics:
        np.save(workdir / 'effectome_full.npy', np.asarray(results.metrics['effectome'], dtype=np.float32))


from tqdm import tqdm

def fit_model(config: ExperimentConfig, splits: DatasetSplits, workdir: str | Path | None = None) -> FitResults:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Open a separate structured log file (tqdm progress bars clobber stdout lines)
    _log_fh = None
    if workdir is not None:
        Path(workdir).mkdir(parents=True, exist_ok=True)
        _log_fh = open(Path(workdir) / 'train.log', 'w', buffering=1)  # line-buffered
    def _log(msg: str) -> None:
        print(msg, flush=True)
        if _log_fh is not None:
            _log_fh.write(msg + '\n')
            _log_fh.flush()
    _log(f"Using device: {device}")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
        _log(f"GPU: {torch.cuda.get_device_name(device)}")
    train_tensors = _prepare_tensor_dataset(splits.train, device)
    val_tensors = _prepare_tensor_dataset(splits.val, device)
    test_tensors = _prepare_tensor_dataset(splits.test, device)
    full_tensors = _prepare_tensor_dataset(splits.full, device)

    torch.manual_seed(config.optim.seed)
    np.random.seed(config.optim.seed)

    model = NeuroVoltron.from_dataset(splits.full, config.model).to(device)
    compiled_model = model
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.optim.learning_rate, weight_decay=config.optim.weight_decay)

    scheduler = None
    plateau_scheduler = False
    if config.optim.lr_schedule == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.optim.epochs,
            eta_min=config.optim.learning_rate * config.optim.lr_min_factor)
    elif config.optim.lr_schedule == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=config.optim.lr_decay,
            patience=config.optim.lr_patience, min_lr=config.optim.lr_min)
        plateau_scheduler = True
    elif config.optim.lr_schedule == 'sgdr':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.optim.lr_cycle_epochs,
            T_mult=config.optim.lr_cycle_mult,
            eta_min=config.optim.lr_min)

    min_ckpt = getattr(config.loss, 'min_checkpoint_epoch', 0)

    history = {
        'train_loss': [], 'train_nll': [], 'train_traj_kl': [], 'train_hidden_kl': [], 'train_message_kl': [], 'train_z0_kl': [], 'train_baseline_reg': [], 'train_rollout_nll': [],
        'val_loss': [], 'val_nll': [], 'val_traj_kl': [], 'val_hidden_kl': [], 'val_message_kl': [], 'val_z0_kl': [], 'val_baseline_reg': [], 'val_rollout_nll': [],
        'beta_hidden': [], 'beta_message': [], 'beta_traj': [], 'beta_z0': [], 'lr': [],
        'grad_norm': [], 'cycle': [], 'phase': [],
    }

    best_val = float('inf')
    best_epoch = -1
    best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})

    n_train = int(train_tensors['spikes'].shape[0])
    _interactive = sys.stdout.isatty()
    epoch_pbar = tqdm(range(config.optim.epochs), desc="Training Epochs", unit="epoch", disable=not _interactive)
    for epoch in epoch_pbar:
        model.train()
        batch_metrics: list[dict[str, float]] = []
        epoch_grad_norm = 0.0
        n_batches = 0
        for idx in _epoch_iterator(n_train, config.optim.batch_size, config.optim.seed + epoch):
            batch = _slice_batch(train_tensors, idx, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = compiled_model(batch, deterministic=False)
            loss, metrics = compute_loss(outputs, batch, model, config.loss, epoch)
            loss.backward()
            if config.optim.grad_clip_norm > 0:
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.grad_clip_norm)
                epoch_grad_norm += float(gn)
            else:
                epoch_grad_norm += float(sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None))
            n_batches += 1
            optimizer.step()
            batch_metrics.append(_detach_metrics(metrics))

        avg_grad_norm = epoch_grad_norm / max(n_batches, 1)
        train_metrics = _mean_metric_dict(batch_metrics)
        val_metrics, _ = _evaluate_dataset(model, val_tensors, config.optim.batch_size, epoch, config.loss, device, collect_outputs=False, forward_fn=compiled_model)

        # Checkpoint on val NLL (not total loss) — total loss is meaningless during beta ramp
        ckpt_metric = val_metrics['nll']
        if epoch >= min_ckpt and ckpt_metric < best_val:
            best_val = ckpt_metric
            best_epoch = epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})

        # Extract cycle/phase info if available
        cycle_num = int(train_metrics.get('_cycle', -1))
        phase_str = str(train_metrics.get('_phase', ''))

        for prefix, metrics in [('train', train_metrics), ('val', val_metrics)]:
            history[f'{prefix}_loss'].append(metrics['loss'])
            history[f'{prefix}_nll'].append(metrics['nll'])
            history[f'{prefix}_traj_kl'].append(metrics['traj_kl'])
            history[f'{prefix}_hidden_kl'].append(metrics['hidden_kl'])
            history[f'{prefix}_message_kl'].append(metrics['message_kl'])
            history[f'{prefix}_z0_kl'].append(metrics['z0_kl'])
            history[f'{prefix}_baseline_reg'].append(metrics['baseline_reg'])
            history[f'{prefix}_rollout_nll'].append(metrics.get('rollout_nll', 0.0))
        history['beta_hidden'].append(train_metrics['beta_hidden'])
        history['beta_message'].append(train_metrics['beta_message'])
        history['beta_traj'].append(train_metrics['beta_traj'])
        history['beta_z0'].append(train_metrics['beta_z0'])
        history['lr'].append(optimizer.param_groups[0]['lr'])
        history['grad_norm'].append(avg_grad_norm)
        history['cycle'].append(cycle_num)
        history['phase'].append(phase_str)

        if scheduler is not None:
            if plateau_scheduler:
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()

        if (epoch + 1) % config.optim.log_every == 0 or epoch == 0 or epoch == config.optim.epochs - 1:
            cur_lr = optimizer.param_groups[0]['lr']
            beta_m = train_metrics['beta_message']
            beta_h = train_metrics['beta_hidden']
            beta_t = train_metrics['beta_traj']
            raw_msg = train_metrics['message_kl'] / max(beta_m, 1e-10) if beta_m > 1e-10 else train_metrics['message_kl']
            raw_hid = train_metrics['hidden_kl'] / max(beta_h, 1e-10) if beta_h > 1e-10 else train_metrics['hidden_kl']

            cyc_tag = f" [C{cycle_num+1} {phase_str}]" if cycle_num >= 0 else ""
            aux_val = train_metrics.get('aux_loss', 0.0)
            aux_tag = f" aux={aux_val:.3f}" if aux_val > 0 else ""
            roll_val = train_metrics.get('rollout_nll', 0.0)
            if isinstance(roll_val, torch.Tensor):
                roll_val = roll_val.item()
            roll_tag = f" roll={roll_val:.4f}" if roll_val > 0 else ""
            aux_tag = aux_tag + roll_tag
            epoch_pbar.set_postfix({'val_nll': f"{val_metrics['nll']:.4f}", 'train_nll': f"{train_metrics['nll']:.4f}"})
            log_line = (
                f"Epoch {epoch + 1:4d}/{config.optim.epochs}{cyc_tag} | "
                f"nll={train_metrics['nll']:.4f} "
                f"msg_kl={raw_msg:.3f}(\u03b2={beta_m:.4f}) "
                f"hid_kl={raw_hid:.3f}(\u03b2={beta_h:.4f}) "
                f"traj_kl={train_metrics['traj_kl']:.4f}{aux_tag} | "
                f"val_nll={val_metrics['nll']:.4f} | "
                f"lr={cur_lr:.2e} |grad|={avg_grad_norm:.3f}"
            )
            _log(log_line)

    # === Evaluate LAST epoch model (before loading best) ===
    last_epoch = config.optim.epochs - 1
    last_test, _ = _evaluate_dataset(model, test_tensors, config.optim.batch_size, last_epoch, config.loss, device, collect_outputs=False, forward_fn=compiled_model)
    last_full, last_outputs = _evaluate_dataset(model, full_tensors, config.optim.batch_size, last_epoch, config.loss, device, collect_outputs=True, forward_fn=compiled_model)
    assert last_outputs is not None

    last_msg_np = np.asarray(last_outputs['msg_sample'])
    gt_effectome = extract_ground_truth_effectome(splits.full.metadata, splits.full.spikes.shape[1])
    last_effectome = effectome_from_messages(last_msg_np)
    last_effectome_cos = effectome_cosine_similarity(last_effectome, gt_effectome)
    last_msg_r2 = message_r2_scores(last_msg_np, splits.full.metadata, splits.full.spikes.shape[1], seed=config.optim.seed)

    # === Evaluate BEST epoch model ===
    model.load_state_dict(best_state)

    test_metrics, _ = _evaluate_dataset(model, test_tensors, config.optim.batch_size, best_epoch, config.loss, device, collect_outputs=False, forward_fn=compiled_model)
    full_metrics, full_outputs = _evaluate_dataset(model, full_tensors, config.optim.batch_size, best_epoch, config.loss, device, collect_outputs=True, forward_fn=compiled_model)
    assert full_outputs is not None

    msg_np = np.asarray(full_outputs['msg_sample'])
    inferred_effectome = effectome_from_messages(msg_np)
    effectome_cos = effectome_cosine_similarity(inferred_effectome, gt_effectome)
    msg_r2 = message_r2_scores(msg_np, splits.full.metadata, splits.full.spikes.shape[1], seed=config.optim.seed)

    metrics = {
        'device': device.type,
        'best_epoch': int(best_epoch + 1),
        'best_val_nll': float(best_val),
        'num_parameters': int(sum(p.numel() for p in model.parameters())),
        'test': test_metrics,
        'full': full_metrics,
        'effectome_cosine_similarity': effectome_cos,
        'effectome': inferred_effectome.tolist(),
        'ground_truth_effectome': gt_effectome.tolist(),
        'message_r2': msg_r2,
        # Last epoch results for comparison
        'last_epoch': int(last_epoch + 1),
        'last_test': last_test,
        'last_full': last_full,
        'last_effectome_cosine_similarity': last_effectome_cos,
        'last_effectome': last_effectome.tolist(),
        'last_message_r2': last_msg_r2,
    }

    state_dict_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    results = FitResults(state_dict=state_dict_cpu, history=history, metrics=metrics, last_outputs=full_outputs)
    if workdir is not None:
        save_artifacts(workdir, config, results, splits.split_indices)
    if _log_fh is not None:
        _log_fh.close()
    return results
