"""Data handling utilities."""

import numpy as np
import importlib
import os
import torch
from pathlib import Path
from typing import Any, Dict, Optional
from torch.utils.data import DataLoader, TensorDataset


class Normalizer:
    """Simple per-channel mean/std normalizer."""
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit(self, data: np.ndarray) -> "Normalizer":
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        self.std = np.where(self.std < 1e-10, 1.0, self.std)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data * self.std + self.mean

    def state_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state: dict):
        self.mean = state["mean"]
        self.std = state["std"]


def load_model_module(config: Dict[str, Any]):
    """Import the Python module that defines the selected model."""
    return importlib.import_module(f"surmod.models.{config['common']['model_name']}")


def load_checkpoint(model, checkpoint_path: str) -> Optional[dict]:
    """Load model weights from a checkpoint file when it exists."""
    if not os.path.exists(checkpoint_path):
        return None
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def get_project_root() -> Path:
    """Locate the project root by searching for marker files.

    Falls back to the current working directory when the package is installed
    (no source-tree markers are present). This keeps data-path resolution and
    Optuna storage working both in editable installs and after ``pip install``.
    """
    current = Path(__file__).resolve().parent
    markers = (
        "configs.yaml",
        "requirements.txt",
        ".git",
        "README.md",
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
    )

    for _ in range(12):
        if any((current / marker).exists() for marker in markers):
            return current
        if current.parent == current:
            break
        current = current.parent

    return Path.cwd()


def resolve_data_path(config: Dict[str, Any]) -> Path:
    """Resolve the HDF5 data file path from a config dictionary.

    Supports both absolute paths (useful for temporary test data) and
    relative paths that are looked up under ``<project_root>/data/``.
    """
    data_file = config["common"]["data_file"]
    p = Path(data_file)
    if p.is_absolute():
        return p
    return get_project_root() / "data" / data_file


def evaluate_model(model, data, process_batch, config, eval_batch_size: int = 512):
    """Evaluate a model and return predictions, targets, and metric values."""
    device = next(model.parameters()).device
    model = model.to(device)
    model.eval()

    if (
        "X_test" in data
        and data.get("X_test") is not None
        and len(data["X_test"]) > 0
    ):
        x_eval = data["X_test"]
        y_eval = data["Y_test"]
        w_eval = data["W_test"]
        eval_name = "test"
    elif data.get("X_val") is not None and len(data["X_val"]) > 0:
        x_eval = data["X_val"]
        y_eval = data["Y_val"]
        w_eval = data["W_val"]
        eval_name = "val"
    else:
        x_eval = data["X_train"]
        y_eval = data["Y_train"]
        w_eval = data["W_train"]
        eval_name = "train (full)"

    axis_norm = data["axis_norm"].to(device)
    eval_loader = DataLoader(
        TensorDataset(x_eval, y_eval, w_eval),
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
    )

    total_loss = 0.0
    total_sse = 0.0
    total_weight = 0.0
    pred_batches = []

    with torch.no_grad():
        for bx, by, bw in eval_loader:
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)
            bw = bw.to(device, non_blocking=True)

            outputs = process_batch(
                model=model,
                bx=bx,
                by=by,
                bw=bw,
                axis_norm=axis_norm,
                train_sse=0.0,
                train_total_weight=0.0,
                device=device,
                config=config,
                mask=None,
            )

            if len(outputs) == 4:
                loss, batch_sse, batch_weight, pred_batch = outputs
            else:
                loss, batch_sse, batch_weight = outputs
                pred_batch = None

            total_loss += loss.item() * batch_weight
            total_sse += batch_sse
            total_weight += batch_weight
            if pred_batch is not None:
                pred_batches.append(pred_batch.cpu())

    pred_raw = torch.cat(pred_batches, dim=0)
    final_loss = total_loss / (total_weight + 1e-12)

    y_eval_cpu = y_eval.cpu()
    pred_raw_cpu = pred_raw.cpu()
    w_eval_cpu = w_eval.cpu()
    x_eval_cpu = x_eval.cpu()

    _, _, _, all_metrics = compute_loss(
        pred_raw.to(device),
        y_eval.to(device),
        weights=w_eval.to(device),
        config=config,
        return_all_metrics=True,
        training=False,
    )

    results = {
        "loss": float(final_loss),
        "pred": pred_raw_cpu.numpy(),
        "true": y_eval_cpu.numpy(),
        "weights": w_eval_cpu.numpy(),
        "X": x_eval_cpu.numpy(),
        "n_samples": len(x_eval),
        "eval_set": eval_name,
        **all_metrics,
    }
    return results


def compute_gain_phase_rmse(
    pred_norm: np.ndarray,
    true_norm: np.ndarray,
    y_normalizer,
    complex_format: Optional[str],
):
    """Compute gain RMSE (dB) and phase RMSE (degrees) from normalized predictions."""
    n_out = pred_norm.shape[-1]
    if n_out == 1:
        return None, None

    shape = pred_norm.shape
    pred = y_normalizer.inverse_transform(pred_norm.reshape(-1, n_out)).reshape(shape)
    true = y_normalizer.inverse_transform(true_norm.reshape(-1, n_out)).reshape(shape)

    if complex_format == "polar" and n_out == 3:
        gain_pred = pred[..., 0]
        gain_true = true[..., 0]
        phase_pred = np.degrees(np.arctan2(pred[..., 1], pred[..., 2]))
        phase_true = np.degrees(np.arctan2(true[..., 1], true[..., 2]))
    elif n_out == 2:
        gain_pred = 20.0 * np.log10(
            np.sqrt(pred[..., 0] ** 2 + pred[..., 1] ** 2) + 1e-12
        )
        gain_true = 20.0 * np.log10(
            np.sqrt(true[..., 0] ** 2 + true[..., 1] ** 2) + 1e-12
        )
        phase_pred = np.degrees(np.arctan2(pred[..., 1], pred[..., 0]))
        phase_true = np.degrees(np.arctan2(true[..., 1], true[..., 0]))
    else:
        return None, None

    gain_rmse = float(np.sqrt(np.mean((gain_pred - gain_true) ** 2)))
    k = np.round((phase_true - phase_pred) / 180.0)
    phase_pred_adj = phase_pred + 180.0 * k
    phase_diff = (phase_pred_adj - phase_true + 180.0) % 360.0 - 180.0
    phase_rmse = float(np.sqrt(np.mean(phase_diff ** 2)))
    return gain_rmse, phase_rmse


def compute_loss(
    pred,
    target,
    weights=None,
    mask=None,
    config=None,
    return_all_metrics: bool = False,
    training: bool = True,
):
    """Compute the configured training loss and optional regularization terms."""
    if config is None:
        raise ValueError("config must be passed to compute_loss")

    metric = config["training"].get("metric", "weighted_mse")
    epsilon = float(config["training"].get("epsilon", 1e-8))
    delta = 1.0

    if weights is None:
        weights = torch.ones_like(pred, device=pred.device)
    elif weights.ndim == 2:
        weights = weights.unsqueeze(-1)
    if mask is not None:
        if mask.ndim == 2:
            mask = mask.unsqueeze(-1)
        weights = weights * mask.float()

    total_weight = weights.sum() + 1e-12

    def _compute_sse(metric_name: str):
        if metric_name == "weighted_mse":
            diff_sq = (pred - target) ** 2
            return (diff_sq * weights).sum()
        if metric_name == "relative_mse":
            diff_sq = (pred - target) ** 2
            denom = target ** 2 + epsilon
            return ((diff_sq / denom) * weights).sum()
        if metric_name == "msle":
            pred_log = torch.log1p(torch.abs(pred))
            target_log = torch.log1p(torch.abs(target))
            diff_sq = (pred_log - target_log) ** 2
            return (diff_sq * weights).sum()
        if metric_name == "huber":
            diff = torch.abs(pred - target)
            quadratic = torch.minimum(diff, torch.full_like(diff, delta))
            linear = diff - quadratic
            huber_loss = 0.5 * quadratic ** 2 + delta * linear
            return (huber_loss * weights).sum()
        raise ValueError(f"Unknown metric: {metric_name}")

    total_sse = _compute_sse(metric)
    loss = total_sse / total_weight

    if training:
        phase_reg_lambda = float(config["training"].get("phase_reg_lambda", 0.0))
        if phase_reg_lambda > 0.0 and pred.shape[-1] == 3:
            sin_p = pred[..., 1]
            cos_p = pred[..., 2]
            unit_err = ((sin_p ** 2 + cos_p ** 2) - 1.0) ** 2
            loss = loss + phase_reg_lambda * unit_err.mean()

        freq_smooth_lambda = float(config["training"].get("freq_smooth_lambda", 0.0))
        if freq_smooth_lambda > 0.0 and pred.ndim == 3 and pred.shape[1] > 1:
            phase_pred = pred[..., 1:] if pred.shape[-1] == 3 else pred
            freq_diff = phase_pred[:, 1:, :] - phase_pred[:, :-1, :]
            loss = loss + freq_smooth_lambda * (freq_diff ** 2).mean()

    if not return_all_metrics:
        return loss, total_sse, total_weight

    all_metrics = {}
    for metric_name in ("weighted_mse", "relative_mse", "msle", "huber"):
        sse_m = _compute_sse(metric_name)
        all_metrics[metric_name] = float(sse_m / total_weight)

    return loss, total_sse, total_weight, all_metrics
