"""Generate metrics and plots from a completed training run.

This version is designed to work primarily from the .pth checkpoint.
It loads config, normalizers, and axis information directly from the checkpoint.
It also saves data split statistics (train/val/test + remaining pool) to the metrics CSV.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import glob
import os

from surmod.utils.common import (
    Normalizer,
    load_checkpoint,
    load_model_module,
    compute_gain_phase_rmse,
    evaluate_model,
    get_project_root,
    resolve_data_path,
)
from surmod.utils.plotting import (
    plot_amplitude_phase,
    plot_predictions_vs_true,
    plot_train_val_convergence,
)


def _load_normalizers_from_checkpoint(checkpoint: dict):
    """Load normalizers directly from the .pth file."""
    if checkpoint and "y_normalizer" in checkpoint and "x_normalizer" in checkpoint:
        y_normalizer = Normalizer()
        y_normalizer.load_state_dict(checkpoint["y_normalizer"])
        x_normalizer = Normalizer()
        x_normalizer.load_state_dict(checkpoint["x_normalizer"])
        return y_normalizer, x_normalizer
    print("WARNING: Normalizers not found in checkpoint.")
    return None, None


def gen_results(
    save_path: str | Path | None = None,
) -> None:
    """
    Generate metrics + plots from a trained model.
    Primary dependency is the .pth file.
    """
    if save_path is None:
        save_path = (
            "/users/t0327721/Documents/projet_GFR_Net/results/mohycan_avecSymetries/test_GFR_Net_20260716_1435_tuning/"
        )

    save_path = Path(save_path)
    if not save_path.exists():
        raise FileNotFoundError(f"Path not found: {save_path}")

    checkpoint_path = glob.glob(os.path.join(save_path, "*.pth"))[0]
    # === Load everything from the .pth ===
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    config = checkpoint.get("config")
    if config is None:
        raise ValueError("Checkpoint is missing 'config'. Cannot continue.")

    n_geom = checkpoint.get("n_geom")
    axis = checkpoint.get("axis")
    axis_norm = checkpoint.get("axis_norm")

    data_path = resolve_data_path(config)

    # === Load model ===
    model_module = load_model_module(config)
    model = model_module.MODEL_CLASS(config, n_geom=n_geom)
    load_checkpoint(model, str(checkpoint_path))
    model.eval()

    # === Load normalizers from .pth ===
    y_normalizer, x_normalizer = _load_normalizers_from_checkpoint(checkpoint)

    # === Load data + compute split statistics ===
    try:
        from surmod.core.data_loader import DataLoader as GFRDataLoader
        dl = GFRDataLoader(str(data_path), config)
        data = dl.get_data()

        if axis is not None:
            data["axis"] = axis
        if axis_norm is not None:
            data["axis_norm"] = axis_norm

        # === Data split statistics ===
        n_train_geoms = len(data.get("X_train", []))
        n_val_geoms = len(data.get("X_val", []))
        n_test_geoms = len(data.get("X_test", [])) if data.get("X_test") is not None else 0

        axis_mask_train = data.get("axis_mask_train")
        axis_mask_val = data.get("axis_mask_val")

        n_train_points = int(axis_mask_train.sum().item()) if axis_mask_train is not None else 0
        n_val_points = int(axis_mask_val.sum().item()) if axis_mask_val is not None else 0

        n_pool_remaining = 0
        if hasattr(dl, "pool_idx") and hasattr(dl, "labeled_mask"):
            seen = np.any(dl.labeled_mask, axis=1)
            n_pool_remaining = len(dl.pool_idx) - np.sum(seen)

    except Exception as e:
        print(f"Warning: Could not fully reload dataloader ({e}).")
        data = {"axis": axis, "axis_norm": axis_norm, "n_geom": n_geom}
        n_train_geoms = n_val_geoms = n_test_geoms = n_train_points = n_val_points = n_pool_remaining = 0

    # === Evaluation ===
    results = evaluate_model(
        model=model,
        data=data,
        process_batch=model_module.process_batch,
        config=config,
    )

    # === Metrics (with split info) ===
    n_out = config["common"].get("n_out", 2)

    metrics = {
        "weighted_mse": results.get("weighted_mse", np.nan),
        config["training"].get("metric", "weighted_mse"): results.get("loss", np.nan),
        "n_geom": n_geom,
        "n_axis": len(axis) if axis is not None else None,
        "n_out": n_out,
        # Data split
        "n_train_geoms": n_train_geoms,
        "n_val_geoms": n_val_geoms,
        "n_test_geoms": n_test_geoms,
        "n_train_points": n_train_points,
        "n_val_points": n_val_points,
        "n_pool_remaining": n_pool_remaining,
    }

    complex_format = config["common"].get("complex_format")
    if n_out >= 2 and y_normalizer is not None:
        gain_rmse, phase_rmse = compute_gain_phase_rmse(
            results["pred"], results["true"], y_normalizer, complex_format
        )
        if gain_rmse is not None:
            metrics["gain_rmse_dB"] = round(gain_rmse, 4)
            metrics["phase_rmse_deg"] = round(phase_rmse, 2)
            print(f"Gain RMSE = {gain_rmse:.4f} dB")
            print(f"Phase RMSE = {phase_rmse:.2f} deg")
    #
    # metrics_path = checkpoint_path.parent / "best_model_metrics.csv"
    # pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    # print(f"Metrics saved ? {metrics_path}")

    # === Plots ===
    plots_dir = save_path / "plots"
    plots_dir.mkdir(exist_ok=True)

    n_samples = 3

    # New train/val convergence plot from train_metrics CSV
    history_csv = save_path / "train_metrics.csv"
    if history_csv.exists():
        plot_train_val_convergence(
            str(history_csv),
            str(plots_dir / "train_val_convergence.png"),
            log_scale=True,
        )

    is_polar = complex_format == "polar" and n_out == 3
    is_cartesian = n_out == 2 and config.get("common", {}).get("complex", False)

    sample_indices = None
    if is_polar or is_cartesian:
        sample_indices = plot_amplitude_phase(
            results=results,
            config=config,
            save_dir=plots_dir,
            n_samples=n_samples,
            y_normalizer=y_normalizer,
            x_normalizer=x_normalizer,
            axis=axis,
            X=results.get("X"),
        )

    plot_predictions_vs_true(
        results=results,
        config=config,
        save_dir=plots_dir,
        n_samples=n_samples,
        y_normalizer=y_normalizer,
        x_normalizer=x_normalizer,
        X=results.get("X"),
        sample_indices=sample_indices,
    )

    # === Generate nicely formatted training report (TXT) ===
    report_path = save_path / "training_report.txt"

    best_epoch = checkpoint.get("best_epoch", "N/A")
    best_val_loss = checkpoint.get("best_val_loss", float("nan"))

    lines = []
    lines.append("=" * 85)
    lines.append("                        GFR-Net TRAINING REPORT")
    lines.append("=" * 85)
    lines.append("")

    # Model Information
    lines.append("MODEL INFORMATION")
    lines.append("-" * 40)
    lines.append(f"  Model Name              : {config.get('common', {}).get('model_name', 'N/A')}")
    lines.append(f"  Checkpoint              : N/A")
    lines.append(f"  Best Epoch              : {best_epoch}")
    lines.append(f"  Best Validation Loss    : {best_val_loss:.6e}")
    lines.append("")

    # Dataset Information
    lines.append("DATASET INFORMATION")
    lines.append("-" * 40)
    lines.append(f"  Data File               : {config['common']['data_file']}")
    lines.append(f"  Complex Format          : {config['common'].get('complex_format', 'N/A')}")
    lines.append(f"  Number of Outputs       : {n_out}")
    lines.append(f"  Geometry Features       : {n_geom}")
    lines.append(f"  Frequency Points        : {len(axis) if axis is not None else 'N/A'}")
    lines.append("")

    # Data Split Summary
    lines.append("DATA SPLIT SUMMARY")
    lines.append("-" * 40)
    lines.append(f"  {'':<22}{'Geometries':>12}{'Points':>12}")
    lines.append(f"  {'Train':<22}{n_train_geoms:>12}{n_train_points:>12}")
    lines.append(f"  {'Validation':<22}{n_val_geoms:>12}{n_val_points:>12}")
    lines.append(f"  {'Test':<22}{n_test_geoms:>12}{'-':>12}")
    lines.append(f"  {'Total Labeled':<22}{'-':>12}{n_train_points + n_val_points:>12}")
    lines.append(f"  {'Remaining in Pool':<22}{n_pool_remaining:>12}{'-':>12}")
    lines.append("")

    # Performance Metrics
    lines.append("PERFORMANCE METRICS")
    lines.append("-" * 40)
    lines.append(f"  Weighted MSE            : {metrics.get('weighted_mse', float('nan')):.6e}")
    if "gain_rmse_dB" in metrics:
        lines.append(f"  Gain RMSE (dB)          : {metrics['gain_rmse_dB']:.4f}")
    if "phase_rmse_deg" in metrics:
        lines.append(f"  Phase RMSE (deg)        : {metrics['phase_rmse_deg']:.2f}")
    lines.append("")

    # Training Configuration
    lines.append("TRAINING CONFIGURATION")
    lines.append("-" * 40)
    training_cfg = config.get("training", {})
    lines.append(f"  Epochs                  : {training_cfg.get('epochs', 'N/A')}")
    lines.append(f"  Batch Size              : {training_cfg.get('batch_size', 'N/A')}")
    lines.append(f"  Learning Rate           : {training_cfg.get('lr', training_cfg.get('learning_rate', 'N/A'))}")
    lines.append(f"  Weight Decay            : {training_cfg.get('w_decay', training_cfg.get('weight_decay', 'N/A'))}")
    lines.append(f"  Patience                : {training_cfg.get('patience', 'N/A')}")
    lines.append(f"  Metric                  : {training_cfg.get('metric', 'weighted_mse')}")
    lines.append("")

    lines.append("=" * 85)
    lines.append(f"Report generated from: {save_path}")
    lines.append("=" * 85)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Saved: {report_path}")


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate metrics and plots from a .pth checkpoint.")
    parser.add_argument("--checkpoint", default=None, help="Path to .pth file")
    parser.add_argument("--data_path", default=None, help="Optional HDF5 data path")
    args = parser.parse_args(argv)
    gen_results(save_path=args.checkpoint)
    return 0


if __name__ == "__main__":
    sys.exit(main())
