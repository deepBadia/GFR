"""Generic comparison report for trained SurMod checkpoints.

Evaluates several trained models (typically: baseline GFR-Net on the full
pool, Active-GFR-Net from the active-learning loop, and the best model found
by Optuna tuning) on the *same* held-out test set and produces a summary
table plus comparison plots, including the key active-learning deliverable:
accuracy vs. number of labeled geometries, against the full-pool baseline.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from surmod.core.data_loader import DataLoader as GFRDataLoader
from surmod.utils.common import load_model_module, evaluate_model, resolve_data_path

from .metrics import compute_metrics, format_metrics


def _load_checkpoint(pth_path: str):
    checkpoint = torch.load(pth_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    n_geom = checkpoint["n_geom"]
    model_module = load_model_module(config)
    model = model_module.MODEL_CLASS(config, n_geom=n_geom)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    return model, model_module, config, checkpoint, n_params


def compare_models(
    runs: Dict[str, str],
    out_dir: str,
    al_history_csv: Optional[str] = None,
) -> pd.DataFrame:
    """Evaluate several trained checkpoints on the same held-out test set.

    Parameters
    ----------
    runs : dict
        ``{label: path_to_pth_checkpoint}``, e.g.
        ``{"GFR-Net (full pool)": ".../GFR-Net.pth",
           "Active-GFR-Net (queried)": ".../Active-GFR-Net.pth",
           "GFR-Net (tuned)": ".../best_GFR-Net.pth"}``.
    out_dir : str
        Where to write ``comparison_summary.csv`` and the plots.
    al_history_csv : str, optional
        Path to an active-learning ``history.csv`` (as produced by
        ``active_learning.run_active_learning``) to overlay a
        "metric vs. number of labeled geometries" curve against the
        full-pool baseline.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    dm_cache = {}
    for label, pth_path in runs.items():
        if not Path(pth_path).exists():
            print(f"[compare] WARNING: checkpoint not found for '{label}': {pth_path} (skipped)")
            continue

        model, model_module, config, checkpoint, n_params = _load_checkpoint(pth_path)
        data_path = str(resolve_data_path(config))
        if data_path not in dm_cache:
            dm_cache[data_path] = GFRDataLoader(data_path, config)
        dm = dm_cache[data_path]

        # Every checkpoint for a given dataset shares the same
        # dataset.size/val_split, so the held-out test split (dm.test_idx)
        # is identical across models -- this is what makes the comparison fair.
        data = dm.get_data()
        results = evaluate_model(model, data, model_module.process_batch, config)
        channel_metrics = compute_metrics(results["pred"], results["true"], dm.get_y_normalizer(), config)

        row = {
            "model": label,
            "architecture": config["common"]["model_name"],
            "n_params": n_params,
            "eval_set": results["eval_set"],
            "n_eval_samples": results["n_samples"],
            "weighted_mse": results["loss"],
            "best_val_loss": checkpoint.get("best_val_loss"),
            "best_epoch": checkpoint.get("best_epoch"),
        }
        row.update({f"rmse_{k}": v for k, v in channel_metrics.items()})
        rows.append(row)
        print(f"[compare] {label}: n_params={n_params:,}  weighted_mse={row['weighted_mse']:.5e}  "
              f"({format_metrics(channel_metrics)})  [eval_set={results['eval_set']}, n={results['n_samples']}]")

    if not rows:
        raise RuntimeError("No checkpoint could be evaluated -- check the `runs` paths.")

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "comparison_summary.csv", index=False)
    print(f"\nSaved: {out_dir / 'comparison_summary.csv'}")
    print(summary.to_string(index=False))

    # --- Bar chart: primary loss metric per model ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(summary["model"], summary["weighted_mse"], color="#4C78A8")
    ax.set_ylabel("Weighted MSE (test)")
    ax.set_title("Model comparison - test loss")
    ax.set_yscale("log")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "comparison_loss.png", dpi=150)
    plt.close()

    # --- Bar chart: per-channel RMSE, one group of bars per model ---
    rmse_cols = [c for c in summary.columns if c.startswith("rmse_")]
    if rmse_cols:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        n_models = len(summary)
        width = 0.8 / max(1, len(rmse_cols))
        xpos = np.arange(n_models)
        for i, col in enumerate(rmse_cols):
            ax.bar(xpos + i * width, summary[col], width=width, label=col.replace("rmse_", ""))
        ax.set_xticks(xpos + width * (len(rmse_cols) - 1) / 2)
        ax.set_xticklabels(summary["model"], rotation=20, ha="right")
        ax.set_ylabel("RMSE")
        ax.set_title("Model comparison - per-output RMSE")
        ax.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "comparison_rmse.png", dpi=150)
        plt.close()

    # --- Active-learning efficiency curve ---
    if al_history_csv and Path(al_history_csv).exists():
        hist = pd.read_csv(al_history_csv)
        if "weighted_mse" in hist.columns:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.plot(hist["n_labeled_geoms"], hist["weighted_mse"], marker="o",
                    color="#E45756", label="Active-GFR-Net (queried)")
            baseline_rows = summary[summary["architecture"] == "GFR_Net"]
            if not baseline_rows.empty:
                baseline_loss = baseline_rows["weighted_mse"].iloc[0]
                ax.axhline(baseline_loss, linestyle="--", color="#4C78A8",
                           label="Baseline GFR-Net (full pool)")
            ax.set_xlabel("Number of labeled geometries")
            ax.set_ylabel("Weighted MSE (test)")
            ax.set_yscale("log")
            ax.set_title("Active learning: accuracy vs. number of measurements")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir / "active_learning_efficiency.png", dpi=150)
            plt.close()
            print(f"Saved: {out_dir / 'active_learning_efficiency.png'}")

    print(f"Saved: {out_dir / 'comparison_loss.png'}"
          + (f" and {out_dir / 'comparison_rmse.png'}" if rmse_cols else ""))
    return summary


def find_checkpoint(pattern_root: str, glob_pattern: str) -> str:
    """Return the most recently modified file under `pattern_root` matching `glob_pattern`."""
    candidates = sorted(Path(pattern_root).rglob(glob_pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No file matching '{glob_pattern}' found under {pattern_root}")
    return str(candidates[0])


def load_tuning_best(tuning_root: str) -> Optional[str]:
    """Read the checkpoint path written by `save_best_artifacts` (see core/tuner.py)."""
    info_path = Path(tuning_root)
    if info_path.is_dir():
        info_path = info_path / "best_run_info.json"
    if not info_path.exists():
        return None
    with open(info_path, encoding="utf-8") as f:
        info = json.load(f)
    return info.get("checkpoint_path")
