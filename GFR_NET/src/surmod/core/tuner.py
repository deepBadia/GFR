"""Hyperparameter tuning for GFR-Net experiments (supervised only for now)."""
from __future__ import annotations
import gc
import json
import os
import shutil
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime

import optuna
import torch

from surmod.core.data_loader import DataLoader
from surmod.utils.yaml_handler import load_config
from surmod.utils.common import get_project_root, resolve_data_path

optuna.logging.set_verbosity(optuna.logging.WARNING)


def data_path(config: Dict[str, Any]) -> str:
    return str(resolve_data_path(config))


def run_standard(config, dm, timestamp, trial=None, save_path=None):
    from surmod.core.trainer import train_model
    from surmod.models.GFR_Net import MODEL_CLASS, process_batch

    n_geom = len(config["dataset"]["X_columns"])
    model = MODEL_CLASS(config, n_geom=n_geom)
    return train_model(
        model=model,
        dataloader=dm,
        config=config,
        process_batch=process_batch,
        save_path=save_path,
        trial=trial,
        timestamp=timestamp,
    )


def _get_tuning_config(config):
    default = {
        "hidden_dim": [64, 128, 192, 256, 320, 384, 512],
        "n_fourier": [16, 32, 48, 64, 96],
        "n_layers": [4, 6, 8, 10],
        "dropout": [0.0, 0.5],
        "lr": [1e-5, 1e-2],
        "w_decay": [1e-7, 1e-2],
        "batch_size": [256, 512, 1024, 2048],
    }
    return {**default, **config.get("tuning", {})}


def _suggest_params(trial, config):
    tuning = _get_tuning_config(config)
    config["model"]["hidden_dim"] = trial.suggest_categorical("hidden_dim", tuning["hidden_dim"])
    config["model"]["n_fourier"] = trial.suggest_categorical("n_fourier", tuning["n_fourier"])
    config["model"]["n_layers"] = trial.suggest_int("n_layers", tuning["n_layers"][0], tuning["n_layers"][-1])
    config["model"]["dropout"] = trial.suggest_float("dropout", tuning["dropout"][0], tuning["dropout"][-1])
    config["training"]["lr"] = trial.suggest_float("lr", tuning["lr"][0], tuning["lr"][-1], log=True)
    config["training"]["w_decay"] = trial.suggest_float("w_decay", tuning["w_decay"][0], tuning["w_decay"][-1], log=True)
    config["training"]["batch_size"] = trial.suggest_categorical("batch_size", tuning["batch_size"])


def objective(trial, study, config, dm, save_path, timestamp):
    config["common"]["verbose"] = False
    _suggest_params(trial, config)
    val_loss = run_standard(config, dm=dm, timestamp=timestamp, trial=trial, save_path=save_path)
    torch.cuda.empty_cache()
    gc.collect()
    return val_loss


# ====================== NEW: LIVE BEST SAVING CALLBACK ======================
def save_best_artifacts(
    study: optuna.Study,
    trial: optuna.trial.FrozenTrial,
    save_path: str,
    config: dict,
    timestamp: str,
):
    """Copy the current best trial's files to the tuning root folder."""
    best_num = trial.number
    model_name = config['common']['model_name']
    prefix = config["common"].get("save_name") or ""
    data_stem = Path(config["common"]["data_file"]).stem
    tuning_root = os.path.join(
        save_path,
        "results",
        data_stem,
        f"{prefix}_{model_name}_{timestamp}_tuning",
    )
    best_trial_dir = os.path.join(tuning_root, "trials", f"trial_{best_num}")

    if not os.path.isdir(best_trial_dir):
        print(f"[Warning] Best trial dir not found: {best_trial_dir}")
        return

    print(f"\n New best trial #{best_num} (val_loss={trial.value:.6f}) -> updating root folder")

    # Copy config and metrics
    for fname in ["config.yaml", "train_metrics.csv"]:
        src = os.path.join(best_trial_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(tuning_root, fname))

    # Copy best .pth
    best_ckpt_path = None
    pth_files = [f for f in os.listdir(best_trial_dir) if f.endswith(".pth")]
    if pth_files:
        src_pth = os.path.join(best_trial_dir, pth_files[0])
        best_ckpt_path = os.path.join(tuning_root, f"best_{pth_files[0]}")
        shutil.copy2(src_pth, best_ckpt_path)

    # Small manifest so downstream scripts (e.g. comparison reports) can find
    # the current best checkpoint without having to glob/guess timestamps.
    info = {
        "best_trial_number": best_num,
        "best_val_loss": trial.value,
        "checkpoint_path": best_ckpt_path,
        "config_path": os.path.join(tuning_root, "config.yaml"),
        "tuning_root": tuning_root,
    }
    with open(os.path.join(tuning_root, "best_run_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print(f"   Best artifacts updated in: {tuning_root}")


def best_trial_callback(
    study: optuna.Study,
    trial: optuna.trial.FrozenTrial,
    save_path: str,
    config: dict,
    timestamp: str,
):
    """Called automatically by Optuna after every trial.
    Only acts when this trial became the new best.
    """
    if study.best_trial is not None and study.best_trial.number == trial.number:
        save_best_artifacts(study, trial, save_path, config, timestamp)


# ====================== MAIN TUNING FUNCTION ======================
def run_tuning(
    config_name: str,
    config_file: Optional[str] = None,
    time_hours: float = 24.0,
    save_path: str = "",
    n_trials: Optional[int] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> optuna.Study:
    """
    Parameters
    ----------
    config_overrides : dict, optional
        Per-section overrides merged into the loaded config, e.g.
        ``{"training": {"epochs": 50, "patience_start": 0}}`` to run a fast
        smoke test without editing the YAML file.
    """
    # Computed once per call (not at import time) so that running tuning for
    # several experiments within the same process gets a distinct timestamp
    # -- and therefore a distinct results/ folder -- for each one.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    config = load_config(config_name, config_file)
    if config_overrides:
        for section, values in config_overrides.items():
            if section in config and isinstance(config[section], dict):
                config[section].update(values)
            else:
                config[section] = values

    print("=" * 70)
    print(f" Starting Hyperparameter Tuning (LIVE best updates enabled)")
    print(f" Experiment : {config_name}")
    print(f" Model      : {config['common']['model_name']}")
    print(f" Save path  : {save_path}")
    print("=" * 70)

    dm = DataLoader(data_path(config), config)

    ts = os.path.basename(save_path).rsplit("_", 1)[-1] if save_path else "default"
    data_stem = Path(config["common"]["data_file"]).stem
    study_name = f"{config['common']['model_name']}_{data_stem}_{ts}_{timestamp}"

    print(study_name)

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=400, interval_steps=80),
        storage=f"sqlite:///{get_project_root()}/optuna.db",
        study_name=study_name,
        load_if_exists=True,
    )

    obj = partial(objective, study=study, config=config, dm=dm, save_path=save_path, timestamp=timestamp)

    # === Use callback for live best updates ===
    callback = partial(best_trial_callback, save_path=save_path, config=config, timestamp=timestamp)

    study.optimize(
        obj,
        n_trials=n_trials,
        timeout=int(3600 * time_hours),
        n_jobs=1,
        show_progress_bar=True,
        gc_after_trial=True,
        callbacks=[callback],
    )

    print("\n" + "=" * 70)
    print(f" Tuning finished!")
    print(f" Best trial : {study.best_trial.number}")
    print(f" Best value : {study.best_trial.value:.6f}")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")
    print("=" * 70)

    return study
