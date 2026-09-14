"""High-level Python API for SurMod package.

This module provides convenient wrapper functions for training, hyperparameter
tuning, post-processing / report generation, and inference.
These functions can be imported and used directly from Python (e.g. in notebooks
or other projects) without going through the command line.

Example usage:
    from surmod.main import train_model, generate_results, predict

    # Standard training
    train_model(experiment="comp_full_gfr_baseline", config_path="configs.yaml")

    # With tuning
    train_model(experiment="comp_full_gfr_baseline", tuner=True, time_hours=2)

    # Generate plots + report from a previous run folder
    generate_results("/path/to/results/.../trial_0")

    # Predict with a trained model
    pred = predict("path/to/model.pth", [0.5, 1.2, 0.8])
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch

from .utils.yaml_handler import load_config
from .utils.common import (
    get_project_root,
    load_model_module,
    resolve_data_path,
    Normalizer,
    load_checkpoint,
)
from .core.data_loader import DataLoader as GFRDataLoader


def train_model(
    experiment: str,
    config_path: Optional[str] = None,
    save_path: Optional[str | Path] = None,
    tuner: bool = False,
    time_hours: float = 24.0,
    **config_overrides: Any,
) -> Any:
    """Run standard training or hyperparameter tuning for a given experiment.

    This is the main entry point for both interactive Python use and the CLI.

    Parameters
    ----------
    experiment : str
        Name of the experiment defined in the YAML config file (under `experiments:`).
    config_path : str, optional
        Path to the YAML configuration file. If None, looks for `configs.yaml`
        in the project root.
    save_path : str or Path, optional
        Base directory where results/ subfolder will be created.
        If None, uses current working directory.
    tuner : bool, default False
        If True, run Optuna hyperparameter tuning instead of a single training run.
    time_hours : float, default 24.0
        Maximum wall-clock time for tuning (only used when `tuner=True`).
    **config_overrides
        Optional keyword arguments to override config values at runtime
        (advanced use, applied after loading).

    Returns
    -------
    float or optuna.Study
        Validation loss of best model (standard training) or the Optuna study object (tuning).
    """
    config = load_config(experiment, config_file=config_path)

    # Apply any runtime overrides (simple top-level merge for common keys)
    if config_overrides:
        for section, values in config_overrides.items():
            if section in config and isinstance(config[section], dict):
                config[section].update(values)
            else:
                config[section] = values

    save_path = str(save_path) if save_path is not None else "."

    if tuner:
        from .core.tuner import run_tuning
        return run_tuning(
            config_name=experiment,
            config_file=config_path,
            time_hours=time_hours,
            save_path=save_path,
        )

    # --- Standard (non-tuning) training path ---
    from .core.trainer import train_model as _core_train_model

    data_path = str(resolve_data_path(config))

    dm = GFRDataLoader(data_path, config)

    model_module = load_model_module(config)
    n_geom = len(config["dataset"].get("X_columns", []))

    model = model_module.MODEL_CLASS(config, n_geom=n_geom)
    process_batch = getattr(model_module, "process_batch", None)
    if process_batch is None:
        raise RuntimeError(
            f"Module {model_module} does not expose a process_batch function. "
            "Check your model file."
        )

    return _core_train_model(
        model=model,
        dataloader=dm,
        config=config,
        process_batch=process_batch,
        save_path=save_path,
    )


def generate_results(save_path: str | Path) -> None:
    """Generate metrics, plots and training report from a completed run folder.

    This is the professional post-processing entry point (replacement for the
    old internal `gen_results` script).

    Parameters
    ----------
    save_path : str or Path
        Path to the experiment output directory that contains a *.pth checkpoint
        (usually the folder created by `train_model`).
    """
    from .core.gen_results import gen_results as _gen_results
    _gen_results(save_path=str(save_path))


# Convenience alias (some users prefer this name)
post_process = generate_results


def predict(
    model_path: str | Path,
    X_values: Union[Sequence[float], np.ndarray],
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Run inference with a trained SurMod model.

    Parameters
    ----------
    model_path :
        Path to a ``.pth`` checkpoint produced by training.
    X_values :
        Geometry vector (raw physical values, **not** normalised).
        Must have the same length as the number of geometry features
        the model was trained with.
    device :
        Optional device string (``"cpu"``, ``"cuda:0"``, ).
        If ``None``, uses CUDA when available, otherwise CPU.

    Returns
    -------
    dict
        {
            "prediction": np.ndarray of shape (n_freq, n_out),  # denormalised
            "axis":       np.ndarray of shape (n_freq,),        # physical axis values
            "prediction_norm": np.ndarray,                      # normalised (optional)
        }

    Raises
    ------
    ValueError
        If the geometry vector has the wrong length.
    FileNotFoundError
        If the checkpoint does not exist.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    # ------------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------------
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    n_geom = checkpoint["n_geom"]
    axis = checkpoint.get("axis")
    axis_norm = checkpoint.get("axis_norm")

    x_normalizer = Normalizer()
    x_normalizer.load_state_dict(checkpoint["x_normalizer"])
    y_normalizer = Normalizer()
    y_normalizer.load_state_dict(checkpoint["y_normalizer"])

    # ------------------------------------------------------------------
    # Check geometry vector size
    # ------------------------------------------------------------------
    geom = np.asarray(X_values, dtype=np.float32).ravel()
    expected = n_geom
    if geom.size != expected:
        x_cols = config.get("dataset", {}).get("X_columns", [f"x{i}" for i in range(expected)])
        raise ValueError(
            f"Geometry vector has length {geom.size}, but the model expects "
            f"{expected} features.\n"
            f"Expected order: {x_cols}"
        )

    # ------------------------------------------------------------------
    # Rebuild model and load weights
    # ------------------------------------------------------------------
    model_module = load_model_module(config)
    model = model_module.MODEL_CLASS(config, n_geom=n_geom)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # ------------------------------------------------------------------
    # Prepare inputs
    # ------------------------------------------------------------------
    x_norm = x_normalizer.transform(geom.reshape(1, -1)).astype(np.float32)
    x_tensor = torch.from_numpy(x_norm).to(device)

    if axis_norm is None:
        raise RuntimeError(
            "Checkpoint is missing 'axis_norm'. "
            "Re-train the model with a recent version of SurMod."
        )
    axis_norm_t = torch.as_tensor(axis_norm, dtype=torch.float32, device=device)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    with torch.no_grad():
        pred_norm = model(x_tensor, axis_norm_t)  # (1, n_freq, n_out)
    pred_norm = pred_norm.cpu().numpy()[0]       # (n_freq, n_out)

    # Denormalise
    n_out = pred_norm.shape[-1]
    pred = y_normalizer.inverse_transform(pred_norm.reshape(-1, n_out)).reshape(pred_norm.shape)

    axis_out = None
    if axis is not None:
        axis_out = np.asarray(axis).ravel()

    return {
        "prediction": pred,               # denormalised (n_freq, n_out)
        "axis": axis_out,                 # physical frequency / sweep values
        "prediction_norm": pred_norm,     # normalised prediction
    }


__all__ = [
    "train_model",
    "generate_results",
    "post_process",
    "predict",
    "load_config",
]