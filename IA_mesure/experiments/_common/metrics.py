"""Dataset-agnostic evaluation metrics shared by the active-learning driver
and the comparison scripts.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from surmod.utils.common import compute_gain_phase_rmse


def effective_output_names(config: Dict[str, Any], n_out: int) -> List[str]:
    """Names of the model's actual output channels.

    Usually just ``dataset.Y_columns``, EXCEPT when ``complex_format: polar``
    turned 2 raw columns (e.g. S_real, S_imag) into 3 derived channels -- in
    that case the model's real outputs are (gain_dB, sin(phase), cos(phase)),
    not the original column names (see `DataLoader._to_polar`).
    """
    if config.get("common", {}).get("complex_format") == "polar" and n_out == 3:
        return ["gain_dB", "sin_phase", "cos_phase"]

    names = list(config.get("dataset", {}).get("Y_columns", []) or [])
    if len(names) < n_out:
        names = names + [f"out_{i}" for i in range(len(names), n_out)]
    return names[:n_out]


def display_name(config: Dict[str, Any], key: str) -> str:
    """Human-readable label for a raw h5 column name.

    Usually just the column name itself, except for datasets like
    mesure_gain.h5 / mesure_couplage.h5 whose h5 column names are confirmed
    to be swapped relative to their actual physical meaning (e.g. the column
    literally named "theta" really holds frequency). Their config.yaml sets
    ``dataset.display_names`` to override the label without touching the
    column names DataLoader uses to index into the h5 file.
    """
    return config.get("dataset", {}).get("display_names", {}).get(key, key)


def per_channel_rmse(
    pred_norm: np.ndarray,
    true_norm: np.ndarray,
    y_normalizer,
    names: List[str],
) -> Dict[str, float]:
    """Denormalise predictions/targets and compute one RMSE per output channel.

    A column whose name contains "phase" (case-insensitive) but is not a
    sin/cos component (e.g. "phase_degrees", but NOT "sin_phase"/"cos_phase")
    is compared with wraparound (shortest angular distance in degrees)
    instead of a plain difference, since such columns store an angle in
    degrees directly. Sin/cos components are bounded in [-1, 1] and must NOT
    be wrapped -- a plain difference is correct for them.
    """
    n_out = pred_norm.shape[-1]
    shape = pred_norm.shape
    pred = y_normalizer.inverse_transform(pred_norm.reshape(-1, n_out)).reshape(shape)
    true = y_normalizer.inverse_transform(true_norm.reshape(-1, n_out)).reshape(shape)

    results = {}
    for c, name in enumerate(names[:n_out]):
        diff = pred[..., c] - true[..., c]
        lname = name.lower()
        if "phase" in lname and "sin" not in lname and "cos" not in lname:
            diff = (diff + 180.0) % 360.0 - 180.0
        results[name] = float(np.sqrt(np.mean(diff ** 2)))
    return results


def compute_metrics(
    pred_norm: np.ndarray,
    true_norm: np.ndarray,
    y_normalizer,
    config: Dict[str, Any],
) -> Dict[str, float]:
    """Return a dict of interpretable metrics appropriate for this config.

    Always includes a plain per-channel RMSE (works for any dataset).
    When the target genuinely represents a complex quantity
    (``common.complex: true``), also adds physically meaningful
    ``gain_dB`` / ``phase_deg`` RMSE computed from the real/imag or
    gain/sin/cos representation -- this must NOT be used for datasets whose
    two output channels are something else entirely (e.g. amplitude_linear
    + phase_degrees stored directly, as in secteur1_amp_phase.h5), which is
    why it is gated on ``common.complex``.
    """
    n_out = pred_norm.shape[-1]
    names = effective_output_names(config, n_out)
    metrics = per_channel_rmse(pred_norm, true_norm, y_normalizer, names)

    if config.get("common", {}).get("complex", False):
        complex_format = config["common"].get("complex_format")
        gain_rmse, phase_rmse = compute_gain_phase_rmse(pred_norm, true_norm, y_normalizer, complex_format)
        if gain_rmse is not None:
            metrics["gain_dB"] = gain_rmse
            metrics["phase_deg"] = phase_rmse

    return metrics


def format_metrics(metrics: Dict[str, float]) -> str:
    return ", ".join(f"{k}={v:.4g}" for k, v in metrics.items())
