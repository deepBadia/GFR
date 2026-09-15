"""Reconstruct a training run's output folder from (save_path, config,
timestamp), matching the naming convention in
`src/surmod/core/trainer.py::train_model` exactly. Keep this in sync with
that function if its naming logic ever changes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


def run_dir(save_path: str, config: Dict[str, Any], timestamp: str, fold_id: Optional[int] = None) -> Path:
    """Folder a `core.trainer.train_model(..., timestamp=timestamp)` call
    writes its checkpoint/config/metrics into (fold_id=None: the common case,
    a plain non-tuning training run).
    """
    model_name = config["common"]["model_name"]
    prefix = config["common"].get("save_name") or ""
    data_stem = Path(config["common"]["data_file"]).stem
    base = Path(save_path) / "results" / data_stem / f"{prefix}_{model_name}_{timestamp}"
    if fold_id is not None:
        base = Path(str(base) + "_tuning") / "trials" / f"trial_{fold_id}"
    return base
