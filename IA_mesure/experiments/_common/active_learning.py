"""Generic pool-based active-learning loop for GFR-Net style surrogate models.

This wires together pieces that already exist in ``src/surmod`` but were not
previously connected end to end:

* ``surmod.core.data_loader.DataLoader`` tracks which geometries have been
  "labeled" (``labeled_mask``) and can restrict a train/val split to just
  those (``get_data(new_points=...)``).
* ``surmod.models.active_gfr_net.ActiveGFRNet`` scores the remaining pool by
  MC-dropout uncertainty (``uncertainty_score``).
* ``surmod.core.trainer.train_model`` now accepts ``new_points=...`` so it
  actually trains on the labeled subset only (see the fix applied to
  ``src/surmod/core/trainer.py`` -- previously it silently ignored the
  active-learning bookkeeping and always trained on the full pool).

The loop below: start from a small random "seed" set of labeled geometries,
train, evaluate on a held-out test set that is *never* queried, ask the
model which remaining geometries it is least confident about, add the most
uncertain ones to the labeled set, and repeat. The resulting
``metric vs. number of labeled geometries`` curve is exactly what you need
to answer "how few measurements/simulations do we actually need?".
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from surmod.core.data_loader import DataLoader as GFRDataLoader
from surmod.core.trainer import train_model as core_train_model
from surmod.utils.common import load_model_module, evaluate_model, resolve_data_path
from surmod.utils.yaml_handler import load_config

from .metrics import compute_metrics, format_metrics


@dataclass
class ActiveLearningResult:
    history: pd.DataFrame
    history_csv: str
    final_checkpoint: str
    run_info_path: str


def run_active_learning(
    experiment: str,
    config_path: str,
    save_path: str,
    run_id: str = "active",
    seed_frac: float = 0.10,
    query_frac: float = 0.10,
    n_rounds: int = 8,
    epochs_per_round: Optional[int] = None,
    patience_per_round: Optional[int] = None,
    n_mc: int = 20,
    warm_start: bool = True,
    seed: int = 42,
) -> ActiveLearningResult:
    """Run the active-learning loop and return its result.

    Parameters
    ----------
    experiment : str
        Name of the experiment in ``config_path`` to use. Its
        ``common.model_name`` MUST point to a model exposing
        ``uncertainty_score`` (i.e. ``active_gfr_net``).
    config_path : str
        Path to the dataset's ``config.yaml``.
    save_path : str
        Base directory under which ``results/<data_stem>/...`` is created
        (same convention as the rest of the library).
    run_id : str
        Short tag used to name this active-learning run's output folder and
        the per-round timestamps, so repeated runs don't collide.
    seed_frac, query_frac : float
        Fraction of the training *pool* (not the held-out test set) used as
        the initial random seed set, and queried at each round.
    n_rounds : int
        Maximum number of rounds (stops earlier if the pool is exhausted).
    epochs_per_round, patience_per_round : int, optional
        Override ``training.epochs`` / ``training.patience`` for each round
        (rounds are usually trained with far fewer epochs than a single
        full training run). Defaults to the config's own values.
    n_mc : int
        Number of Monte-Carlo dropout forward passes used to score
        uncertainty on the remaining pool.
    warm_start : bool
        If True (default), each round continues training the previous
        round's weights instead of reinitialising from scratch -- much
        faster and generally at least as accurate.
    seed : int
        RNG seed controlling the initial random seed set.
    """
    config = load_config(experiment, config_file=config_path)
    config["common"]["verbose"] = False
    if epochs_per_round is not None:
        config["training"]["epochs"] = epochs_per_round
    if patience_per_round is not None:
        config["training"]["patience"] = patience_per_round
    config["training"].setdefault("patience_start", 0)

    model_name = config["common"]["model_name"]
    model_module = load_model_module(config)
    if not hasattr(model_module.MODEL_CLASS, "uncertainty_score"):
        raise ValueError(
            f"Model '{model_name}' has no `uncertainty_score` method; active learning "
            "requires the Active-GFR-Net model (common.model_name: active_gfr_net)."
        )

    data_path = str(resolve_data_path(config))
    dm = GFRDataLoader(data_path, config)
    n_geom = len(config["dataset"]["X_columns"])
    process_batch = model_module.process_batch

    rng = np.random.default_rng(seed)
    pool = dm.pool_idx.copy()
    rng.shuffle(pool)

    n_seed = max(2, int(round(seed_frac * len(pool))))
    n_query = max(1, int(round(query_frac * len(pool))))

    labeled = list(pool[:n_seed])
    remaining = list(pool[n_seed:])

    prefix = config["common"].get("save_name") or ""
    data_stem = Path(config["common"]["data_file"]).stem
    summary_dir = Path(save_path) / "results" / data_stem / f"{run_id}_active_learning"
    summary_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" ACTIVE LEARNING")
    print(f" Experiment      : {experiment}")
    print(f" Pool size       : {len(pool)} geometries (test set held out separately: {len(dm.test_idx)})")
    print(f" Seed / query    : {n_seed} / {n_query} geometries per round")
    print(f" Max rounds      : {n_rounds}")
    print("=" * 70)

    model = model_module.MODEL_CLASS(config, n_geom=n_geom)
    history = []
    round_idx = 0
    round_dir = None

    while True:
        labeled_arr = np.asarray(sorted(labeled))

        if not warm_start and round_idx > 0:
            model = model_module.MODEL_CLASS(config, n_geom=n_geom)

        round_timestamp = f"{run_id}_r{round_idx:02d}"
        best_val_loss = core_train_model(
            model=model,
            dataloader=dm,
            config=config,
            process_batch=process_batch,
            save_path=save_path,
            new_points=labeled_arr,
            timestamp=round_timestamp,
        )
        round_dir = Path(save_path) / "results" / data_stem / f"{prefix}_{model_name}_{round_timestamp}"

        # Evaluate on the held-out test set, unaffected by `labeled_mask`.
        data = dm.get_data(new_points=labeled_arr)
        row = {
            "round": round_idx,
            "n_labeled_geoms": len(labeled),
            "n_pool_geoms": len(pool),
            "labeled_fraction": len(labeled) / len(pool),
            "best_val_loss": best_val_loss,
        }
        if data.get("X_test") is not None and len(data["X_test"]) > 0:
            results = evaluate_model(model, data, process_batch, config)
            row["weighted_mse"] = results["loss"]
            channel_metrics = compute_metrics(results["pred"], results["true"], dm.get_y_normalizer(), config)
            row.update({f"test_{k}": v for k, v in channel_metrics.items()})
            metric_str = f", test_weighted_mse={results['loss']:.5e} ({format_metrics(channel_metrics)})"
        else:
            metric_str = ""

        history.append(row)
        print(f"[round {round_idx:02d}] {len(labeled):>5d}/{len(pool)} geometries labeled "
              f"-> best_val_loss={best_val_loss:.5e}{metric_str}")

        if not remaining or round_idx + 1 >= n_rounds:
            break

        # --- Query the most uncertain geometries left in the pool ---
        remaining_arr = np.asarray(remaining)
        device = next(model.parameters()).device
        x_pool = torch.as_tensor(dm.X[remaining_arr], dtype=torch.float32, device=device)
        axis_norm_t = torch.as_tensor(dm.axis_norm, dtype=torch.float32, device=device)
        model.eval()
        scores = model.uncertainty_score(x_pool, axis_norm_t, n_mc=n_mc)

        n_take = min(n_query, len(remaining))
        query_local_idx = np.argsort(scores)[::-1][:n_take]
        queried = remaining_arr[query_local_idx]

        queried_set = set(queried.tolist())
        labeled.extend(queried.tolist())
        remaining = [g for g in remaining if g not in queried_set]
        round_idx += 1

    history_df = pd.DataFrame(history)
    history_csv = summary_dir / "history.csv"
    history_df.to_csv(history_csv, index=False)

    final_checkpoint = round_dir / f"{model.name}.pth"
    run_info = {
        "experiment": experiment,
        "run_id": run_id,
        "n_rounds_run": round_idx + 1,
        "final_checkpoint": str(final_checkpoint),
        "history_csv": str(history_csv),
    }
    run_info_path = summary_dir / "run_info.json"
    with open(run_info_path, "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)

    print("=" * 70)
    print(f" Finished after {round_idx + 1} round(s) "
          f"({len(labeled)}/{len(pool)} = {100 * len(labeled) / len(pool):.1f}% of the pool labeled)")
    print(f" History        : {history_csv}")
    print(f" Final checkpoint: {final_checkpoint}")
    print("=" * 70)

    return ActiveLearningResult(
        history=history_df,
        history_csv=str(history_csv),
        final_checkpoint=str(final_checkpoint),
        run_info_path=str(run_info_path),
    )
