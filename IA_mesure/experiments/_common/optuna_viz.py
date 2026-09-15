"""Visualize an Optuna hyperparameter search (see 03_tune_hyperparams.py).

Two ways to look at a tuning run:
  1. Static plots (this module, used by 05_visualize_tuning.py) -- no extra
     dependency beyond matplotlib+optuna, works headless/offline.
  2. `optuna-dashboard` -- an interactive web UI over the same SQLite
     storage, better for exploring a run live (see the README for install
     instructions).

Both read the same storage: `sqlite:///<project_root>/optuna.db`, shared by
every dataset (studies are told apart by name -- see `find_studies`).
"""
from __future__ import annotations

from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import optuna.visualization.matplotlib as ovm
from pathlib import Path

from surmod.utils.common import get_project_root

optuna.logging.set_verbosity(optuna.logging.WARNING)

# name -> plotting function. Each can fail on a study that doesn't have
# enough data yet for that particular plot (e.g. plot_intermediate_values
# needs at least one trial that called trial.report()) -- callers should
# expect and tolerate that.
PLOTS = {
    "optuna_history.png": ovm.plot_optimization_history,
    "optuna_importances.png": ovm.plot_param_importances,
    "optuna_parallel_coordinate.png": ovm.plot_parallel_coordinate,
    "optuna_slice.png": ovm.plot_slice,
    "optuna_pruning.png": ovm.plot_intermediate_values,
}


def storage_url() -> str:
    return f"sqlite:///{get_project_root()}/optuna.db"


def find_studies(data_stem: str) -> List[str]:
    """Names of every study for this dataset in the shared optuna.db,
    most recently started first. `03_tune_hyperparams.py` names studies
    "<model_name>_<data_stem>_..._<timestamp>", so matching on
    "_<data_stem>_" reliably identifies this dataset's studies regardless
    of how many times tuning has been (re-)run.
    """
    summaries = optuna.study.get_all_study_summaries(storage=storage_url())
    needle = f"_{data_stem}_"
    matches = [s for s in summaries if needle in s.study_name]
    matches.sort(key=lambda s: s.datetime_start or 0, reverse=True)
    return [s.study_name for s in matches]


def _save_fig(ax_or_axes, path: Path) -> None:
    axes = np.atleast_1d(ax_or_axes).ravel()
    fig = axes[0].get_figure()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_study(study_name: str, out_dir: str) -> List[str]:
    """Save every available plot for `study_name` into `out_dir`.

    Returns the list of files actually written (a plot that isn't
    meaningful yet for this study, e.g. too few trials, is skipped with a
    printed reason rather than raising).
    """
    study = optuna.load_study(study_name=study_name, storage=storage_url())
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    states = [t.state.name for t in study.trials]
    n_complete = states.count("COMPLETE")
    n_pruned = states.count("PRUNED")
    n_failed = states.count("FAIL")
    print(f"Study '{study_name}': {len(study.trials)} trial(s) "
          f"({n_complete} complete, {n_pruned} pruned, {n_failed} failed)")

    saved = []
    for fname, fn in PLOTS.items():
        try:
            ax = fn(study)
        except (ValueError, RuntimeError, ZeroDivisionError) as e:
            print(f"  [skip] {fname}: {e}")
            continue
        path = out_path / fname
        _save_fig(ax, path)
        saved.append(str(path))
        print(f"  Saved: {path}")

    if n_complete > 0:
        best = study.best_trial
        print(f"  Best trial so far: #{best.number}  value={best.value:.6e}")
        for k, v in best.params.items():
            print(f"    {k}: {v}")

    return saved
