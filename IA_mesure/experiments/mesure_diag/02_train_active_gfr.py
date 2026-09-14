"""Train Active-GFR-Net with a pool-based active-learning loop on mesure_diag.h5.

Starts from a small random seed set of labeled geometries, trains, scores
the remaining pool by MC-dropout uncertainty, queries the most uncertain
ones, and repeats -- recording accuracy vs. number of labeled geometries.
Run `04_compare_models.py` afterwards to see how this compares against the
full-pool baseline from `01_train_gfr.py` (the point of active learning here
is to answer "how few of the mesure_diag geometries do we actually need?").

Usage:
    python 02_train_active_gfr.py [--n-rounds N] [--epochs-per-round N]
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # import the shared `_common` package
from _common.active_learning import run_active_learning

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-rounds", type=int, default=8)
    parser.add_argument("--epochs-per-round", type=int, default=150)
    parser.add_argument("--patience-per-round", type=int, default=150)
    parser.add_argument("--seed-frac", type=float, default=0.15)
    parser.add_argument("--query-frac", type=float, default=0.1)
    args = parser.parse_args()

    result = run_active_learning(
        experiment="mesure_diag_active_gfr",
        config_path=str(HERE / "config.yaml"),
        save_path=str(HERE),
        run_id="active",
        seed_frac=args.seed_frac,
        query_frac=args.query_frac,
        n_rounds=args.n_rounds,
        epochs_per_round=args.epochs_per_round,
        patience_per_round=args.patience_per_round,
    )
    print(f"\nHistory   : {result.history_csv}")
    print(f"Checkpoint: {result.final_checkpoint}")
