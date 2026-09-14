"""Optuna hyperparameter search for the baseline GFR-Net on mesure_couplage.h5.

Tunes hidden_dim / n_fourier / n_layers / dropout / lr / w_decay / batch_size
(see the `tuning:` block in config.yaml for the search space and edit it to
taste). Stops at whichever of `--n-trials` / `--time-hours` is hit first.
`--epochs` / `--patience-start` let you run a fast smoke test (e.g. a couple
of trials with few epochs) without editing config.yaml.

Usage:
    python 03_tune_hyperparams.py [--n-trials N] [--time-hours H] [--epochs N]
"""
import argparse
from pathlib import Path

from surmod.core.tuner import run_tuning

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--time-hours", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs per trial")
    parser.add_argument("--patience-start", type=int, default=None, help="Override training.patience_start per trial")
    args = parser.parse_args()

    overrides = {}
    if args.epochs is not None:
        overrides["training"] = {"epochs": args.epochs, "patience": args.epochs}
    if args.patience_start is not None:
        overrides.setdefault("training", {})["patience_start"] = args.patience_start

    study = run_tuning(
        config_name="mesure_couplage_gfr",
        config_file=str(HERE / "config.yaml"),
        save_path=str(HERE),
        n_trials=args.n_trials,
        time_hours=args.time_hours,
        config_overrides=overrides or None,
    )
    print(f"\nBest trial: #{study.best_trial.number}  val_loss={study.best_trial.value:.6e}")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")
