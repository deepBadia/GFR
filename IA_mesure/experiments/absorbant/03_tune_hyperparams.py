"""Optuna hyperparameter search for the baseline GFR-Net on absorbant.h5.

Tunes hidden_dim / n_fourier / n_layers / dropout / lr / w_decay / batch_size
(see the `tuning:` block in config.yaml for the search space and edit it to
taste). Stops at whichever of `--n-trials` / `--time-hours` is hit first.
`--epochs` / `--patience-start` let you run a fast smoke test (e.g. a couple
of trials with few epochs) without editing config.yaml.

Usage:
    python 03_tune_hyperparams.py [--n-trials N] [--time-hours H] [--epochs N] [--cpus N]
"""
import argparse
import sys
from pathlib import Path

import optuna
from surmod.core.tuner import run_tuning

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # import the shared `_common` package
from _common.cpu import add_cpu_arg, apply_cpu_limit

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--time-hours", type=float, default=3.0)
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs per trial")
    parser.add_argument("--patience-start", type=int, default=None, help="Override training.patience_start per trial")
    add_cpu_arg(parser)
    args = parser.parse_args()
    apply_cpu_limit(args.cpus)

    overrides = {}
    if args.epochs is not None:
        overrides["training"] = {"epochs": args.epochs, "patience": args.epochs}
    if args.patience_start is not None:
        overrides.setdefault("training", {})["patience_start"] = args.patience_start

    study = run_tuning(
        config_name="absorbant_gfr",
        config_file=str(HERE / "config.yaml"),
        save_path=str(HERE),
        n_trials=args.n_trials,
        time_hours=args.time_hours,
        config_overrides=overrides or None,
    )
    completed = any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    if completed:
        print(f"\nBest trial: #{study.best_trial.number}  val_loss={study.best_trial.value:.6e}")
        for k, v in study.best_trial.params.items():
            print(f"  {k}: {v}")
    else:
        print("\nNo trial completed successfully -- check the logs / results/.../trials/trial_*/ folders.")
