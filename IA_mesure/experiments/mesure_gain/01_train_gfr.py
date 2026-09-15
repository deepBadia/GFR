"""Train the baseline GFR-Net (full pool, no active learning) on mesure_gain.h5.

Usage:
    python 01_train_gfr.py [--epochs N] [--cpus N]
"""
import argparse
import sys
from pathlib import Path

from surmod import train_model

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # import the shared `_common` package
from _common.cpu import add_cpu_arg, apply_cpu_limit

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs from config.yaml")
    add_cpu_arg(parser)
    args = parser.parse_args()
    apply_cpu_limit(args.cpus)

    overrides = {}
    if args.epochs is not None:
        overrides["training"] = {"epochs": args.epochs, "patience": args.epochs}

    best_val_loss = train_model(
        experiment="mesure_gain_gfr",
        config_path=str(HERE / "config.yaml"),
        save_path=str(HERE),
        **overrides,
    )
    print(f"\nBest validation loss (GFR-Net baseline, full pool): {best_val_loss:.6e}")
