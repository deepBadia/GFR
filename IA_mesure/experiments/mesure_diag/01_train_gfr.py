"""Train the baseline GFR-Net (full pool, no active learning) on mesure_diag.h5.

Usage:
    python 01_train_gfr.py [--epochs N]
"""
import argparse
from pathlib import Path

from surmod import train_model

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs from config.yaml")
    args = parser.parse_args()

    overrides = {}
    if args.epochs is not None:
        overrides["training"] = {"epochs": args.epochs, "patience": args.epochs}

    best_val_loss = train_model(
        experiment="mesure_diag_gfr",
        config_path=str(HERE / "config.yaml"),
        save_path=str(HERE),
        **overrides,
    )
    print(f"\nBest validation loss (GFR-Net baseline, full pool): {best_val_loss:.6e}")
