"""Train the baseline GFR-Net (full pool, no active learning) on mesure_couplage.h5.

After training, calls `surmod.generate_results()` on the run folder to
produce the standard plots/report (predictions vs. true, amplitude/phase
if applicable, train/val convergence curve, training_report.txt) under
results/.../plots/. Pass --skip-report to skip that.

Usage:
    python 01_train_gfr.py [--epochs N] [--cpus N] [--skip-report]
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

from surmod import train_model, generate_results
from surmod.utils.yaml_handler import load_config

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # import the shared `_common` package
from _common.cpu import add_cpu_arg, apply_cpu_limit
from _common.paths import run_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs from config.yaml")
    parser.add_argument("--skip-report", action="store_true", help="Skip generate_results() after training")
    add_cpu_arg(parser)
    args = parser.parse_args()
    apply_cpu_limit(args.cpus)

    overrides = {}
    if args.epochs is not None:
        overrides["training"] = {"epochs": args.epochs, "patience": args.epochs}

    # Pick the timestamp ourselves so we know exactly which results/ folder
    # this run wrote to afterwards (train_model() only returns the loss).
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    best_val_loss = train_model(
        experiment="mesure_couplage_gfr",
        config_path=str(HERE / "config.yaml"),
        save_path=str(HERE),
        timestamp=timestamp,
        **overrides,
    )
    print(f"\nBest validation loss (GFR-Net baseline, full pool): {best_val_loss:.6e}")

    if not args.skip_report:
        config = load_config("mesure_couplage_gfr", config_file=str(HERE / "config.yaml"))
        out_dir = run_dir(str(HERE), config, timestamp)
        print(f"\nGenerating plots/report (generate_results) -> {out_dir / 'plots'}")
        generate_results(out_dir)
