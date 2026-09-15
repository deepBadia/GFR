"""Visualize the Optuna hyperparameter search for mesure_couplage.h5.

Reads studies from the shared optuna.db (see 03_tune_hyperparams.py) and
saves static plots to comparison/: optimization history, parameter
importances, parallel coordinates, per-parameter slices, and pruning
curves (which trials got cut short, and when).

For an interactive view instead, install optuna-dashboard and point it at
the same storage (see ../README.md):
    pip install optuna-dashboard
    optuna-dashboard sqlite:///<project_root>/optuna.db

Usage:
    python 05_visualize_tuning.py [--study-name NAME]
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # import the shared `_common` package
from _common.optuna_viz import find_studies, plot_study

DATA_STEM = "mesure_couplage"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study-name", default=None,
                        help="Specific study to plot (default: most recent one found for this dataset)")
    args = parser.parse_args()

    if args.study_name:
        study_name = args.study_name
    else:
        studies = find_studies(DATA_STEM)
        if not studies:
            raise SystemExit(f"No Optuna study found for '{DATA_STEM}' -- run 03_tune_hyperparams.py first.")
        if len(studies) > 1:
            print(f"Found {len(studies)} studies for '{DATA_STEM}', using the most recent one:")
            for s in studies:
                print(f"  {s}")
        study_name = studies[0]

    plot_study(study_name, out_dir=str(HERE / "comparison"))
