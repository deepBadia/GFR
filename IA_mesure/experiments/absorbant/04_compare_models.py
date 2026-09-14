"""Compare the baseline GFR-Net, the active-learning Active-GFR-Net, and the
Optuna-tuned GFR-Net on the same held-out test set of absorbant.h5.

Run `01_train_gfr.py`, `02_train_active_gfr.py` and `03_tune_hyperparams.py`
first (in any order) -- this script evaluates whichever checkpoints it can
find and warns about (but does not fail on) the rest, so you can also use it
to compare just two of the three at a time.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # import the shared `_common` package
from _common.compare import compare_models, find_checkpoint, load_tuning_best

DATA_STEM = "absorbant"
RESULTS_ROOT = HERE / "results" / DATA_STEM

if __name__ == "__main__":
    runs = {}

    try:
        runs["GFR-Net (full pool)"] = find_checkpoint(RESULTS_ROOT, "baseline_GFR_Net_*/GFR-Net.pth")
    except FileNotFoundError as e:
        print(f"[compare] {e} -- run 01_train_gfr.py first")

    al_run_info = RESULTS_ROOT / "active_active_learning" / "run_info.json"
    al_history_csv = None
    if al_run_info.exists():
        info = json.loads(al_run_info.read_text())
        runs["Active-GFR-Net (queried)"] = info["final_checkpoint"]
        al_history_csv = info["history_csv"]
    else:
        print(f"[compare] {al_run_info} not found -- run 02_train_active_gfr.py first")

    try:
        info_path = find_checkpoint(RESULTS_ROOT, "baseline_GFR_Net_*_tuning/best_run_info.json")
        best_ckpt = load_tuning_best(info_path)
        if best_ckpt:
            runs["GFR-Net (tuned)"] = best_ckpt
    except FileNotFoundError as e:
        print(f"[compare] {e} -- run 03_tune_hyperparams.py first")

    if not runs:
        raise SystemExit("No trained checkpoint found at all -- run the training scripts first.")

    compare_models(
        runs=runs,
        out_dir=str(HERE / "comparison"),
        al_history_csv=al_history_csv,
    )
