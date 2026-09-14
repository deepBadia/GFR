#!/usr/bin/env python3
"""Orchestrator: run the pipeline for every dataset under IA_mesure/experiments/.

For each dataset folder (auto-discovered: any subfolder containing a
config.yaml), runs the requested steps in order:

    train   -> 01_train_gfr.py            (baseline GFR-Net, full pool)
    active  -> 02_train_active_gfr.py      (Active-GFR-Net, active-learning loop)
    tune    -> 03_tune_hyperparams.py      (Optuna search -- NOT run by default,
                                             it is the slowest step; opt in explicitly)
    compare -> 04_compare_models.py        (evaluates whatever checkpoints exist)

Each step runs as its own subprocess (so it fails independently and doesn't
take the whole run down), with stdout/stderr captured to a log file per
(dataset, step). A summary table and a combined comparison CSV across all
datasets are printed/written at the end.

Examples
--------
    # Everything except hyperparameter tuning, all 6 datasets, default budgets
    # from each config.yaml (can take a long time -- see each README.md)
    python run_all.py

    # Just two datasets
    python run_all.py --datasets secteur1 mesure_couplage

    # Include the Optuna search too
    python run_all.py --only train active tune compare

    # Quick sanity check with tiny epoch/round/trial counts before a real run
    python run_all.py --quick

    # Re-run comparisons only (e.g. after training finished separately)
    python run_all.py --only compare

    # Run datasets in parallel (careful: N datasets x training at once)
    python run_all.py --jobs 3
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

HERE = Path(__file__).resolve().parent

STEPS = ["train", "active", "tune", "compare"]
SCRIPT_FOR_STEP = {
    "train": "01_train_gfr.py",
    "active": "02_train_active_gfr.py",
    "tune": "03_tune_hyperparams.py",
    "compare": "04_compare_models.py",
}
DEFAULT_STEPS = ["train", "active", "compare"]  # "tune" is opt-in (slowest step)

# Applied when --quick is passed: small enough to finish in seconds/minutes per
# dataset just to confirm nothing is broken -- NOT meaningful results.
QUICK_ARGS = {
    "train": ["--epochs", "20"],
    "active": ["--n-rounds", "3", "--epochs-per-round", "30", "--patience-per-round", "30"],
    "tune": ["--n-trials", "3", "--time-hours", "0.25", "--epochs", "30"],
    "compare": [],
}


def discover_datasets() -> List[str]:
    return sorted(p.parent.name for p in HERE.glob("*/config.yaml"))


def run_step(dataset: str, step: str, quick: bool, log_dir: Path, env: Optional[dict] = None) -> Tuple[str, str, bool, float]:
    script = SCRIPT_FOR_STEP[step]
    dataset_dir = HERE / dataset
    cmd = [sys.executable, script] + (QUICK_ARGS[step] if quick else [])

    log_path = log_dir / f"{dataset}__{step}.log"
    print(f"[{dataset}] {step:8s} -> {script}  (log: {log_path.name})", flush=True)

    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"$ cd {dataset_dir}\n$ {' '.join(cmd)}\n\n")
        log_file.flush()
        result = subprocess.run(cmd, cwd=dataset_dir, stdout=log_file, stderr=subprocess.STDOUT, env=env)
    elapsed = time.time() - t0

    ok = result.returncode == 0
    status = "OK" if ok else f"FAILED (exit {result.returncode})"
    print(f"[{dataset}] {step:8s} {status} in {elapsed / 60:.1f} min", flush=True)
    if not ok:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
        print(f"  --- last lines of {log_path.name} ---")
        for line in tail:
            print("  " + line)
        print("  " + "-" * 40)
    return dataset, step, ok, elapsed


def run_dataset_chain(dataset: str, steps: List[str], quick: bool, log_dir: Path, stop_on_error: bool,
                       env: Optional[dict] = None):
    """Run `steps` for one dataset in order; stop early on failure if requested."""
    results = []
    for step in steps:
        result = run_step(dataset, step, quick, log_dir, env=env)
        results.append(result)
        if not result[2] and stop_on_error:
            break
    return results


def build_combined_summary(datasets: List[str], out_path: Path) -> Optional[pd.DataFrame]:
    """Concatenate every dataset's comparison/comparison_summary.csv into one file."""
    frames = []
    for dataset in datasets:
        csv_path = HERE / dataset / "comparison" / "comparison_summary.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        df.insert(0, "dataset", dataset)
        frames.append(df)

    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined.to_csv(out_path, index=False)
    return combined


def print_summary(results: List[Tuple[str, str, bool, float]], total_elapsed: float, combined_csv: Optional[Path]):
    print("\n" + "=" * 72)
    print(" SUMMARY")
    print("=" * 72)
    for dataset, step, ok, elapsed in results:
        mark = "OK  " if ok else "FAIL"
        print(f" [{mark}] {dataset:<20s} {step:<8s} {elapsed / 60:6.1f} min")
    n_fail = sum(1 for *_, ok, _ in results if not ok)
    print("-" * 72)
    print(f" Total: {len(results)} step(s), {n_fail} failure(s), {total_elapsed / 60:.1f} min elapsed")
    if combined_csv is not None:
        print(f" Combined comparison table: {combined_csv}")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Subset of dataset folders to run (default: all found under experiments/)")
    parser.add_argument("--only", nargs="+", choices=STEPS, default=None,
                        help=f"Only run these steps (default: {' '.join(DEFAULT_STEPS)})")
    parser.add_argument("--skip", nargs="+", choices=STEPS, default=None,
                        help="Steps to skip, applied after --only")
    parser.add_argument("--quick", action="store_true",
                        help="Use tiny epoch/round/trial counts -- sanity check only, not real results")
    parser.add_argument("--stop-on-error", action="store_true",
                        help="Stop a dataset's chain (and, without --jobs, the whole run) at its first failed step")
    parser.add_argument("--jobs", type=int, default=1,
                        help="Number of datasets to run concurrently as separate processes (default: 1, sequential)")
    parser.add_argument("--log-dir", default=None,
                        help="Where to write per-step logs (default: run_all_logs/<timestamp>/)")
    args = parser.parse_args()

    available = discover_datasets()
    if not available:
        sys.exit("No dataset folder with a config.yaml found under " + str(HERE))

    datasets = args.datasets or available
    unknown = [d for d in datasets if d not in available]
    if unknown:
        sys.exit(
            f"Unknown dataset folder(s): {', '.join(unknown)}\n"
            f"Available: {', '.join(available)}"
        )

    steps = args.only or DEFAULT_STEPS
    if args.skip:
        steps = [s for s in steps if s not in args.skip]
    if not steps:
        sys.exit("Nothing to do: all requested steps were skipped.")

    log_dir = Path(args.log_dir) if args.log_dir else HERE / "run_all_logs" / time.strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Datasets : {', '.join(datasets)}")
    print(f"Steps    : {', '.join(steps)}" + ("  (tune skipped by default -- use --only ... tune to include it)"
                                               if "tune" not in steps else ""))
    print(f"Jobs     : {args.jobs}")
    print(f"Logs     : {log_dir}")
    if args.quick:
        print("Mode     : QUICK (sanity check only, NOT meaningful results)")

    # With several datasets training at once, PyTorch's default intra-op
    # thread pool (one per process, sized to all CPU cores) causes heavy
    # oversubscription (jobs x cores threads competing for `cores` cores).
    # Cap each subprocess to a fair share of the machine instead.
    env = None
    if args.jobs > 1:
        threads_per_job = max(1, (os.cpu_count() or args.jobs) // args.jobs)
        env = {**os.environ, "OMP_NUM_THREADS": str(threads_per_job), "MKL_NUM_THREADS": str(threads_per_job)}
        print(f"Threads  : capped to {threads_per_job} per job (OMP/MKL_NUM_THREADS) to avoid CPU oversubscription")
    print()

    t_start = time.time()
    all_results: List[Tuple[str, str, bool, float]] = []

    if args.jobs <= 1:
        for dataset in datasets:
            all_results.extend(run_dataset_chain(dataset, steps, args.quick, log_dir, args.stop_on_error, env=env))
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {
                pool.submit(run_dataset_chain, dataset, steps, args.quick, log_dir, args.stop_on_error, env): dataset
                for dataset in datasets
            }
            for future in as_completed(futures):
                all_results.extend(future.result())

    combined_csv = log_dir / "all_datasets_comparison.csv"
    combined = build_combined_summary(datasets, combined_csv)
    if combined is None:
        combined_csv = None
    elif "compare" not in steps:
        # comparison_summary.csv files predate this run (e.g. --only train) --
        # still useful to aggregate, but don't claim it reflects this run alone.
        print(f"\nNote: aggregated existing comparison_summary.csv files (this run did not include 'compare').")

    print_summary(all_results, time.time() - t_start, combined_csv)
    sys.exit(0 if all(ok for *_, ok, _ in all_results) else 1)


if __name__ == "__main__":
    main()
