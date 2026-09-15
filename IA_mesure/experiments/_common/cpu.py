"""Shared CLI helper to cap PyTorch's CPU thread usage.

PyTorch grabs every core on the machine for its intra-op thread pool by
default. That's fine on a personal laptop, but on a shared cluster node it
means a single `python 03_tune_hyperparams.py` can starve every other job
running on the same machine. `--cpus N` (added to 01/02/03 by
`add_cpu_arg`/`apply_cpu_limit`) caps it -- pass it explicitly, since the
default stays "use everything available" to not silently change behavior
for people running on their own machine.
"""
from __future__ import annotations

import argparse
from typing import Optional


def add_cpu_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cpus",
        type=int,
        default=None,
        help="Cap the number of CPU threads PyTorch uses for this run "
             "(default: all available cores). Set this on a shared "
             "machine/cluster, e.g. --cpus 4.",
    )


def apply_cpu_limit(cpus: Optional[int]) -> None:
    """Must be called before any tensor op (DataLoader/model construction)."""
    if cpus is None:
        return
    import torch

    n = max(1, cpus)
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Only settable once, before any parallel work has started; if
        # something already triggered it, intra-op capping above still helps.
        pass
    print(f"[cpu] Capped PyTorch to {n} thread(s) (--cpus {cpus}).")
