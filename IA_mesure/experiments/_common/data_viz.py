"""Visualize a dataset's geometries/axis and which points went to
train / validation / test.

Works generically across all 6 datasets regardless of how many geometry
(X) columns they have:
  - 1 or 2 X columns  -> plotted directly (physically interpretable)
  - 3+ X columns      -> projected to 2D via PCA (numpy SVD, no sklearn dep)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

SPLIT_COLORS = {"train": "#1f77b4", "val": "#2ca02c", "test": "#d62728"}  # blue / green / red


def _pca_2d(X: np.ndarray) -> np.ndarray:
    """Project X (already normalised) onto its first 2 principal components."""
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    n_components = min(2, vt.shape[0])
    coords = Xc @ vt[:n_components].T
    if n_components < 2:
        coords = np.concatenate([coords, np.zeros((coords.shape[0], 2 - n_components))], axis=1)
    return coords


def plot_data_split(dm, config: Dict[str, Any], out_dir: str) -> str:
    """Save `data_split_overview.png`: geometry-space scatter (left) + a
    (geometry x axis) coverage grid (right), both colored by train/val/test.

    Parameters
    ----------
    dm : surmod.core.data_loader.DataLoader
        Already constructed for this dataset/config.
    config : dict
        The resolved experiment config (for column names / data_file / title).
    out_dir : str
        Directory to save the plot into (created if needed).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = dm.get_data()  # standard (non active-learning) split: full pool -> train/val, test held out
    train_geoms = np.asarray(data["train_geoms"])
    val_geoms = np.asarray(data["val_geoms"])
    test_geoms = np.asarray(data["test_geoms"])

    n_geoms = dm.n_geoms
    n_axis = dm.n_axis
    X_columns = list(config.get("dataset", {}).get("X_columns", []))
    axis_name = config.get("dataset", {}).get("axis_column", "axis")
    data_file = config.get("common", {}).get("data_file", "")

    X = dm.X  # normalised geometry features, shape (n_geoms, n_features)
    n_features = X.shape[1]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # ---- Panel A: geometry-space scatter ----
    ax = axes[0]
    if n_features == 1:
        coords = np.stack([X[:, 0], np.zeros(n_geoms)], axis=1)
        xlabel = X_columns[0] if X_columns else "x0"
        ylabel = ""
        ax.set_yticks([])
        title_suffix = ""
    elif n_features == 2:
        coords = X
        xlabel = X_columns[0] if len(X_columns) > 0 else "x0"
        ylabel = X_columns[1] if len(X_columns) > 1 else "x1"
        title_suffix = ""
    else:
        coords = _pca_2d(X)
        xlabel, ylabel = "PCA 1", "PCA 2"
        title_suffix = " (PCA projection)"

    for label, idx in [("train", train_geoms), ("val", val_geoms), ("test", test_geoms)]:
        if len(idx) == 0:
            continue
        ax.scatter(coords[idx, 0], coords[idx, 1], s=24, alpha=0.75,
                   color=SPLIT_COLORS[label], label=f"{label} ({len(idx)})", edgecolors="none")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Geometries -- {n_features} feature{'s' if n_features != 1 else ''}{title_suffix}")
    ax.legend(loc="best", fontsize=9, title="split (n geometries)")
    ax.grid(True, alpha=0.3)

    # ---- Panel B: coverage grid (geometry x axis), colored by split ----
    ax = axes[1]
    grid = np.zeros((n_geoms, n_axis), dtype=np.uint8)
    grid[val_geoms, :] = 1
    grid[test_geoms, :] = 2
    # (train stays 0; any geometry not in pool/test at all would too, but every
    # geometry is always in exactly one of train/val/test by construction)

    sort_key = X[:, 0]
    order = np.argsort(sort_key)
    cmap = ListedColormap([SPLIT_COLORS["train"], SPLIT_COLORS["val"], SPLIT_COLORS["test"]])
    ax.imshow(grid[order].T, aspect="auto", cmap=cmap, vmin=0, vmax=2,
             interpolation="nearest", origin="lower")
    ax.set_xlabel(f"Geometry index (sorted by {X_columns[0] if X_columns else 'x0'})")
    ax.set_ylabel(f"{axis_name} index ({n_axis} points)")
    ax.set_title("Coverage: which (geometry, axis) points are used")
    handles = [plt.Rectangle((0, 0), 1, 1, color=SPLIT_COLORS[k]) for k in ("train", "val", "test")]
    ax.legend(handles, ["train", "val", "test"], loc="upper right", fontsize=9)

    fig.suptitle(f"{data_file} -- train / validation / test split", fontsize=13)
    plt.tight_layout()
    save_path = out_dir / "data_split_overview.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    n_train_pts = len(train_geoms) * n_axis
    n_val_pts = len(val_geoms) * n_axis
    n_test_pts = len(test_geoms) * n_axis
    print(f"Saved: {save_path}")
    print(f"  Geometries : train={len(train_geoms)}  val={len(val_geoms)}  test={len(test_geoms)}  "
          f"(pool={len(train_geoms) + len(val_geoms)} / total={n_geoms})")
    print(f"  Points     : train={n_train_pts}  val={n_val_pts}  test={n_test_pts}  "
          f"({n_axis} axis points per geometry)")
    return str(save_path)
