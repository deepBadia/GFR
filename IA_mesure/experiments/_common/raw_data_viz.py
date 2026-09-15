"""Plot the raw/measured data itself -- not the train/val/test split
(`_common/data_viz.py`), not model predictions (`generate_results`), but
the actual target values in the dataset.

Two views, both generic across all 6 datasets:
  - "cuts" (`plot_slices`): a handful of example geometries, output vs. the
    continuous axis, as line curves -- works for any number of geometry
    features.
  - "map" (`plot_map`): a 2D heatmap of output vs. (geometry, axis). When
    there is a single geometry feature (secteur1, secteur1_amp_phase,
    mesure_couplage, mesure_gain) the y-axis is that feature in physical
    units -- e.g. theta vs frequency for mesure_gain. With several geometry
    features (absorbant, mesure_diag) there is no single physical y-axis,
    so geometries are ordered by their first principal component instead
    (same projection used in `_common/data_viz.py`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .metrics import display_name, effective_output_names


def _denormalized(dm) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_phys, axis_phys, Y_phys): geometries, axis and targets back
    in physical units. `dm.axis` is already physical (only `axis_norm` is
    0-1); `dm.X`/`dm.Y` are normalised in place by DataLoader, so both need
    their normalizer's inverse_transform.
    """
    X_phys = dm.x_normalizer.inverse_transform(dm.X)
    axis_phys = dm.axis
    n_out = dm.Y.shape[-1]
    Y_phys = dm.y_normalizer.inverse_transform(dm.Y.reshape(-1, n_out)).reshape(dm.Y.shape)
    return X_phys, axis_phys, Y_phys


def plot_slices(dm, config: Dict[str, Any], out_dir: str, n_samples: int = 6) -> str:
    """"Coupe" view: `n_samples` example geometries, output vs. axis, one
    subplot per output channel, saved as raw_data_slices.png.
    """
    X_phys, axis_phys, Y_phys = _denormalized(dm)
    n_geoms, n_axis, n_out = Y_phys.shape
    names = effective_output_names(config, n_out)
    axis_name = display_name(config, config.get("dataset", {}).get("axis_column", "axis"))
    X_columns = list(config.get("dataset", {}).get("X_columns", []))
    data_file = config.get("common", {}).get("data_file", "")

    idx = np.linspace(0, n_geoms - 1, min(n_samples, n_geoms), dtype=int)
    cmap = plt.get_cmap("viridis", max(len(idx), 2))

    fig, axes = plt.subplots(1, n_out, figsize=(6.5 * n_out, 5.2), squeeze=False)
    axes = axes[0]
    for c in range(n_out):
        ax = axes[c]
        for i, g in enumerate(idx):
            geom_str = ", ".join(f"{display_name(config, n)}={v:.3g}" for n, v in zip(X_columns, np.atleast_1d(X_phys[g])))
            ax.plot(axis_phys, Y_phys[g, :, c], color=cmap(i), linewidth=1.8, label=geom_str)
        ax.set_xlabel(axis_name)
        ax.set_ylabel(names[c])
        ax.set_title(names[c])
        ax.grid(True, alpha=0.3)

    # One shared legend below the whole figure instead of one per panel: with
    # long geometry-parameter strings (e.g. absorbant's 8 X_columns), a
    # per-axis "loc=best" legend is often wider than its own subplot and
    # visually spills into the neighboring one.
    handles, labels = axes[0].get_legend_handles_labels()
    n_legend_cols = 1 if n_out == 1 else 2
    fig.legend(handles, labels, fontsize=7, loc="upper center",
               bbox_to_anchor=(0.5, 0.0), ncol=n_legend_cols)

    fig.suptitle(f"{data_file} -- example cuts ({len(idx)} geometries)", fontsize=13)
    plt.tight_layout()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    save_path = out_path / "raw_data_slices.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")
    return str(save_path)


def plot_map(dm, config: Dict[str, Any], out_dir: str) -> str:
    """"Carte" view: 2D heatmap of output vs. (geometry, axis), one panel
    per output channel, saved as raw_data_map.png.
    """
    X_phys, axis_phys, Y_phys = _denormalized(dm)
    n_geoms, n_axis, n_out = Y_phys.shape
    names = effective_output_names(config, n_out)
    axis_name = display_name(config, config.get("dataset", {}).get("axis_column", "axis"))
    X_columns = list(config.get("dataset", {}).get("X_columns", []))
    data_file = config.get("common", {}).get("data_file", "")
    n_features = X_phys.shape[1]

    if n_features == 1:
        order = np.argsort(X_phys[:, 0])
        ylabel = display_name(config, X_columns[0]) if X_columns else "x0"
        yvals = X_phys[order, 0]
        map_title = f"{ylabel} x {axis_name}"
    else:
        # No single physical y-axis with several geometry features -- order by
        # first principal component instead (same trick as data_viz.py).
        Xc = dm.X - dm.X.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(Xc, full_matrices=False)
        pc1 = Xc @ vt[0]
        order = np.argsort(pc1)
        ylabel = "geometries sorted by PC1 (a.u.)"
        yvals = np.arange(n_geoms)
        map_title = f"geometries (PC1-sorted) x {axis_name}"

    fig, axes = plt.subplots(1, n_out, figsize=(7 * n_out, 5.5), squeeze=False)
    axes = axes[0]
    for c in range(n_out):
        ax = axes[c]
        im = ax.pcolormesh(axis_phys, yvals, Y_phys[order, :, c], shading="auto", cmap="viridis")
        ax.set_xlabel(axis_name)
        ax.set_ylabel(ylabel)
        ax.set_title(names[c])
        fig.colorbar(im, ax=ax, label=names[c])

    fig.suptitle(f"{data_file} -- map ({map_title})", fontsize=13)
    plt.tight_layout()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    save_path = out_path / "raw_data_map.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")
    return str(save_path)
