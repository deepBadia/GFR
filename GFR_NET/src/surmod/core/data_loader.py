"""
HDF5 DataLoader for GFR-Net / Active Learning experiments.

======================== CONFIGURATION PARAMETERS ========================

All configuration is read from the YAML config under two sections:
  - config["dataset"]
  - config["common"]

--- config["dataset"] ---
X_columns (list[str]) [REQUIRED]
    List of column names to use as input features (geometry parameters).

Y_columns (list[str]) [REQUIRED]
    List of column names to use as model targets.

axis_column (str) [REQUIRED]
    Name of the column representing the frequency / sweep axis.

column_filters (dict, default={})
    Filter rows before processing.

weights_type (str, default="equal")
    Weighting strategy (currently only "equal" is supported).

size (int, default=None)
    Number of geometries used for train + val (rest go to test).

val_split (float, default=0.2)
    Validation split ratio.

use_grouping (bool, default=False)
    Whether to split at group level.

group_columns (list[str], default=[])
    Columns defining groups when use_grouping=True.

--- config["common"] ---
complex_format (str, default=None)
    Set to "polar" for gain/phase representation.

seed (int, default=42)
    Random seed.
"""

from __future__ import annotations

import h5py
import numpy as np
import torch
from pathlib import Path

from surmod.utils.common import Normalizer


class DataLoader:
    """Loads data from HDF5 and handles standard + active learning splits."""

    def __init__(self, h5_path: str, config: dict):
        self.config = config
        self.h5_path = Path(h5_path)

        self._load_raw_data()
        self._process_data()

        self.pool_idx = None
        self.test_idx = None
        self.labeled_mask = np.zeros((self.n_geoms, self.n_axis), dtype=bool)

        self._setup_initial_split()

    # ====================== PRIVATE METHODS ======================

    def _load_raw_data(self):
        with h5py.File(self.h5_path, "r") as f:
            features = f["features"][:]
            raw_names = f["features"].attrs["column_names"]

        if isinstance(raw_names, np.ndarray):
            raw_names = raw_names.tolist()
        if raw_names and isinstance(raw_names[0], (bytes, np.bytes_)):
            self.column_names = [n.decode("utf-8").strip() for n in raw_names]
        else:
            self.column_names = [str(n).strip() for n in raw_names]

        filters = self.config["dataset"].get("column_filters", {})
        if filters:
            features = self._apply_filters(features, filters)

        self.features = features

    def _apply_filters(self, features: np.ndarray, filters: dict) -> np.ndarray:
        mask = np.ones(len(features), dtype=bool)
        for col, allowed in filters.items():
            if col not in self.column_names:
                raise ValueError(f"Filter column '{col}' not found.")
            idx = self.column_names.index(col)
            if isinstance(allowed, (int, float, str)):
                allowed = [allowed]
            mask &= np.isin(features[:, idx], allowed)
        return features[mask]

    def _process_data(self):
        ds_cfg = self.config["dataset"]
        common_cfg = self.config["common"]

        X_names = ds_cfg["X_columns"]
        Y_names = ds_cfg["Y_columns"]
        axis_name = ds_cfg["axis_column"]

        self.x_idx = [self.column_names.index(name) for name in X_names]
        self.y_idx = [self.column_names.index(name) for name in Y_names]
        self.axis_idx = self.column_names.index(axis_name)

        X_raw = self.features[:, self.x_idx].astype(np.float32)
        Y_raw = self.features[:, self.y_idx].astype(np.float32)
        axis_raw = self.features[:, self.axis_idx].astype(np.float32)

        self._reshape_to_grid(X_raw, Y_raw, axis_raw)

        if common_cfg.get("complex_format") == "polar":
            self.Y = self._to_polar(self.Y)

        weights_type = ds_cfg.get("weights_type", "equal")
        self.W = self._compute_weights(self.Y, weights_type)

        self.x_normalizer = Normalizer().fit(self.X)
        self.y_normalizer = Normalizer().fit(self.Y.reshape(-1, self.Y.shape[-1]))

        self.X = self.x_normalizer.transform(self.X).astype(np.float32)
        self.Y = self.y_normalizer.transform(
            self.Y.reshape(-1, self.Y.shape[-1])
        ).reshape(self.Y.shape).astype(np.float32)

        axis_min, axis_max = float(self.axis.min()), float(self.axis.max())
        self.axis_norm = ((self.axis - axis_min) / (axis_max - axis_min)).astype(np.float32)

    def _reshape_to_grid(self, X_raw, Y_raw, axis_raw):
        unique_X, geom_inv = np.unique(X_raw, axis=0, return_inverse=True)
        unique_ax, axis_inv = np.unique(axis_raw, return_inverse=True)

        n_geoms = len(unique_X)
        n_axis = len(unique_ax)

        Y = np.zeros((n_geoms, n_axis, Y_raw.shape[1]), dtype=np.float32)
        Y[geom_inv, axis_inv] = Y_raw

        self.X = unique_X
        self.Y = Y
        self.axis = unique_ax
        self.n_geoms = n_geoms
        self.n_axis = n_axis

    def _to_polar(self, Y: np.ndarray) -> np.ndarray:
        S = Y[..., 0] + 1j * Y[..., 1]
        gain_dB = 20.0 * np.log10(np.abs(S) + 1e-12)
        phase = np.angle(S)
        return np.stack([gain_dB, np.sin(phase), np.cos(phase)], axis=-1).astype(np.float32)

    def _compute_weights(self, Y: np.ndarray, weights_type: str) -> np.ndarray:
        n_out = Y.shape[-1]
        if weights_type == "equal":
            W = np.ones_like(Y, dtype=np.float32)
            for c in range(n_out):
                W[..., c] = 1.0 / (np.sqrt(np.mean(Y[..., c] ** 2)) + 1e-8)
            return W
        return np.ones_like(Y, dtype=np.float32)

    def _setup_initial_split(self):
        size = self.config["dataset"].get("size", None)
        if size is not None and size < self.n_geoms:
            self.pool_idx = np.linspace(0, self.n_geoms - 1, size, dtype=int)
        else:
            self.pool_idx = np.arange(self.n_geoms)

        self.test_idx = np.setdiff1d(np.arange(self.n_geoms), self.pool_idx)
        self.labeled_mask = np.zeros((self.n_geoms, self.n_axis), dtype=bool)

    def _add_points_to_seen(self, new_points: np.ndarray):
        for g, a in new_points:
            if g in self.pool_idx:
                self.labeled_mask[g, a] = True

    def _print_state(self):
        seen_geoms = np.sum(np.any(self.labeled_mask, axis=1))
        labeled_points = np.sum(self.labeled_mask)

        print("\n" + "=" * 60)
        print(" DATALOADER STATE")
        print("=" * 60)
        print(f" Total Geometries      : {self.n_geoms}")
        print(f" Frequency Points      : {self.n_axis}")
        print(f" Pool Size (Train+Val) : {len(self.pool_idx)}")
        print(f" Test Set Size         : {len(self.test_idx)}")
        print(f" Seen Geometries       : {seen_geoms}")
        print(f" Total Labeled Points  : {labeled_points}")
        print(f" Grouping Enabled      : {self.config['dataset'].get('use_grouping', False)}")
        print("=" * 60 + "\n")

    # ====================== PUBLIC API ======================

    def get_data(self, new_points: np.ndarray | list | None = None, verbose: bool = False) -> dict:
        if new_points is not None:
            new_points = np.asarray(new_points)

            if new_points.ndim == 1:
                geom_indices = new_points
                axis_indices = np.arange(self.n_axis)
                full_points = np.array([[g, a] for g in geom_indices for a in axis_indices])
                self._add_points_to_seen(full_points)
            elif new_points.ndim == 2 and new_points.shape[1] == 2:
                self._add_points_to_seen(new_points)
            else:
                raise ValueError("new_points must be 1D (geometries) or 2D (N,2)")

        if verbose:
            self._print_state()

        use_seen = new_points is not None
        return self._build_split(use_seen=use_seen)

    def get_pool_values(self) -> list[dict]:
        return [
            {"index": int(idx), "X": self.X[idx].copy()}
            for idx in self.pool_idx
        ]

    def get_x_normalizer(self):
        """Return the normalizer used for input features (X)."""
        return self.x_normalizer

    def get_y_normalizer(self):
        """Return the normalizer used for targets (Y)."""
        return self.y_normalizer

    def save_state(self, filepath: str):
        np.savez_compressed(
            filepath,
            pool_idx=self.pool_idx,
            test_idx=self.test_idx,
            labeled_mask=self.labeled_mask
        )
        print(f"State saved to: {filepath}")

    def load_state(self, filepath: str):
        data = np.load(filepath)
        self.pool_idx = data["pool_idx"]
        self.test_idx = data["test_idx"]
        self.labeled_mask = data["labeled_mask"]
        print(f"State loaded from: {filepath}")

    # ====================== INTERNAL ======================

    def _build_split(self, use_seen: bool = False) -> dict:
        ds_cfg = self.config["dataset"]
        val_split = ds_cfg.get("val_split", 0.2)
        use_grouping = ds_cfg.get("use_grouping", False)
        group_columns = ds_cfg.get("group_columns", [])

        if use_seen:
            seen_geoms = np.where(np.any(self.labeled_mask, axis=1))[0]
            active_geoms = np.intersect1d(self.pool_idx, seen_geoms)
        else:
            active_geoms = self.pool_idx.copy()

        if len(active_geoms) == 0:
            raise ValueError("No points available to split.")

        if use_grouping and len(group_columns) > 0:
            train_geoms, val_geoms = self._group_split(active_geoms, val_split, group_columns)
        else:
            n_val = max(1, int(len(active_geoms) * val_split))
            rng = np.random.default_rng(self.config["common"].get("seed", 42))
            perm = rng.permutation(len(active_geoms))
            val_geoms = active_geoms[perm[:n_val]]
            train_geoms = active_geoms[perm[n_val:]]

        if use_seen:
            axis_mask_train = self.labeled_mask[train_geoms]
            axis_mask_val = self.labeled_mask[val_geoms]
        else:
            axis_mask_train = np.ones((len(train_geoms), self.n_axis), dtype=bool)
            axis_mask_val = np.ones((len(val_geoms), self.n_axis), dtype=bool)

        data = {
            "X_train": torch.tensor(self.X[train_geoms], dtype=torch.float32),
            "Y_train": torch.tensor(self.Y[train_geoms], dtype=torch.float32),
            "W_train": torch.tensor(self.W[train_geoms], dtype=torch.float32),
            "axis_mask_train": torch.tensor(axis_mask_train, dtype=torch.bool),

            "X_val": torch.tensor(self.X[val_geoms], dtype=torch.float32),
            "Y_val": torch.tensor(self.Y[val_geoms], dtype=torch.float32),
            "W_val": torch.tensor(self.W[val_geoms], dtype=torch.float32),
            "axis_mask_val": torch.tensor(axis_mask_val, dtype=torch.bool),

            "axis": torch.tensor(self.axis, dtype=torch.float32),
            "axis_norm": torch.tensor(self.axis_norm, dtype=torch.float32),

            "X_test": None,
            "Y_test": None,
            "W_test": None,
        }

        if len(self.test_idx) > 0:
            data["X_test"] = torch.tensor(self.X[self.test_idx], dtype=torch.float32)
            data["Y_test"] = torch.tensor(self.Y[self.test_idx], dtype=torch.float32)
            data["W_test"] = torch.tensor(self.W[self.test_idx], dtype=torch.float32)

        return data

    def _group_split(self, active_geoms, val_split, group_columns):
        X_cols = list(self.config["dataset"].get("X_columns", []))
        g_idx = [X_cols.index(c) for c in group_columns if c in X_cols]

        if not g_idx:
            n_val = max(1, int(len(active_geoms) * val_split))
            rng = np.random.default_rng(self.config["common"].get("seed", 42))
            perm = rng.permutation(len(active_geoms))
            return active_geoms[perm[n_val:]], active_geoms[perm[:n_val]]

        groups = self.X[active_geoms][:, g_idx]
        _, inv = np.unique(groups, axis=0, return_inverse=True)
        grp_ids = np.unique(inv)

        rng = np.random.default_rng(self.config["common"].get("seed", 42))
        rng.shuffle(grp_ids)

        n_val_grps = max(1, int(len(grp_ids) * val_split))
        val_grps = set(grp_ids[:n_val_grps])

        val_mask = np.isin(inv, list(val_grps))
        val_geoms = active_geoms[val_mask]
        train_geoms = active_geoms[~val_mask]

        return train_geoms, val_geoms