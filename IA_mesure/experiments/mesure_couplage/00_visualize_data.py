"""Visualize mesure_couplage.h5: which geometries/points go to train, validation
and test (as used by 01_train_gfr.py / 04_compare_models.py).

Produces `data_split_overview.png`: a geometry-space scatter (left) and a
(geometry x axis) coverage grid (right), both colored by split. Run this
before training to sanity-check the split (e.g. that the test set is
actually spread across the geometry space, not just "the last N rows").
"""
from pathlib import Path

from surmod.core.data_loader import DataLoader as GFRDataLoader
from surmod.utils.common import resolve_data_path
from surmod.utils.yaml_handler import load_config

import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # import the shared `_common` package
from _common.data_viz import plot_data_split

if __name__ == "__main__":
    config = load_config("mesure_couplage_gfr", config_file=str(HERE / "config.yaml"))
    dm = GFRDataLoader(str(resolve_data_path(config)), config)
    plot_data_split(dm, config, out_dir=str(HERE / "comparison"))
