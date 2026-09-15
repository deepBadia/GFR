"""Plot the raw data of mesure_couplage.h5 itself -- NOT the train/val/test split
(that's 00_visualize_data.py) and NOT model predictions (that's
generate_results, wired into 01/02/03) -- the actual target values.

Two views, saved to comparison/:
  - raw_data_slices.png ("coupe"): a handful of example geometries, output
    vs. the continuous axis, as line curves.
  - raw_data_map.png ("carte"): a 2D heatmap of output vs. (geometry, axis).
    With a single geometry feature this dataset's axis is in physical units
    (e.g. theta vs frequency); with several features, geometries are
    ordered by their first principal component instead (no single physical
    y-axis possible).

Usage:
    python 06_plot_raw_data.py [--n-samples N]
"""
import argparse
import sys
from pathlib import Path

from surmod.core.data_loader import DataLoader as GFRDataLoader
from surmod.utils.common import resolve_data_path
from surmod.utils.yaml_handler import load_config

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # import the shared `_common` package
from _common.raw_data_viz import plot_slices, plot_map

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-samples", type=int, default=6, help="Number of example geometries in raw_data_slices.png")
    args = parser.parse_args()

    config = load_config("mesure_couplage_gfr", config_file=str(HERE / "config.yaml"))
    dm = GFRDataLoader(str(resolve_data_path(config)), config)

    plot_slices(dm, config, out_dir=str(HERE / "comparison"), n_samples=args.n_samples)
    plot_map(dm, config, out_dir=str(HERE / "comparison"))
