"""
SurMod

Professional surrogate modeling library for RF and electromagnetic responses
using Geometry-Frequency Response Networks (GFR-Net) and variants.
"""

from __future__ import annotations

from importlib.metadata import version, PackageNotFoundError

try:
    from ._version import __version__, __version_tuple__
except (ImportError, ModuleNotFoundError):
    try:
        __version__ = version("surmod")
    except PackageNotFoundError:
        __version__ = "0.0.0+unknown"
    __version_tuple__ = tuple(int(x) if x.isdigit() else x for x in __version__.split(".")[:3])

# High-level public API
from .main import (
    train_model,
    generate_results,
    post_process,
    predict,
    load_config,
)

from . import core
from . import models
from . import utils

__all__ = [
    "__version__",
    "__version_tuple__",
    "train_model",
    "generate_results",
    "post_process",
    "predict",
    "load_config",
    "core",
    "models",
    "utils",
]

# After installation you can verify the whole stack with:
#   self-test
#   # or
#   python -m surmod.cli.self_test
