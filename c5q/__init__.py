"""
C5 Quantum Modeling Package

A comprehensive machine learning framework for quantum state prediction
with least-20 prediction capabilities using Neural Plackett-Luce and
Discrete Subset Diffusion models.
"""

__version__ = "0.1.0"
__author__ = "C5Q Development Team"
__email__ = "dev@c5quantum.com"

# Core modules
from . import utils
from . import io
from . import eda
from . import dataset

__all__ = [
    "utils",
    "io",
    "eda",
    "dataset"
]