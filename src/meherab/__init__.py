"""MEHERAB: Zero-Gradient Mid-layer Evolutionary Homophilic Exploration for
Remote-sensing Adaptation with Frozen Backbones (WACV 2027).

This package is a modularized release of the exact method implementation
used to produce every number, table, and figure in the paper. See
``docs/PAPER_MAP.md`` for the full equation/section <-> module mapping.
"""
__version__ = "1.0.0"

from .config import MEHERABConfig, GLOBAL_SEED, EVAL_SEEDS, set_all_seeds, print_config
from .backbone import MEHERABBackbone, load_frozen_backbone

__all__ = [
    "MEHERABConfig",
    "GLOBAL_SEED",
    "EVAL_SEEDS",
    "set_all_seeds",
    "print_config",
    "MEHERABBackbone",
    "load_frozen_backbone",
]
