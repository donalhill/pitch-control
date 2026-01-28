"""
Pitch Control Analysis Library

Vectorized implementation of Spearman's pitch control model with EPV fusion.
"""

from pitch_control.models.pitch_control import (
    compute_pitch_control,
    default_model_params,
)
from pitch_control.models.epv import load_epv_grid, get_epv_at_location
from pitch_control.models.obso import compute_obso, compute_player_obso

__version__ = "0.1.0"

__all__ = [
    "compute_pitch_control",
    "default_model_params",
    "load_epv_grid",
    "get_epv_at_location",
    "compute_obso",
    "compute_player_obso",
]
