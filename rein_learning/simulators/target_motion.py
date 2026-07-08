from __future__ import annotations

import numpy as np

from .geometry import clip_position, unit_vector_towards


def velocity_towards_asset(
    target_position: np.ndarray,
    asset_position: np.ndarray,
    speed: float,
) -> np.ndarray:
    return unit_vector_towards(target_position, asset_position) * speed


def advance_position(
    position: np.ndarray,
    velocity: np.ndarray,
    dt: float,
    map_size: float,
) -> np.ndarray:
    return clip_position(position + velocity * dt, map_size)
