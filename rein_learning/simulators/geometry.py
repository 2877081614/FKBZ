from __future__ import annotations

import numpy as np


def euclidean_distance(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.linalg.norm(first - second))


def unit_vector_towards(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    direction = destination - source
    norm = np.linalg.norm(direction)
    if norm == 0:
        return np.zeros_like(direction, dtype=np.float32)
    return (direction / norm).astype(np.float32)


def clip_position(position: np.ndarray, map_size: float) -> np.ndarray:
    return np.clip(position, -map_size, map_size).astype(np.float32)
