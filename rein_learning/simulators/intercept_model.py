from __future__ import annotations

from dataclasses import dataclass

from .geometry import euclidean_distance


@dataclass(frozen=True)
class InterceptResult:
    hit_probability: float
    hit: bool


def compute_hit_probability(
    defense_position,
    target_position,
    max_range: float,
    base_hit_probability: float,
    target_evasion: float = 0.0,
) -> float:
    distance = euclidean_distance(defense_position, target_position)
    if max_range <= 0 or distance > max_range:
        return 0.0

    range_factor = max(0.0, 1.0 - distance / max_range)
    target_factor = max(0.0, 1.0 - target_evasion)
    probability = base_hit_probability * range_factor * target_factor
    return float(min(1.0, max(0.0, probability)))


def sample_intercept(
    rng,
    defense_position,
    target_position,
    max_range: float,
    base_hit_probability: float,
    target_evasion: float = 0.0,
) -> InterceptResult:
    probability = compute_hit_probability(
        defense_position=defense_position,
        target_position=target_position,
        max_range=max_range,
        base_hit_probability=base_hit_probability,
        target_evasion=target_evasion,
    )
    return InterceptResult(
        hit_probability=probability,
        hit=bool(rng.random() < probability),
    )
