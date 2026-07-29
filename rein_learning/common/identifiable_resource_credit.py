from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ResourceCreditComponents:
    """Semantic contract for the frozen R2 resource-cost decomposition.

    ``direct_cost`` is the probed unit's immediate cost in the engage branch.
    The three substitution terms use the N-minus-E direction, so positive
    values reduce the engage branch's episode-level incremental cost.
    """

    direct_cost: float
    same_step_other_substitution: float = 0.0
    future_probe_substitution: float = 0.0
    future_other_substitution: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.direct_cost,
            self.same_step_other_substitution,
            self.future_probe_substitution,
            self.future_other_substitution,
        )
        if not all(torch.isfinite(torch.tensor(value)) for value in values):
            raise ValueError("resource-credit components must be finite")
        if self.direct_cost < 0.0:
            raise ValueError("direct_cost must be non-negative")

    @property
    def total_substitution(self) -> float:
        return float(
            self.same_step_other_substitution
            + self.future_probe_substitution
            + self.future_other_substitution
        )

    @property
    def episode_cost_delta(self) -> float:
        return float(self.direct_cost - self.total_substitution)

    def as_vector(self) -> tuple[float, float, float, float]:
        return (
            float(self.direct_cost),
            float(self.same_step_other_substitution),
            float(self.future_probe_substitution),
            float(self.future_other_substitution),
        )


def compose_component_auxiliary_loss(
    *,
    joint_ppo_loss: torch.Tensor,
    component_auxiliary_loss: torch.Tensor,
    coefficient: float,
) -> torch.Tensor:
    """Compose a candidate auxiliary term while preserving strict fallback.

    This function is only the N1 software contract. It does not authorize an
    online algorithm. A later implementation must additionally prove sampling,
    ratio, clipping, optimizer-state, and one-step parameter-update equality
    when the coefficient is zero.
    """

    if coefficient < 0.0:
        raise ValueError("coefficient must be non-negative")
    if coefficient == 0.0:
        return joint_ppo_loss
    return joint_ppo_loss + coefficient * component_auxiliary_loss


def scalar_label_is_semantically_ambiguous(
    components: ResourceCreditComponents,
    *,
    tolerance: float = 1e-12,
) -> bool:
    """Return whether substitution fully masks positive immediate cost."""

    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    return (
        components.direct_cost > tolerance
        and components.episode_cost_delta <= tolerance
    )
