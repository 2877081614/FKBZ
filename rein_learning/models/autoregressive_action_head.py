from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.distributions import Categorical


@dataclass(frozen=True)
class AutoregressiveActionEvaluation:
    actions: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor


class AutoregressiveMaskedMultiCategorical:
    """Condition unit actions on the targets selected by earlier units."""

    def __init__(
        self,
        logits: torch.Tensor,
        action_dims: tuple[int, ...],
        action_masks: np.ndarray | torch.Tensor,
        unit_order: tuple[int, ...] | None = None,
    ) -> None:
        if logits.ndim != 2:
            raise ValueError(f"Expected 2D logits, got shape {tuple(logits.shape)}")
        if not action_dims or len(set(action_dims)) != 1:
            raise ValueError("All autoregressive action dimensions must be equal")
        if action_dims[0] < 2:
            raise ValueError("Each unit needs at least one target action and no-op")

        self.action_dims = tuple(int(value) for value in action_dims)
        self.num_units = len(self.action_dims)
        self.num_actions = self.action_dims[0]
        self.num_targets = self.num_actions - 1
        self.noop_action = self.num_targets
        self.unit_order = self._validate_unit_order(unit_order)
        expected_logits = sum(self.action_dims)
        if logits.shape[1] != expected_logits:
            raise ValueError(
                f"Expected {expected_logits} logits, got {logits.shape[1]}"
            )

        masks = torch.as_tensor(action_masks, device=logits.device, dtype=torch.bool)
        if masks.numel() % expected_logits != 0:
            raise ValueError(
                f"Action mask with {masks.numel()} values cannot be reshaped "
                f"into blocks of {expected_logits}"
            )
        masks = masks.reshape(-1, self.num_units, self.num_actions)
        if masks.shape[0] == 1 and logits.shape[0] > 1:
            masks = masks.expand(logits.shape[0], -1, -1)
        if masks.shape[0] != logits.shape[0]:
            raise ValueError(
                f"Mask batch {masks.shape[0]} does not match logits batch "
                f"{logits.shape[0]}"
            )
        if not bool(torch.all(torch.any(masks, dim=2))):
            raise ValueError("Every unit must have at least one legal base action")

        self.logits = logits.reshape(-1, self.num_units, self.num_actions)
        self.base_masks = masks

    def get_actions(self, deterministic: bool = False) -> torch.Tensor:
        return self.sample(deterministic=deterministic).actions

    def sample(self, deterministic: bool = False) -> AutoregressiveActionEvaluation:
        actions = torch.empty(
            (self.logits.shape[0], self.num_units),
            device=self.logits.device,
            dtype=torch.long,
        )
        log_probs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        used_targets = torch.zeros(
            (self.logits.shape[0], self.num_targets),
            device=self.logits.device,
            dtype=torch.bool,
        )

        for unit_index in self.unit_order:
            conditional_mask = self._unit_mask(unit_index, used_targets)
            distribution = self._categorical(unit_index, conditional_mask)
            action = (
                torch.argmax(distribution.logits, dim=1)
                if deterministic
                else distribution.sample()
            )
            actions[:, unit_index] = action
            log_probs.append(distribution.log_prob(action))
            entropies.append(distribution.entropy())
            used_targets = self._add_selected_target(used_targets, action)

        return AutoregressiveActionEvaluation(
            actions=actions,
            log_prob=torch.stack(log_probs, dim=1).sum(dim=1),
            entropy=torch.stack(entropies, dim=1).sum(dim=1),
        )

    def evaluate(self, actions: torch.Tensor) -> AutoregressiveActionEvaluation:
        actions = actions.long().reshape(-1, self.num_units)
        if actions.shape[0] != self.logits.shape[0]:
            raise ValueError(
                f"Action batch {actions.shape[0]} does not match logits batch "
                f"{self.logits.shape[0]}"
            )

        log_probs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        used_targets = torch.zeros(
            (self.logits.shape[0], self.num_targets),
            device=self.logits.device,
            dtype=torch.bool,
        )
        batch_indices = torch.arange(self.logits.shape[0], device=self.logits.device)

        for unit_index in self.unit_order:
            action = actions[:, unit_index]
            if bool(torch.any((action < 0) | (action >= self.num_actions))):
                raise ValueError("Action is outside the per-unit action range")
            conditional_mask = self._unit_mask(unit_index, used_targets)
            if not bool(torch.all(conditional_mask[batch_indices, action])):
                raise ValueError(
                    "Action is illegal under the reconstructed autoregressive mask"
                )
            distribution = self._categorical(unit_index, conditional_mask)
            log_probs.append(distribution.log_prob(action))
            entropies.append(distribution.entropy())
            used_targets = self._add_selected_target(used_targets, action)

        return AutoregressiveActionEvaluation(
            actions=actions,
            log_prob=torch.stack(log_probs, dim=1).sum(dim=1),
            entropy=torch.stack(entropies, dim=1).sum(dim=1),
        )

    def conditional_masks(self, actions: torch.Tensor) -> torch.Tensor:
        """Return each unit's mask under the supplied action prefix."""

        actions = actions.long().reshape(-1, self.num_units)
        if actions.shape[0] != self.logits.shape[0]:
            raise ValueError("Action and logits batches must match")
        masks = torch.empty_like(self.base_masks)
        used_targets = torch.zeros(
            (self.logits.shape[0], self.num_targets),
            device=self.logits.device,
            dtype=torch.bool,
        )
        for unit_index in self.unit_order:
            masks[:, unit_index, :] = self._unit_mask(unit_index, used_targets)
            used_targets = self._add_selected_target(
                used_targets,
                actions[:, unit_index],
            )
        return masks

    def diagnostics(self, deterministic: bool = True) -> dict[str, torch.Tensor]:
        """Evaluate engagement statistics along the sampled action prefix."""

        evaluation = self.sample(deterministic=deterministic)
        actions = evaluation.actions
        used_targets = torch.zeros(
            (actions.shape[0], self.num_targets),
            device=actions.device,
            dtype=torch.bool,
        )
        fields: dict[str, list[torch.Tensor]] = {
            "engage_probability": [],
            "noop_probability": [],
            "noop_margin": [],
            "engagement_entropy": [],
            "conditional_target_entropy": [],
            "actionable": [],
        }
        for unit_index in self.unit_order:
            mask = self._unit_mask(unit_index, used_targets)
            distribution = self._categorical(unit_index, mask)
            probabilities = distribution.probs
            legal_targets = mask[:, : self.num_targets]
            actionable = legal_targets.any(dim=1)
            engage_probability = probabilities[:, : self.num_targets].sum(dim=1)
            noop_probability = probabilities[:, self.noop_action]
            unit_logits = self.logits[:, unit_index, :]
            best_target = unit_logits[:, : self.num_targets].masked_fill(
                ~legal_targets, -torch.inf
            ).max(dim=1).values
            noop_margin = unit_logits[:, self.noop_action] - best_target
            noop_margin = torch.where(
                actionable,
                noop_margin,
                torch.full_like(noop_margin, torch.inf),
            )
            target_probabilities = probabilities[:, : self.num_targets]
            target_probabilities = target_probabilities / engage_probability[
                :, None
            ].clamp_min(1e-20)
            binary = torch.stack((engage_probability, noop_probability), dim=1)
            for key, value in (
                ("engage_probability", engage_probability),
                ("noop_probability", noop_probability),
                ("noop_margin", noop_margin),
                ("engagement_entropy", self._probability_entropy(binary)),
                (
                    "conditional_target_entropy",
                    self._probability_entropy(target_probabilities),
                ),
                ("actionable", actionable.to(probabilities.dtype)),
            ):
                fields[key].append(value)
            used_targets = self._add_selected_target(
                used_targets, actions[:, unit_index]
            )
        return {
            "actions": actions,
            **{key: torch.stack(values, dim=1) for key, values in fields.items()},
        }

    @staticmethod
    def _probability_entropy(probabilities: torch.Tensor) -> torch.Tensor:
        terms = torch.where(
            probabilities > 0,
            probabilities * torch.log(probabilities.clamp_min(1e-20)),
            torch.zeros_like(probabilities),
        )
        return -terms.sum(dim=1)

    def _validate_unit_order(
        self,
        unit_order: tuple[int, ...] | None,
    ) -> tuple[int, ...]:
        order = (
            tuple(range(self.num_units))
            if unit_order is None
            else tuple(int(value) for value in unit_order)
        )
        if len(order) != self.num_units or set(order) != set(range(self.num_units)):
            raise ValueError(
                "unit_order must be a permutation of all unit indices"
            )
        return order

    def _unit_mask(
        self,
        unit_index: int,
        used_targets: torch.Tensor,
    ) -> torch.Tensor:
        mask = self.base_masks[:, unit_index, :].clone()
        mask[:, : self.num_targets] &= ~used_targets
        if not bool(torch.all(torch.any(mask, dim=1))):
            raise ValueError("Autoregressive masking produced an empty action set")
        return mask

    def _categorical(
        self,
        unit_index: int,
        mask: torch.Tensor,
    ) -> Categorical:
        masked_logits = self.logits[:, unit_index, :].masked_fill(~mask, -1e8)
        return Categorical(logits=masked_logits)

    def _add_selected_target(
        self,
        used_targets: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        selected = torch.nn.functional.one_hot(
            actions.long(),
            num_classes=self.num_actions,
        )[:, : self.num_targets].bool()
        return used_targets | selected
