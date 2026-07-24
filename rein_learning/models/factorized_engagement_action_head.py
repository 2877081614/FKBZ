from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from .air_defense_observation_layout import AirDefenseV1ObservationLayout
from .autoregressive_action_head import AutoregressiveActionEvaluation


@dataclass(frozen=True)
class FactorizedEngagementActionHeadConfig:
    entity_embedding_dim: int = 32
    context_dim: int = 96
    relation_hidden_dim: int = 64
    initial_engage_bias: float = 0.0

    def __post_init__(self) -> None:
        dimensions = (
            self.entity_embedding_dim,
            self.context_dim,
            self.relation_hidden_dim,
        )
        if min(dimensions) <= 0:
            raise ValueError("Factorized head dimensions must be positive")

    def signature(self) -> dict[str, object]:
        return {
            "type": "factorized_engagement_unit_target_relation_mlp",
            **{key: value for key, value in asdict(self).items()},
            "unit_index_embedding": False,
            "engagement_distribution": "bernoulli",
            "target_distribution": "conditional_legal_categorical",
        }


class FactorizedEngagementAirDefenseActionHead(nn.Module):
    """Shared relation scorer with an explicit per-unit engagement logit."""

    TARGET_ALIVE_INDEX = 13
    UNIT_POSITION = slice(0, 2)
    UNIT_RANGE_INDEX = 6
    TARGET_POSITION = slice(0, 2)

    def __init__(
        self,
        layout: AirDefenseV1ObservationLayout,
        config: FactorizedEngagementActionHeadConfig | None = None,
    ) -> None:
        super().__init__()
        self.layout = layout
        self.config = config or FactorizedEngagementActionHeadConfig()
        embedding_dim = self.config.entity_embedding_dim

        self.zone_encoder = nn.Sequential(
            nn.Linear(layout.zone_feature_dim, embedding_dim),
            nn.Tanh(),
        )
        self.target_encoder = nn.Sequential(
            nn.Linear(layout.target_feature_dim, embedding_dim),
            nn.Tanh(),
        )
        self.unit_encoder = nn.Sequential(
            nn.Linear(layout.unit_feature_dim, embedding_dim),
            nn.Tanh(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(layout.global_feature_dim, embedding_dim),
            nn.Tanh(),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(4 * embedding_dim, self.config.context_dim),
            nn.Tanh(),
        )

        pair_input_dim = 2 * embedding_dim + self.config.context_dim + 4
        self.pair_hidden = nn.Sequential(
            nn.Linear(pair_input_dim, self.config.relation_hidden_dim),
            nn.Tanh(),
        )
        self.pair_output = nn.Linear(self.config.relation_hidden_dim, 1)

        engage_input_dim = 2 * embedding_dim + self.config.context_dim
        self.engage_hidden = nn.Sequential(
            nn.Linear(engage_input_dim, self.config.relation_hidden_dim),
            nn.Tanh(),
        )
        self.engage_output = nn.Linear(self.config.relation_hidden_dim, 1)

    @property
    def num_actions(self) -> int:
        return self.layout.num_targets + 1

    def forward(
        self,
        observation: torch.Tensor,
        action_masks: np.ndarray | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        structured = self.layout.split(observation)
        zone_embeddings = self.zone_encoder(structured.zones)
        target_embeddings = self.target_encoder(structured.targets)
        unit_embeddings = self.unit_encoder(structured.units)
        global_embedding = self.global_encoder(structured.global_features)

        alive = structured.targets[:, :, self.TARGET_ALIVE_INDEX] > 0.5
        target_context = self._masked_mean(target_embeddings, alive)
        context = self.context_encoder(
            torch.cat(
                (
                    zone_embeddings.mean(dim=1),
                    target_context,
                    unit_embeddings.mean(dim=1),
                    global_embedding,
                ),
                dim=1,
            )
        )

        batch_size = observation.shape[0]
        units = unit_embeddings.unsqueeze(2).expand(
            -1, -1, self.layout.num_targets, -1
        )
        targets = target_embeddings.unsqueeze(1).expand(
            -1, self.layout.num_units, -1, -1
        )
        pair_context = context[:, None, None, :].expand(
            -1, self.layout.num_units, self.layout.num_targets, -1
        )
        pair_features = torch.cat(
            (
                units,
                targets,
                pair_context,
                self._relative_features(structured.units, structured.targets),
            ),
            dim=3,
        )
        target_logits = self.pair_output(
            self.pair_hidden(pair_features)
        ).squeeze(-1)

        masks = torch.as_tensor(
            action_masks, device=observation.device, dtype=torch.bool
        ).reshape(-1, self.layout.num_units, self.num_actions)
        if masks.shape[0] == 1 and batch_size > 1:
            masks = masks.expand(batch_size, -1, -1)
        if masks.shape[0] != batch_size:
            raise ValueError("Action-mask batch does not match observation batch")
        legal_target_context = self._per_unit_masked_target_mean(
            target_embeddings,
            masks[:, :, : self.layout.num_targets],
        )
        engage_context = context[:, None, :].expand(
            -1, self.layout.num_units, -1
        )
        engage_features = torch.cat(
            (unit_embeddings, legal_target_context, engage_context), dim=2
        )
        engage_logits = self.engage_output(
            self.engage_hidden(engage_features)
        ).squeeze(-1)
        return target_logits, engage_logits

    def signature(self) -> dict[str, object]:
        return {
            **self.config.signature(),
            "observation_layout": self.layout.signature(),
        }

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(values.dtype).unsqueeze(-1)
        return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    @staticmethod
    def _per_unit_masked_target_mean(
        target_embeddings: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        weights = masks.to(target_embeddings.dtype).unsqueeze(-1)
        totals = (target_embeddings.unsqueeze(1) * weights).sum(dim=2)
        return totals / weights.sum(dim=2).clamp_min(1.0)

    def _relative_features(
        self,
        unit_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> torch.Tensor:
        unit_positions = unit_features[:, :, self.UNIT_POSITION].unsqueeze(2)
        target_positions = target_features[:, :, self.TARGET_POSITION].unsqueeze(1)
        delta = target_positions - unit_positions
        distance = torch.linalg.vector_norm(delta, dim=3, keepdim=True)
        unit_range = unit_features[:, :, self.UNIT_RANGE_INDEX].unsqueeze(2).unsqueeze(3)
        return torch.cat((delta, distance, unit_range - distance), dim=3)


class FactorizedEngagementAutoregressiveDistribution:
    """Conflict-free joint distribution with engagement separated from targeting."""

    def __init__(
        self,
        target_logits: torch.Tensor,
        engage_logits: torch.Tensor,
        action_dims: tuple[int, ...],
        action_masks: np.ndarray | torch.Tensor,
        unit_order: tuple[int, ...] | None = None,
    ) -> None:
        if target_logits.ndim != 3 or engage_logits.ndim != 2:
            raise ValueError("Target logits must be 3D and engage logits must be 2D")
        self.action_dims = tuple(int(value) for value in action_dims)
        if not self.action_dims or len(set(self.action_dims)) != 1:
            raise ValueError("All action dimensions must be equal")
        self.num_units = len(self.action_dims)
        self.num_actions = self.action_dims[0]
        self.num_targets = self.num_actions - 1
        self.noop_action = self.num_targets
        if target_logits.shape[1:] != (self.num_units, self.num_targets):
            raise ValueError("Target-logit shape does not match action dimensions")
        if engage_logits.shape != target_logits.shape[:2]:
            raise ValueError("Engage-logit shape does not match target logits")
        self.unit_order = self._validate_unit_order(unit_order)

        masks = torch.as_tensor(
            action_masks, device=target_logits.device, dtype=torch.bool
        )
        expected_mask_size = self.num_units * self.num_actions
        if masks.numel() % expected_mask_size != 0:
            raise ValueError("Action masks cannot be reshaped into unit blocks")
        masks = masks.reshape(-1, self.num_units, self.num_actions)
        if masks.shape[0] == 1 and target_logits.shape[0] > 1:
            masks = masks.expand(target_logits.shape[0], -1, -1)
        if masks.shape[0] != target_logits.shape[0]:
            raise ValueError("Mask and logit batches must match")
        if not bool(torch.all(masks[:, :, self.noop_action])):
            raise ValueError("No-op must remain legal for every unit")

        self.target_logits = target_logits
        self.engage_logits = engage_logits
        self.base_masks = masks

    def get_actions(self, deterministic: bool = False) -> torch.Tensor:
        return self.sample(deterministic=deterministic).actions

    def sample(self, deterministic: bool = False) -> AutoregressiveActionEvaluation:
        if deterministic:
            return self.sample_with_engagement_threshold(0.5)
        batch_size = self.target_logits.shape[0]
        actions = torch.empty(
            (batch_size, self.num_units),
            device=self.target_logits.device,
            dtype=torch.long,
        )
        used_targets = torch.zeros(
            (batch_size, self.num_targets),
            device=self.target_logits.device,
            dtype=torch.bool,
        )
        log_probs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        for unit_index in self.unit_order:
            probabilities, _ = self._unit_probabilities(unit_index, used_targets)
            action = torch.distributions.Categorical(probabilities).sample()
            actions[:, unit_index] = action
            log_probs.append(self._selected_log_prob(probabilities, action))
            entropies.append(self._entropy(probabilities))
            used_targets = self._add_selected_target(used_targets, action)
        return AutoregressiveActionEvaluation(
            actions=actions,
            log_prob=torch.stack(log_probs, dim=1).sum(dim=1),
            entropy=torch.stack(entropies, dim=1).sum(dim=1),
        )

    def sample_with_engagement_threshold(
        self, threshold: float
    ) -> AutoregressiveActionEvaluation:
        """Select engagement by threshold, then the best legal target."""

        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Engagement threshold must be in [0, 1]")
        batch_size = self.target_logits.shape[0]
        actions = torch.empty(
            (batch_size, self.num_units),
            device=self.target_logits.device,
            dtype=torch.long,
        )
        used_targets = torch.zeros(
            (batch_size, self.num_targets),
            device=self.target_logits.device,
            dtype=torch.bool,
        )
        log_probs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        for unit_index in self.unit_order:
            probabilities, _ = self._unit_probabilities(unit_index, used_targets)
            engage_probability = probabilities[:, : self.num_targets].sum(dim=1)
            target_action = probabilities[:, : self.num_targets].argmax(dim=1)
            actionable = engage_probability > 0.0
            action = torch.where(
                actionable & (engage_probability >= threshold),
                target_action,
                torch.full_like(target_action, self.noop_action),
            )
            actions[:, unit_index] = action
            log_probs.append(self._selected_log_prob(probabilities, action))
            entropies.append(self._entropy(probabilities))
            used_targets = self._add_selected_target(used_targets, action)
        return AutoregressiveActionEvaluation(
            actions=actions,
            log_prob=torch.stack(log_probs, dim=1).sum(dim=1),
            entropy=torch.stack(entropies, dim=1).sum(dim=1),
        )

    def sample_with_fixed_actions(
        self,
        fixed_actions: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> AutoregressiveActionEvaluation:
        """Complete a partially fixed action prefix under dynamic masks.

        Fixed actions use their regular action index; -1 marks units that should
        be sampled from the current conditional policy.
        """

        fixed = fixed_actions.long().reshape(-1, self.num_units)
        if fixed.shape[0] == 1 and self.target_logits.shape[0] > 1:
            fixed = fixed.expand(self.target_logits.shape[0], -1)
        if fixed.shape[0] != self.target_logits.shape[0]:
            raise ValueError("Fixed-action and distribution batches must match")
        if bool(torch.any((fixed < -1) | (fixed >= self.num_actions))):
            raise ValueError("Fixed actions must be -1 or a valid action index")

        actions = torch.empty_like(fixed)
        used_targets = torch.zeros(
            (fixed.shape[0], self.num_targets),
            device=fixed.device,
            dtype=torch.bool,
        )
        log_probs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        batch_indices = torch.arange(fixed.shape[0], device=fixed.device)
        for unit_index in self.unit_order:
            probabilities, mask = self._unit_probabilities(unit_index, used_targets)
            requested = fixed[:, unit_index]
            sampled = (
                probabilities.argmax(dim=1)
                if deterministic
                else torch.distributions.Categorical(probabilities).sample()
            )
            action = torch.where(requested >= 0, requested, sampled)
            if not bool(torch.all(mask[batch_indices, action])):
                raise ValueError("A fixed action is illegal under its prefix mask")
            actions[:, unit_index] = action
            log_probs.append(self._selected_log_prob(probabilities, action))
            entropies.append(self._entropy(probabilities))
            used_targets = self._add_selected_target(used_targets, action)
        return AutoregressiveActionEvaluation(
            actions=actions,
            log_prob=torch.stack(log_probs, dim=1).sum(dim=1),
            entropy=torch.stack(entropies, dim=1).sum(dim=1),
        )

    def complete_fixed_actions_with_engagement_threshold(
        self,
        fixed_actions: torch.Tensor,
        *,
        threshold: float = 0.5,
    ) -> AutoregressiveActionEvaluation:
        """Complete unfixed units using factorized deterministic semantics."""

        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Engagement threshold must be in [0, 1]")
        fixed = fixed_actions.long().reshape(-1, self.num_units)
        if fixed.shape[0] == 1 and self.target_logits.shape[0] > 1:
            fixed = fixed.expand(self.target_logits.shape[0], -1)
        if fixed.shape[0] != self.target_logits.shape[0]:
            raise ValueError("Fixed-action and distribution batches must match")
        if bool(torch.any((fixed < -1) | (fixed >= self.num_actions))):
            raise ValueError("Fixed actions must be -1 or a valid action index")

        actions = torch.empty_like(fixed)
        used_targets = torch.zeros(
            (fixed.shape[0], self.num_targets),
            device=fixed.device,
            dtype=torch.bool,
        )
        log_probs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        batch_indices = torch.arange(fixed.shape[0], device=fixed.device)
        for unit_index in self.unit_order:
            probabilities, mask = self._unit_probabilities(unit_index, used_targets)
            requested = fixed[:, unit_index]
            engage_probability = probabilities[:, : self.num_targets].sum(dim=1)
            target_action = probabilities[:, : self.num_targets].argmax(dim=1)
            actionable = mask[:, : self.num_targets].any(dim=1)
            deterministic_action = torch.where(
                actionable & (engage_probability >= threshold),
                target_action,
                torch.full_like(target_action, self.noop_action),
            )
            action = torch.where(requested >= 0, requested, deterministic_action)
            if not bool(torch.all(mask[batch_indices, action])):
                raise ValueError("A fixed action is illegal under its prefix mask")
            actions[:, unit_index] = action
            log_probs.append(self._selected_log_prob(probabilities, action))
            entropies.append(self._entropy(probabilities))
            used_targets = self._add_selected_target(used_targets, action)
        return AutoregressiveActionEvaluation(
            actions=actions,
            log_prob=torch.stack(log_probs, dim=1).sum(dim=1),
            entropy=torch.stack(entropies, dim=1).sum(dim=1),
        )

    def hierarchical_diagnostics(
        self,
        actions: torch.Tensor | None = None,
        *,
        engagement_threshold: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        """Return factor-level probabilities and log-probs for chosen actions."""

        if actions is None:
            actions = self.sample_with_engagement_threshold(
                engagement_threshold
            ).actions
        actions = actions.long().reshape(-1, self.num_units)
        if actions.shape[0] != self.target_logits.shape[0]:
            raise ValueError("Action and logit batches must match")

        shape = (actions.shape[0], self.num_units)
        probability_dtype = torch.promote_types(
            self.target_logits.dtype, self.engage_logits.dtype
        )
        engage_probabilities = torch.zeros(
            shape, device=actions.device, dtype=probability_dtype
        )
        selected_engage = torch.zeros(
            shape, device=actions.device, dtype=probability_dtype
        )
        target_probabilities = torch.full(
            shape, torch.nan, device=actions.device, dtype=probability_dtype
        )
        engagement_log_probs = torch.zeros(
            shape, device=actions.device, dtype=probability_dtype
        )
        target_log_probs = torch.zeros(
            shape, device=actions.device, dtype=probability_dtype
        )
        actionable_fields = torch.zeros(
            shape, device=actions.device, dtype=probability_dtype
        )
        used_targets = torch.zeros(
            (actions.shape[0], self.num_targets),
            device=actions.device,
            dtype=torch.bool,
        )
        batch_indices = torch.arange(actions.shape[0], device=actions.device)
        tiny = torch.finfo(probability_dtype).tiny

        for unit_index in self.unit_order:
            probabilities, mask = self._unit_probabilities(unit_index, used_targets)
            action = actions[:, unit_index]
            if not bool(torch.all(mask[batch_indices, action])):
                raise ValueError("Action is illegal under the autoregressive mask")
            engage_probability = probabilities[:, : self.num_targets].sum(dim=1)
            engaged = action != self.noop_action
            chosen_probability = probabilities[batch_indices, action]
            conditional_target_probability = chosen_probability / engage_probability.clamp_min(
                tiny
            )

            engage_probabilities[:, unit_index] = engage_probability
            selected_engage[:, unit_index] = engaged.to(probability_dtype)
            target_probabilities[:, unit_index] = torch.where(
                engaged,
                conditional_target_probability,
                torch.full_like(conditional_target_probability, torch.nan),
            )
            engagement_log_probs[:, unit_index] = torch.where(
                engaged,
                torch.log(engage_probability.clamp_min(tiny)),
                torch.log((1.0 - engage_probability).clamp_min(tiny)),
            )
            target_log_probs[:, unit_index] = torch.where(
                engaged,
                torch.log(conditional_target_probability.clamp_min(tiny)),
                torch.zeros_like(conditional_target_probability),
            )
            actionable_fields[:, unit_index] = mask[
                :, : self.num_targets
            ].any(dim=1).to(probability_dtype)
            used_targets = self._add_selected_target(used_targets, action)

        return {
            "actions": actions,
            "engage_probability": engage_probabilities,
            "selected_engage": selected_engage,
            "target_probability": target_probabilities,
            "engagement_log_prob": engagement_log_probs,
            "target_log_prob": target_log_probs,
            "actionable": actionable_fields,
        }

    def conditional_probabilities(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-unit action probabilities and masks along an action prefix."""

        actions = actions.long().reshape(-1, self.num_units)
        if actions.shape[0] != self.target_logits.shape[0]:
            raise ValueError("Action and logit batches must match")
        probabilities = torch.zeros(
            (actions.shape[0], self.num_units, self.num_actions),
            device=actions.device,
            dtype=torch.promote_types(
                self.target_logits.dtype, self.engage_logits.dtype
            ),
        )
        masks = torch.zeros(
            (actions.shape[0], self.num_units, self.num_actions),
            device=actions.device,
            dtype=torch.bool,
        )
        used_targets = torch.zeros(
            (actions.shape[0], self.num_targets),
            device=actions.device,
            dtype=torch.bool,
        )
        batch_indices = torch.arange(actions.shape[0], device=actions.device)
        for unit_index in self.unit_order:
            unit_probabilities, unit_mask = self._unit_probabilities(
                unit_index, used_targets
            )
            action = actions[:, unit_index]
            if not bool(torch.all(unit_mask[batch_indices, action])):
                raise ValueError("Action is illegal under the autoregressive mask")
            probabilities[:, unit_index, :] = unit_probabilities
            masks[:, unit_index, :] = unit_mask
            used_targets = self._add_selected_target(used_targets, action)
        return probabilities, masks

    def evaluate(self, actions: torch.Tensor) -> AutoregressiveActionEvaluation:
        actions = actions.long().reshape(-1, self.num_units)
        if actions.shape[0] != self.target_logits.shape[0]:
            raise ValueError("Action and logit batches must match")
        used_targets = torch.zeros(
            (actions.shape[0], self.num_targets),
            device=self.target_logits.device,
            dtype=torch.bool,
        )
        log_probs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        batch_indices = torch.arange(actions.shape[0], device=actions.device)
        for unit_index in self.unit_order:
            action = actions[:, unit_index]
            if bool(torch.any((action < 0) | (action >= self.num_actions))):
                raise ValueError("Action is outside the per-unit action range")
            probabilities, mask = self._unit_probabilities(unit_index, used_targets)
            if not bool(torch.all(mask[batch_indices, action])):
                raise ValueError("Action is illegal under the autoregressive mask")
            log_probs.append(self._selected_log_prob(probabilities, action))
            entropies.append(self._entropy(probabilities))
            used_targets = self._add_selected_target(used_targets, action)
        return AutoregressiveActionEvaluation(
            actions=actions,
            log_prob=torch.stack(log_probs, dim=1).sum(dim=1),
            entropy=torch.stack(entropies, dim=1).sum(dim=1),
        )

    def diagnostics(self, deterministic: bool = True) -> dict[str, torch.Tensor]:
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
            probabilities, mask = self._unit_probabilities(unit_index, used_targets)
            legal_targets = mask[:, : self.num_targets]
            actionable = legal_targets.any(dim=1)
            engage_probability = probabilities[:, : self.num_targets].sum(dim=1)
            noop_probability = probabilities[:, self.noop_action]
            masked_targets = self.target_logits[:, unit_index, :].masked_fill(
                ~legal_targets, -torch.inf
            )
            best_target = masked_targets.max(dim=1).values
            noop_margin = -self.engage_logits[:, unit_index]
            noop_margin = torch.where(
                actionable,
                noop_margin - best_target + torch.logsumexp(masked_targets, dim=1),
                torch.full_like(noop_margin, torch.inf),
            )
            target_probabilities = torch.softmax(masked_targets, dim=1)
            target_probabilities = torch.where(
                legal_targets, target_probabilities, torch.zeros_like(target_probabilities)
            )
            target_entropy = self._entropy(target_probabilities)
            binary = torch.stack((engage_probability, noop_probability), dim=1)
            for key, value in (
                ("engage_probability", engage_probability),
                ("noop_probability", noop_probability),
                ("noop_margin", noop_margin),
                ("engagement_entropy", self._entropy(binary)),
                ("conditional_target_entropy", target_entropy),
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

    def _unit_probabilities(
        self,
        unit_index: int,
        used_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = self.base_masks[:, unit_index, :].clone()
        mask[:, : self.num_targets] &= ~used_targets
        legal_targets = mask[:, : self.num_targets]
        actionable = legal_targets.any(dim=1)

        masked_logits = self.target_logits[:, unit_index, :].masked_fill(
            ~legal_targets, -torch.inf
        )
        target_probabilities = torch.softmax(masked_logits, dim=1)
        target_probabilities = torch.where(
            legal_targets, target_probabilities, torch.zeros_like(target_probabilities)
        )
        engage_probability = torch.sigmoid(self.engage_logits[:, unit_index])
        engage_probability = torch.where(
            actionable, engage_probability, torch.zeros_like(engage_probability)
        )
        probabilities = torch.cat(
            (
                engage_probability[:, None] * target_probabilities,
                (1.0 - engage_probability)[:, None],
            ),
            dim=1,
        )
        return probabilities, mask

    @staticmethod
    def _selected_log_prob(
        probabilities: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        selected = probabilities.gather(1, actions[:, None]).squeeze(1)
        return torch.log(selected.clamp_min(torch.finfo(probabilities.dtype).tiny))

    @staticmethod
    def _entropy(probabilities: torch.Tensor) -> torch.Tensor:
        terms = torch.where(
            probabilities > 0,
            probabilities * torch.log(probabilities.clamp_min(1e-20)),
            torch.zeros_like(probabilities),
        )
        return -terms.sum(dim=1)

    def _validate_unit_order(
        self, unit_order: tuple[int, ...] | None
    ) -> tuple[int, ...]:
        order = tuple(range(self.num_units)) if unit_order is None else tuple(unit_order)
        if len(order) != self.num_units or set(order) != set(range(self.num_units)):
            raise ValueError("unit_order must be a permutation of all unit indices")
        return order

    def _add_selected_target(
        self,
        used_targets: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        selected = torch.nn.functional.one_hot(
            actions.long(), num_classes=self.num_actions
        )[:, : self.num_targets].bool()
        return used_targets | selected
