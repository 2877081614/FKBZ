from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


@dataclass(frozen=True)
class PolicyProbeCorpus:
    observations: np.ndarray
    action_masks: np.ndarray
    scenarios: np.ndarray
    sources: np.ndarray
    phases: np.ndarray
    environment_seeds: np.ndarray
    episode_indices: np.ndarray
    step_indices: np.ndarray

    def __post_init__(self) -> None:
        size = int(self.observations.shape[0])
        if size <= 0:
            raise ValueError("Probe corpus must contain at least one state")
        arrays = (
            self.action_masks,
            self.scenarios,
            self.sources,
            self.phases,
            self.environment_seeds,
            self.episode_indices,
            self.step_indices,
        )
        if any(array.shape[0] != size for array in arrays):
            raise ValueError("All probe arrays must have the same first dimension")
        if self.observations.ndim != 2 or self.action_masks.ndim != 2:
            raise ValueError("Probe observations and masks must be two-dimensional")

    @property
    def size(self) -> int:
        return int(self.observations.shape[0])

    def content_sha256(self) -> str:
        digest = hashlib.sha256()
        for name, array in self._named_arrays():
            contiguous = np.ascontiguousarray(array)
            digest.update(name.encode("utf-8"))
            digest.update(str(contiguous.dtype).encode("ascii"))
            digest.update(json.dumps(contiguous.shape).encode("ascii"))
            if contiguous.dtype.kind in {"U", "O"}:
                digest.update(
                    json.dumps(
                        contiguous.tolist(),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                )
            else:
                digest.update(contiguous.tobytes(order="C"))
        return digest.hexdigest()

    def save(
        self,
        output_dir: str | Path,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        corpus_path = directory / "probe_states.npz"
        np.savez_compressed(corpus_path, **dict(self._named_arrays()))
        scenario_counts = self._counts(self.scenarios)
        source_counts = self._counts(self.sources)
        phase_counts = self._counts(self.phases)
        manifest = {
            "schema_version": 1,
            "content_sha256": self.content_sha256(),
            "num_states": self.size,
            "observation_shape": list(self.observations.shape[1:]),
            "action_mask_shape": list(self.action_masks.shape[1:]),
            "scenario_counts": scenario_counts,
            "source_counts": source_counts,
            "phase_counts": phase_counts,
            "metadata": metadata or {},
        }
        (directory / "probe_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with (directory / "probe_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=("dimension", "name", "count")
            )
            writer.writeheader()
            for dimension, counts in (
                ("scenario", scenario_counts),
                ("source", source_counts),
                ("phase", phase_counts),
            ):
                for name, count in counts.items():
                    writer.writerow(
                        {"dimension": dimension, "name": name, "count": count}
                    )
        return manifest

    @classmethod
    def load(cls, path: str | Path, *, verify_hash: bool = True) -> "PolicyProbeCorpus":
        source = Path(path)
        corpus_path = source / "probe_states.npz" if source.is_dir() else source
        with np.load(corpus_path, allow_pickle=False) as payload:
            corpus = cls(
                observations=payload["observations"],
                action_masks=payload["action_masks"],
                scenarios=payload["scenarios"],
                sources=payload["sources"],
                phases=payload["phases"],
                environment_seeds=payload["environment_seeds"],
                episode_indices=payload["episode_indices"],
                step_indices=payload["step_indices"],
            )
        manifest_path = corpus_path.parent / "probe_manifest.json"
        if verify_hash and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("content_sha256") != corpus.content_sha256():
                raise ValueError("Probe corpus hash does not match its manifest")
        return corpus

    def _named_arrays(self) -> tuple[tuple[str, np.ndarray], ...]:
        return (
            ("observations", np.asarray(self.observations, dtype=np.float32)),
            ("action_masks", np.asarray(self.action_masks, dtype=np.bool_)),
            ("scenarios", np.asarray(self.scenarios, dtype=np.str_)),
            ("sources", np.asarray(self.sources, dtype=np.str_)),
            ("phases", np.asarray(self.phases, dtype=np.str_)),
            ("environment_seeds", np.asarray(self.environment_seeds, dtype=np.int64)),
            ("episode_indices", np.asarray(self.episode_indices, dtype=np.int64)),
            ("step_indices", np.asarray(self.step_indices, dtype=np.int64)),
        )

    @staticmethod
    def _counts(values: np.ndarray) -> dict[str, int]:
        unique, counts = np.unique(values.astype(str), return_counts=True)
        return {str(name): int(count) for name, count in zip(unique, counts)}


def make_policy_probe_corpus(records: Iterable[dict[str, Any]]) -> PolicyProbeCorpus:
    rows = list(records)
    if not rows:
        raise ValueError("At least one probe record is required")
    return PolicyProbeCorpus(
        observations=np.stack([row["observation"] for row in rows]).astype(np.float32),
        action_masks=np.stack([row["action_mask"] for row in rows]).astype(np.bool_),
        scenarios=np.asarray([row["scenario"] for row in rows], dtype=np.str_),
        sources=np.asarray([row["source"] for row in rows], dtype=np.str_),
        phases=np.asarray([row["phase"] for row in rows], dtype=np.str_),
        environment_seeds=np.asarray(
            [row["environment_seed"] for row in rows], dtype=np.int64
        ),
        episode_indices=np.asarray(
            [row["episode_index"] for row in rows], dtype=np.int64
        ),
        step_indices=np.asarray([row["step_index"] for row in rows], dtype=np.int64),
    )


def evaluate_policy_probe(
    model: Any,
    corpus: PolicyProbeCorpus,
    *,
    deterministic: bool = True,
    batch_size: int = 256,
) -> list[dict[str, float | int | str]]:
    """Evaluate a policy on frozen states without advancing an environment."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    policy = model.policy
    scenario_names = sorted(set(corpus.scenarios.astype(str).tolist()))
    rows: list[dict[str, float | int | str]] = []
    for scenario in ("all", *scenario_names):
        indices = (
            np.arange(corpus.size)
            if scenario == "all"
            else np.flatnonzero(corpus.scenarios.astype(str) == scenario)
        )
        collected: dict[str, list[np.ndarray]] = {
            "engage_probability": [],
            "noop_probability": [],
            "noop_margin": [],
            "engagement_entropy": [],
            "conditional_target_entropy": [],
            "actionable": [],
            "actions": [],
            "values": [],
        }
        with torch.no_grad():
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start : start + batch_size]
                observations, _ = policy.obs_to_tensor(
                    corpus.observations[batch_indices]
                )
                masks = corpus.action_masks[batch_indices]
                distribution = policy.get_distribution(observations, masks)
                if not hasattr(distribution, "diagnostics"):
                    raise TypeError("Policy distribution does not expose diagnostics()")
                diagnostics = distribution.diagnostics(
                    deterministic=deterministic
                )
                for key in collected:
                    if key == "values":
                        tensor = policy.predict_values(observations).reshape(-1)
                    else:
                        tensor = diagnostics[key]
                    collected[key].append(tensor.detach().cpu().numpy())
        merged = {key: np.concatenate(values, axis=0) for key, values in collected.items()}
        actionable = merged["actionable"].astype(bool)
        finite_margin = actionable & np.isfinite(merged["noop_margin"])
        actions = merged["actions"]
        num_targets = int(corpus.action_masks.shape[1] / actions.shape[1]) - 1
        deterministic_engagement = (actions != num_targets) & actionable
        rows.append(
            {
                "probe_scenario": scenario,
                "probe_states": int(len(indices)),
                "actionable_decisions": int(actionable.sum()),
                "engage_probability_mean": _masked_mean(
                    merged["engage_probability"], actionable
                ),
                "noop_probability_mean": _masked_mean(
                    merged["noop_probability"], actionable
                ),
                "noop_margin_mean": _masked_mean(
                    merged["noop_margin"], finite_margin
                ),
                "engagement_entropy_mean": _masked_mean(
                    merged["engagement_entropy"], actionable
                ),
                "conditional_target_entropy_mean": _masked_mean(
                    merged["conditional_target_entropy"], actionable
                ),
                "deterministic_engagement_rate": _masked_mean(
                    deterministic_engagement.astype(float), actionable
                ),
                "probe_value_mean": float(np.mean(merged["values"])),
            }
        )
    return rows


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(values)[np.asarray(mask, dtype=bool)]
    return float(np.mean(selected)) if selected.size else float("nan")
