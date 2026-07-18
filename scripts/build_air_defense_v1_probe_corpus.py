from __future__ import annotations

import argparse
from math import ceil
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rein_learning.algorithms.policy_gradient import (
    AutoregressiveMaskablePPO,
    RoleConditionedAutoregressiveMaskablePPO,
)
from rein_learning.baselines import HungarianDamageReductionPolicy
from rein_learning.common import make_policy_probe_corpus
from rein_learning.envs import (
    AirDefenseResourceAssignmentEnvV1,
    get_air_defense_v1_scenario,
)


DEFAULT_TASK10_MODEL = (
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "task10_order_screening_30k_3seeds"
    / "models"
    / "medium"
    / "autoregressive_ppo_order_012_seed0.zip"
)
DEFAULT_TASK11_MODEL = (
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "task11_role_conditioned_screening_30k_3seeds"
    / "models"
    / "medium"
    / "role_conditioned_ar_ppo_order_012_seed0.zip"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the frozen AirDefense v1 Task 12 policy-probe corpus."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results"
            / "air_defense_v1"
            / "task12_probe_corpus"
        ),
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=("medium", "time_pressure", "heterogeneity_pressure"),
    )
    parser.add_argument("--states-per-scenario", type=int, default=256)
    parser.add_argument("--seed-start", type=int, default=40_000)
    parser.add_argument("--task10-model", type=Path, default=DEFAULT_TASK10_MODEL)
    parser.add_argument("--task11-model", type=Path, default=DEFAULT_TASK11_MODEL)
    return parser.parse_args()


def _model_action_selector(
    model: Any,
) -> Callable[[Any, Any, np.ndarray], np.ndarray]:
    def select(
        env: Any, observation: Any, action_mask: np.ndarray
    ) -> np.ndarray:
        action, _ = model.predict(
            observation,
            deterministic=True,
            action_masks=action_mask,
        )
        return np.asarray(action, dtype=np.int64)

    return select


def _collect_source_records(
    *,
    scenario: str,
    source: str,
    quota: int,
    seed_start: int,
    action_selector: Callable[[Any, Any, np.ndarray], np.ndarray],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    episode_index = 0
    target_candidates = max(4 * quota, quota + 32)
    while len(candidates) < target_candidates:
        environment_seed = seed_start + episode_index
        env = AirDefenseResourceAssignmentEnvV1(
            config=get_air_defense_v1_scenario(scenario)
        )
        observation, _ = env.reset(seed=environment_seed)
        episode_records: list[dict[str, Any]] = []
        terminated = False
        truncated = False
        step_index = 0
        while not (terminated or truncated):
            action_mask = np.asarray(env.action_masks(), dtype=np.bool_)
            reshaped = action_mask.reshape(env.num_defense_units, -1)
            if bool(np.any(reshaped[:, : env.num_targets])):
                episode_records.append(
                    {
                        "observation": np.asarray(observation, dtype=np.float32).copy(),
                        "action_mask": action_mask.copy(),
                        "scenario": scenario,
                        "source": source,
                        "environment_seed": environment_seed,
                        "episode_index": episode_index,
                        "step_index": step_index,
                    }
                )
            action = action_selector(env, observation, action_mask)
            observation, _, terminated, truncated, _ = env.step(action)
            step_index += 1
        env.close()
        episode_length = max(step_index, 1)
        for record in episode_records:
            fraction = float(record["step_index"]) / episode_length
            record["phase_index"] = min(int(fraction * 4), 3)
            record["phase"] = (
                "initial",
                "early",
                "middle",
                "near_terminal",
            )[record["phase_index"]]
        candidates.extend(episode_records)
        episode_index += 1
        if episode_index > 2_000:
            raise RuntimeError(
                f"Could not collect enough actionable states for {scenario}/{source}"
            )
    return _stratified_select(candidates, quota)


def _stratified_select(
    candidates: list[dict[str, Any]], quota: int
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    per_phase = ceil(quota / 4)
    for phase in range(4):
        phase_rows = [row for row in candidates if row["phase_index"] == phase]
        for row in phase_rows[:per_phase]:
            selected.append(row)
            selected_ids.add(id(row))
            if len(selected) == quota:
                return selected
    for row in candidates:
        if id(row) in selected_ids:
            continue
        selected.append(row)
        if len(selected) == quota:
            break
    if len(selected) != quota:
        raise RuntimeError("Probe phase stratification did not fill its quota")
    return selected


def main() -> None:
    args = parse_args()
    if args.states_per_scenario < 4:
        raise ValueError("states-per-scenario must be at least 4")
    for path in (args.task10_model, args.task11_model):
        if not path.exists():
            raise FileNotFoundError(path)

    task10 = AutoregressiveMaskablePPO.load(args.task10_model, device="cpu")
    task11 = RoleConditionedAutoregressiveMaskablePPO.load(
        args.task11_model, device="cpu"
    )
    hungarian = HungarianDamageReductionPolicy()
    sources: tuple[
        tuple[str, Callable[[Any, Any, np.ndarray], np.ndarray]], ...
    ] = (
        (
            "hungarian",
            lambda env, observation, mask: hungarian.select_action(env),
        ),
        ("task10_order_012_seed0", _model_action_selector(task10)),
        ("task11_role_order_012_seed0", _model_action_selector(task11)),
    )

    records: list[dict[str, Any]] = []
    source_quotas = [
        args.states_per_scenario // len(sources)
        + (1 if index < args.states_per_scenario % len(sources) else 0)
        for index in range(len(sources))
    ]
    for scenario_index, scenario in enumerate(args.scenarios):
        for source_index, ((source_name, selector), quota) in enumerate(
            zip(sources, source_quotas)
        ):
            seed_start = (
                args.seed_start + scenario_index * 10_000 + source_index * 1_000
            )
            source_records = _collect_source_records(
                scenario=scenario,
                source=source_name,
                quota=quota,
                seed_start=seed_start,
                action_selector=selector,
            )
            records.extend(source_records)
            print(
                f"collected scenario={scenario} source={source_name} "
                f"states={len(source_records)}",
                flush=True,
            )

    corpus = make_policy_probe_corpus(records)
    manifest = corpus.save(
        args.output_dir,
        metadata={
            "scenarios": list(args.scenarios),
            "states_per_scenario": args.states_per_scenario,
            "seed_start": args.seed_start,
            "phase_bins": ["initial", "early", "middle", "near_terminal"],
            "task10_model": str(args.task10_model.resolve()),
            "task11_model": str(args.task11_model.resolve()),
        },
    )
    print(f"probe_dir={args.output_dir.resolve()}")
    print(f"num_states={manifest['num_states']}")
    print(f"content_sha256={manifest['content_sha256']}")

if __name__ == "__main__":
    main()
