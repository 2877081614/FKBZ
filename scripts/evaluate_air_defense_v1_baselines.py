from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rein_learning.baselines import (
    GreedyDamageReductionPolicy,
    HighestThreatJointPolicy,
    HungarianDamageReductionPolicy,
    NearestTargetJointPolicy,
    RandomLegalJointPolicy,
    TimeToImpactJointPolicy,
    evaluate_air_defense_v1_policy,
)
from rein_learning.envs import AirDefenseResourceAssignmentEnvV1


def main() -> None:
    episodes = 50
    policies = {
        "random_joint": lambda seed: RandomLegalJointPolicy(seed=seed),
        "nearest_joint": lambda seed: NearestTargetJointPolicy(),
        "highest_threat": lambda seed: HighestThreatJointPolicy(),
        "time_to_impact": lambda seed: TimeToImpactJointPolicy(),
        "greedy_damage": lambda seed: GreedyDamageReductionPolicy(),
        "hungarian_damage": lambda seed: HungarianDamageReductionPolicy(),
    }

    print(f"Evaluating AirDefense v1 baselines over {episodes} episodes")
    print(
        "policy              avg_reward  success  intercept  leak   damage  "
        "ammo  shots  hit/shot  invalid  decision_ms  high_leak  conflict  "
        "overkill  dmg/ammo  cost"
    )

    for name, policy_factory in policies.items():
        metrics = evaluate_air_defense_v1_policy(
            env_factory=AirDefenseResourceAssignmentEnvV1,
            policy_factory=policy_factory,
            episodes=episodes,
            seed=200,
        )
        print(
            f"{name:<18}"
            f"{metrics['avg_reward']:>10.2f}"
            f"{metrics['success_rate']:>9.2f}"
            f"{metrics['intercept_rate']:>11.2f}"
            f"{metrics['leak_rate']:>7.2f}"
            f"{metrics['avg_total_damage']:>9.2f}"
            f"{metrics['avg_ammo_used']:>7.2f}"
            f"{metrics['avg_shots']:>7.2f}"
            f"{metrics['hit_rate_per_shot']:>10.2f}"
            f"{metrics['avg_invalid_actions']:>9.2f}"
            f"{metrics['avg_decision_time_ms']:>13.3f}"
            f"{metrics['high_threat_leak_rate']:>11.2f}"
            f"{metrics['assignment_conflict_rate']:>10.2f}"
            f"{metrics['overkill_rate']:>10.2f}"
            f"{metrics['damage_reduction_per_ammo']:>10.2f}"
            f"{metrics['avg_resource_cost']:>7.2f}"
        )


if __name__ == "__main__":
    main()
