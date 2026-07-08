from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rein_learning.baselines import (
    GreedyExpectedBenefitPolicy,
    HighestThreatPolicy,
    NearestTargetPolicy,
    RandomLegalPolicy,
    evaluate_air_defense_policy,
)
from rein_learning.envs import AirDefenseResourceAssignmentEnv


def main() -> None:
    episodes = 30
    policies = {
        "random_legal": lambda seed: RandomLegalPolicy(seed=seed),
        "nearest_target": lambda seed: NearestTargetPolicy(),
        "highest_threat": lambda seed: HighestThreatPolicy(),
        "greedy_expected": lambda seed: GreedyExpectedBenefitPolicy(),
    }

    print(f"Evaluating AirDefense baselines over {episodes} episodes")
    print(
        "policy              avg_reward  success  intercept  leak   "
        "ammo  shots  hit/shot  invalid"
    )

    for name, policy_factory in policies.items():
        metrics = evaluate_air_defense_policy(
            env_factory=AirDefenseResourceAssignmentEnv,
            policy_factory=policy_factory,
            episodes=episodes,
            seed=100,
        )
        print(
            f"{name:<18}"
            f"{metrics['avg_reward']:>10.2f}"
            f"{metrics['success_rate']:>9.2f}"
            f"{metrics['intercept_rate']:>11.2f}"
            f"{metrics['leak_rate']:>7.2f}"
            f"{metrics['avg_ammo_used']:>7.2f}"
            f"{metrics['avg_shots']:>7.2f}"
            f"{metrics['hit_rate_per_shot']:>10.2f}"
            f"{metrics['avg_invalid_actions']:>9.2f}"
        )


if __name__ == "__main__":
    main()
