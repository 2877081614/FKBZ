from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rein_learning.envs import AirDefenseResourceAssignmentEnvV1
from rein_learning.simulators import euclidean_distance


def main() -> None:
    output_dir = PROJECT_ROOT / "docs" / "environments" / "air_defense"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "air_defense_v1_scenario_seed0.png"
    env = AirDefenseResourceAssignmentEnvV1()
    env.reset(seed=0)

    fig, ax = plt.subplots(figsize=(9, 8), dpi=160)
    ax.set_title("AirDefenseResourceAssignmentEnv v1.0 scenario, seed=0")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-105, 105)
    ax.set_ylim(-105, 105)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)

    zone_label_offsets = {
        0: (-8, -8),
        1: (3, -8),
    }
    for index, zone in enumerate(env.protected_zones):
        color = "#4c78a8" if zone.zone_type == "command" else "#72b7b2"
        circle = plt.Circle(
            zone.position,
            zone.radius,
            color=color,
            alpha=0.28,
            linewidth=2,
            fill=True,
        )
        ax.add_patch(circle)
        ax.scatter(*zone.position, s=160, marker="s", color=color, edgecolor="black")
        dx, dy = zone_label_offsets.get(index, (2, 2))
        ax.text(
            zone.position[0] + dx,
            zone.position[1] + dy,
            f"Z{index} {zone.zone_type}\nvalue={zone.value}",
            fontsize=8,
        )

    unit_label_offsets = {
        0: (-38, 2),
        1: (2, 4),
        2: (2, 3),
    }
    for index, unit in enumerate(env.defense_units):
        color = "#e45756" if unit.resource_type == "missile" else "#f2cf5b"
        range_circle = plt.Circle(
            unit.position,
            unit.max_range,
            color=color,
            alpha=0.07,
            linewidth=1,
            fill=True,
        )
        ax.add_patch(range_circle)
        ax.scatter(*unit.position, s=140, marker="^", color=color, edgecolor="black")
        dx, dy = unit_label_offsets.get(index, (2, 2))
        ax.text(
            unit.position[0] + dx,
            unit.position[1] + dy,
            f"D{index} {unit.resource_type}\nrange={unit.max_range}, ammo={unit.ammo}",
            fontsize=8,
        )

    for index, target in enumerate(env.targets):
        zone = env.protected_zones[target.target_zone]
        target_color = "#54a24b" if target.target_class != "decoy" else "#b279a2"
        ax.scatter(
            *target.position,
            s=110,
            marker="x",
            color=target_color,
            linewidth=2.5,
        )
        direction = zone.position - target.position
        distance = euclidean_distance(target.position, zone.position)
        if distance > 0:
            unit_direction = direction / distance
            arrow_length = min(18.0, distance)
            ax.arrow(
                target.position[0],
                target.position[1],
                unit_direction[0] * arrow_length,
                unit_direction[1] * arrow_length,
                width=0.35,
                head_width=3.0,
                head_length=4.5,
                length_includes_head=True,
                color=target_color,
                alpha=0.7,
            )
        ax.text(
            target.position[0] + 2,
            target.position[1] + 2,
            f"T{index}->Z{target.target_zone}\n"
            f"thr={target.threat:.2f}, payload={target.payload:.2f}\n"
            f"tti={target.time_to_impact:.1f}",
            fontsize=7,
        )

    legend_items = [
        plt.Line2D([0], [0], marker="s", color="w", label="Protected zone", markerfacecolor="#4c78a8", markeredgecolor="black", markersize=9),
        plt.Line2D([0], [0], marker="^", color="w", label="Defense unit", markerfacecolor="#e45756", markeredgecolor="black", markersize=9),
        plt.Line2D([0], [0], marker="x", color="#54a24b", label="Hostile UAV target", markersize=9, linestyle="None"),
    ]
    ax.legend(handles=legend_items, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path)
    env.close()
    print(output_path)


if __name__ == "__main__":
    main()
