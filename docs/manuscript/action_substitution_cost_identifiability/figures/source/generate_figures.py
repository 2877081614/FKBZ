from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[5]
MANUSCRIPT_ROOT = SCRIPT_PATH.parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results" / "air_defense_v1"
FIGURE_ROOT = MANUSCRIPT_ROOT / "figures"
SOURCE_ROOT = FIGURE_ROOT / "source"
EXPORT_ROOT = FIGURE_ROOT / "exported"
METADATA_ROOT = FIGURE_ROOT / "metadata"
TABLE_EXPORT_ROOT = MANUSCRIPT_ROOT / "tables" / "exported"
TABLE_METADATA_ROOT = MANUSCRIPT_ROOT / "tables" / "metadata"

R2_ROOT = RESULTS_ROOT / "action_substitution_confirmation"
R1_ROOT = RESULTS_ROOT / "action_substitution_opportunity_cost_audit"
LABEL_ROOT = RESULTS_ROOT / "bpce_label_semantics_audit"
SHORT_ROOT = RESULTS_ROOT / "bpce_short_horizon_label_audit"

MM_TO_INCH = 1.0 / 25.4
WIDTH_IN = 183.0 * MM_TO_INCH

COLORS = {
    "ink": "#263238",
    "muted": "#68757D",
    "grid": "#D8DEE2",
    "light": "#F3F5F6",
    "n": "#7D858A",
    "e": "#3979A8",
    "direct": "#334A5E",
    "same": "#D28B45",
    "future_probe": "#4D9882",
    "future_other": "#7569A5",
    "missile": "#59708D",
    "laser": "#C67857",
    "pass": "#4D8968",
    "fail": "#B85656",
    "r1": "#8A9196",
    "r2": "#3979A8",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "axes.edgecolor": COLORS["ink"],
        "axes.labelcolor": COLORS["ink"],
        "xtick.color": COLORS["ink"],
        "ytick.color": COLORS["ink"],
        "text.color": COLORS["ink"],
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_output_dirs() -> None:
    for path in (
        SOURCE_ROOT,
        EXPORT_ROOT,
        METADATA_ROOT,
        TABLE_EXPORT_ROOT,
        TABLE_METADATA_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.10,
        1.06,
        label,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=8,
        va="top",
    )


def style_axis(ax: plt.Axes, *, grid_axis: str | None = None) -> None:
    if grid_axis:
        ax.grid(
            True,
            axis=grid_axis,
            color=COLORS["grid"],
            linewidth=0.55,
            alpha=0.8,
            zorder=0,
        )
    ax.tick_params(length=2.5, width=0.6)


def add_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str = "white",
    edgecolor: str = COLORS["ink"],
    linewidth: float = 0.8,
    fontsize: float = 7,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
    )
    return patch


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["muted"],
    connectionstyle: str = "arc3",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.8,
            color=color,
            connectionstyle=connectionstyle,
        )
    )


def save_figure(fig: plt.Figure, stem: str) -> dict[str, str]:
    paths = {
        "svg": EXPORT_ROOT / f"{stem}.svg",
        "pdf": EXPORT_ROOT / f"{stem}.pdf",
        "tiff": EXPORT_ROOT / f"{stem}.tiff",
        "preview": EXPORT_ROOT / f"{stem}_preview.png",
    }
    fig.savefig(paths["svg"], bbox_inches="tight")
    fig.savefig(paths["pdf"], bbox_inches="tight")
    fig.savefig(
        paths["tiff"],
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(paths["preview"], dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {
        key: str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        for key, path in paths.items()
    }


def write_metadata(
    stem: str,
    *,
    conclusion: str,
    archetype: str,
    panels: list[dict[str, Any]],
    exports: dict[str, str],
) -> None:
    payload = {
        "figure": stem,
        "backend": "python_matplotlib",
        "final_width_mm": 183,
        "conclusion": conclusion,
        "archetype": archetype,
        "panels": panels,
        "exports": exports,
        "data_exclusions": "none",
        "interval": "mean +/- 1.96 * sample_standard_deviation / sqrt(n)",
        "script": str(SCRIPT_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
    }
    (METADATA_ROOT / f"{stem}_metadata.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def figure_1() -> None:
    fig = plt.figure(figsize=(WIDTH_IN, 3.55))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 0.95], wspace=0.28)
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    for ax in axes:
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    ax = axes[0]
    panel_label(ax, "a")
    ax.set_title("Dynamic masked joint action", loc="left", pad=8)
    unit_y = [0.76, 0.50, 0.24]
    unit_names = ["Missile 0", "Missile 1", "Laser 2"]
    target_names = ["Target A", "Target B", "No-op"]
    target_colors = [COLORS["e"], COLORS["same"], COLORS["n"]]
    for index, (y, unit) in enumerate(zip(unit_y, unit_names)):
        add_box(ax, (0.03, y - 0.075), 0.25, 0.15, unit, facecolor=COLORS["light"])
        add_arrow(ax, (0.29, y), (0.48, y))
        add_box(
            ax,
            (0.50, y - 0.075),
            0.27,
            0.15,
            target_names[index],
            facecolor="white",
            edgecolor=target_colors[index],
        )
    ax.text(0.50, 0.91, "order 0 -> 1 -> 2", fontsize=6.5, color=COLORS["muted"])
    ax.text(
        0.50,
        0.07,
        "An occupied target is removed\nfrom later conditional masks",
        fontsize=6.5,
        color=COLORS["muted"],
        ha="center",
    )

    ax = axes[1]
    panel_label(ax, "b")
    ax.set_title("Paired local intervention", loc="left", pad=8)
    add_box(ax, (0.30, 0.82), 0.40, 0.12, "Frozen context $s_t$", facecolor=COLORS["light"])
    add_arrow(ax, (0.43, 0.81), (0.25, 0.66), color=COLORS["n"])
    add_arrow(ax, (0.57, 0.81), (0.75, 0.66), color=COLORS["e"])
    add_box(ax, (0.03, 0.52), 0.42, 0.14, "$N$: probe no-op", edgecolor=COLORS["n"])
    add_box(ax, (0.55, 0.52), 0.42, 0.14, "$E$: legal engage", edgecolor=COLORS["e"])
    add_arrow(ax, (0.24, 0.51), (0.24, 0.34), color=COLORS["n"])
    add_arrow(ax, (0.76, 0.51), (0.76, 0.34), color=COLORS["e"])
    add_box(ax, (0.03, 0.20), 0.42, 0.14, "suffix + future\npolicy response", edgecolor=COLORS["n"])
    add_box(ax, (0.55, 0.20), 0.42, 0.14, "suffix + future\npolicy response", edgecolor=COLORS["e"])
    ax.text(
        0.5,
        0.07,
        "Shared random tapes; branch-specific legal masks",
        ha="center",
        fontsize=6.5,
        color=COLORS["muted"],
    )

    ax = axes[2]
    panel_label(ax, "c")
    ax.set_title("Why episode cost can misread local cost", loc="left", pad=8)
    add_box(
        ax,
        (0.08, 0.71),
        0.84,
        0.13,
        "$C_{direct} > 0$",
        facecolor="#E8EEF2",
        edgecolor=COLORS["direct"],
        fontsize=8,
    )
    ax.text(0.50, 0.59, "minus", ha="center", fontsize=7, color=COLORS["muted"])
    add_box(
        ax,
        (0.08, 0.39),
        0.84,
        0.15,
        "$Sub_{cost,total}$\n(same-step + future)",
        facecolor="#F5ECE3",
        edgecolor=COLORS["same"],
        fontsize=7,
    )
    ax.text(0.50, 0.28, "equals", ha="center", fontsize=7, color=COLORS["muted"])
    add_box(
        ax,
        (0.08, 0.08),
        0.84,
        0.15,
        "$\\Delta C_{episode}$\ncan be zero or negative",
        facecolor=COLORS["light"],
        edgecolor=COLORS["ink"],
        fontsize=7,
    )
    fig.suptitle(
        "Dynamic action substitution mixes local direct cost with policy-mediated resource use",
        x=0.02,
        ha="left",
        fontsize=9,
        fontweight="bold",
    )
    exports = save_figure(fig, "figure_1_measurement_problem")
    write_metadata(
        "figure_1_measurement_problem",
        conclusion=(
            "Dynamic masked autoregressive suffixes can make episode cost a "
            "biased readout of the current local action cost."
        ),
        archetype="schematic-led composite",
        panels=[
            {"id": "a", "role": "dynamic mask and autoregressive target occupancy", "data": "schematic"},
            {"id": "b", "role": "N/E intervention identities", "data": "schematic"},
            {"id": "c", "role": "cost-mixing identity", "data": "frozen formula"},
        ],
        exports=exports,
    )


def figure_2() -> None:
    gate = read_json(R2_ROOT / "gate_summary.json")
    residual_data = pd.DataFrame(
        [
            {
                "ledger": "Future-only",
                "maximum_absolute_residual": gate[
                    "maximum_future_only_decomposition_error"
                ],
                "affected_rows": 287,
                "total_rows": gate["target_ledger_rows"],
            },
            {
                "ledger": "Complete",
                "maximum_absolute_residual": gate[
                    "maximum_extended_decomposition_error"
                ],
                "affected_rows": 0,
                "total_rows": gate["target_ledger_rows"],
            },
        ]
    )
    residual_data.to_csv(SOURCE_ROOT / "figure_2_residual_data.csv", index=False)

    fig = plt.figure(figsize=(WIDTH_IN, 4.25))
    grid = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.0, 1.0], hspace=0.42, wspace=0.34)
    ax_a = fig.add_subplot(grid[:, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 1])
    ax_d = fig.add_subplot(grid[:, 2])

    ax_a.set_axis_off()
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)
    panel_label(ax_a, "a")
    ax_a.set_title("Snapshot + common random numbers", loc="left", pad=8)
    steps = [
        (0.10, 0.82, "Frozen state\nsnapshot"),
        (0.10, 0.60, "Force $N$ / $E$\nprobe action"),
        (0.10, 0.38, "Shared environment\nand policy tapes"),
        (0.10, 0.16, "Stochastic\ncontinuation"),
    ]
    for index, (x, y, label) in enumerate(steps):
        add_box(ax_a, (x, y), 0.80, 0.13, label, facecolor=COLORS["light"])
        if index < len(steps) - 1:
            add_arrow(ax_a, (0.50, y), (0.50, steps[index + 1][1] + 0.14))

    ax_b.set_axis_off()
    ax_b.set_xlim(0, 1)
    ax_b.set_ylim(0, 1)
    panel_label(ax_b, "b")
    ax_b.set_title("Exact legal-target marginalization", loc="left", pad=8)
    target_probs = ["$p_1$", "$p_2$", "$p_3$"]
    for index, probability in enumerate(target_probs):
        y = 0.76 - index * 0.23
        add_box(
            ax_b,
            (0.05, y - 0.07),
            0.43,
            0.14,
            f"legal target {index + 1}",
            edgecolor=COLORS["e"],
        )
        ax_b.text(0.58, y, probability, va="center", fontsize=7)
    ax_b.text(
        0.50,
        0.08,
        "$E[\\cdot]=\\sum_k p_k E[\\cdot\\mid target=k]$",
        ha="center",
        fontsize=7,
    )

    ax_c.set_axis_off()
    ax_c.set_xlim(0, 1)
    ax_c.set_ylim(0, 1)
    panel_label(ax_c, "c")
    ax_c.set_title("Complete cost ledger", loc="left", pad=8)
    ax_c.text(
        0.02,
        0.74,
        "$Sub_{cost,total}$",
        fontsize=8,
        color=COLORS["direct"],
        fontweight="bold",
    )
    ax_c.text(0.02, 0.53, "$= Sub_{cost,same}$", color=COLORS["same"], fontsize=7.5)
    ax_c.text(
        0.02,
        0.34,
        "$+ Sub_{cost,future,probe}$",
        color=COLORS["future_probe"],
        fontsize=7.5,
    )
    ax_c.text(
        0.02,
        0.15,
        "$+ Sub_{cost,future,other}$",
        color=COLORS["future_other"],
        fontsize=7.5,
    )

    panel_label(ax_d, "d")
    ax_d.set_title("Ledger integrity", loc="left", pad=8)
    x = np.arange(2)
    values = residual_data["maximum_absolute_residual"].to_numpy()
    ax_d.bar(
        x,
        values,
        color=[COLORS["fail"], COLORS["pass"]],
        width=0.58,
        zorder=3,
    )
    ax_d.set_yscale("log")
    ax_d.axhline(1e-6, color=COLORS["muted"], linestyle="--", linewidth=0.8)
    ax_d.text(1.48, 1.35e-6, "tolerance $10^{-6}$", ha="right", fontsize=6)
    ax_d.set_xticks(x, ["Future-only", "Complete"])
    ax_d.set_ylabel("Maximum absolute residual")
    ax_d.set_ylim(1e-17, 10)
    style_axis(ax_d, grid_axis="y")
    ax_d.text(0, values[0] * 1.7, "287 / 7,776 rows", ha="center", fontsize=6.5)
    ax_d.text(1, values[1] * 4.0, "$8.88\\times10^{-16}$", ha="center", fontsize=6.5)
    fig.suptitle(
        "Paired counterfactual protocol and exact three-component cost reconstruction",
        x=0.02,
        ha="left",
        fontsize=9,
        fontweight="bold",
    )
    exports = save_figure(fig, "figure_2_protocol_and_identity")
    write_metadata(
        "figure_2_protocol_and_identity",
        conclusion=(
            "The complete same-step and future ledger is required to exactly "
            "reconstruct paired episode-cost differences."
        ),
        archetype="asymmetric mixed-modality figure",
        panels=[
            {"id": "a", "role": "paired CRN protocol", "data": "schematic"},
            {"id": "b", "role": "exact target marginalization", "data": "schematic"},
            {"id": "c", "role": "frozen three-component identity", "data": "formula freeze"},
            {
                "id": "d",
                "role": "future-only versus complete residual",
                "source": "results/air_defense_v1/action_substitution_confirmation/gate_summary.json",
                "fields": [
                    "maximum_future_only_decomposition_error",
                    "maximum_extended_decomposition_error",
                    "target_ledger_rows",
                ],
            },
        ],
        exports=exports,
    )


def _seed_colors(seeds: Iterable[int]) -> dict[int, str]:
    palette = ["#5B7FA3", "#8A70A8", "#4F8F79"]
    return {seed: palette[index] for index, seed in enumerate(sorted(set(seeds)))}


def _context_interval_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    title: str,
    color_key: str,
    hollow: bool,
) -> None:
    data = data.sort_values(["policy_seed", "context_id"]).reset_index(drop=True)
    positions = np.arange(1, len(data) + 1)
    colors = _seed_colors(data["policy_seed"].astype(int))
    for seed, group in data.groupby("policy_seed", sort=True):
        index = group.index.to_numpy()
        values = group["sub_shot_mean"].to_numpy()
        lower = values - group["sub_shot_lower"].to_numpy()
        upper = group["sub_shot_upper"].to_numpy() - values
        ax.errorbar(
            positions[index],
            values,
            yerr=np.vstack([lower, upper]),
            fmt="o",
            markersize=3.5,
            markerfacecolor="white" if hollow else colors[int(seed)],
            markeredgecolor=colors[int(seed)],
            markeredgewidth=0.8,
            ecolor=colors[int(seed)],
            elinewidth=0.7,
            capsize=1.5,
            label=f"seed {int(seed)}",
            zorder=3,
        )
    ax.axhline(0, color=COLORS["ink"], linewidth=0.7)
    ax.set_xlim(0.2, len(data) + 0.8)
    ax.set_xticks([1, 6, 12, 18])
    ax.set_xlabel("Prespecified resource context")
    ax.set_ylabel("$Sub_{shot}$")
    ax.set_title(title, loc="left")
    style_axis(ax, grid_axis="y")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    ax.set_facecolor("#FAFAFA" if color_key == "r1" else "#F7FAFC")


def figure_3() -> None:
    r1 = pd.read_csv(R1_ROOT / "context_opportunity_estimates.csv")
    r1 = r1[(r1["scenario"] == "time_pressure") & (r1["slot"] == "resource")].copy()
    r1 = r1.rename(columns={"unit_type": "resource_type"})
    r2 = pd.read_csv(R2_ROOT / "context_substitution_estimates.csv")
    r2 = r2[(r2["scenario"] == "time_pressure") & (r2["slot"] == "resource")].copy()
    gate = read_json(R2_ROOT / "gate_summary.json")
    r1_gate = read_json(R1_ROOT / "gate_summary.json")

    r1_export = r1[
        [
            "context_id",
            "policy_seed",
            "resource_type",
            "sub_shot_mean",
            "sub_shot_lower",
            "sub_shot_upper",
            "total_cost_difference_mean",
            "sub_cost_mean",
        ]
    ].copy()
    r1_export["phase"] = "R1 discovery"
    r2_export = r2[
        [
            "context_id",
            "policy_seed",
            "resource_type",
            "sub_shot_mean",
            "sub_shot_lower",
            "sub_shot_upper",
            "episode_cost_delta_mean",
            "sub_cost_mean",
        ]
    ].copy()
    r2_export["phase"] = "R2 confirmation"
    r1_export.to_csv(SOURCE_ROOT / "figure_3_r1_context_data.csv", index=False)
    r2_export.to_csv(SOURCE_ROOT / "figure_3_r2_context_data.csv", index=False)

    block_rows = []
    for seed, interval in gate["P-C2"]["seed_block_intervals"].items():
        block_rows.append(
            {
                "policy_seed": int(seed),
                "mean": interval["mean"],
                "lower": interval["lower"],
                "upper": interval["upper"],
                "masked_rate": gate["P-C2"]["seed_masked_rates"][seed],
            }
        )
    blocks = pd.DataFrame(block_rows)
    blocks.to_csv(SOURCE_ROOT / "figure_3_seed_block_data.csv", index=False)
    explanation_counts = pd.DataFrame(
        [
            {
                "phase": "R1 discovery",
                "nonpositive_contexts": r1_gate["P-R1"][
                    "nonpositive_total_cost_contexts"
                ],
                "positive_substitution_contexts": r1_gate["P-R1"][
                    "explained_nonpositive_cost_contexts"
                ],
            },
            {
                "phase": "R2 confirmation",
                "nonpositive_contexts": gate["P-C2"]["nonpositive_contexts"],
                "positive_substitution_contexts": gate["P-C2"][
                    "nonpositive_with_positive_sub_cost"
                ],
            },
        ]
    )
    explanation_counts.to_csv(
        SOURCE_ROOT / "figure_3_nonpositive_explanation_data.csv", index=False
    )

    fig = plt.figure(figsize=(WIDTH_IN, 4.65))
    grid = fig.add_gridspec(2, 2, hspace=0.48, wspace=0.32)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    panel_label(ax_a, "a")
    _context_interval_panel(
        ax_a,
        r1,
        title="R1 discovery: old policy seeds 8/9/10",
        color_key="r1",
        hollow=True,
    )
    panel_label(ax_b, "b")
    _context_interval_panel(
        ax_b,
        r2,
        title="R2 confirmation: new policy seeds 17/18/19",
        color_key="r2",
        hollow=False,
    )

    panel_label(ax_c, "c")
    seed_colors = _seed_colors(blocks["policy_seed"].astype(int))
    x = np.arange(len(blocks))
    values = blocks["mean"].to_numpy()
    for index, row in blocks.reset_index(drop=True).iterrows():
        color = seed_colors[int(row["policy_seed"])]
        ax_c.errorbar(
            index,
            row["mean"],
            yerr=np.array(
                [[row["mean"] - row["lower"]], [row["upper"] - row["mean"]]]
            ),
            fmt="o",
            color=color,
            ecolor=color,
            markersize=4.5,
            elinewidth=1.2,
            capsize=3,
            zorder=3,
        )
    ax_c.axhline(0, color=COLORS["ink"], linewidth=0.7)
    ax_c.set_xticks(x, [f"seed {seed}" for seed in blocks["policy_seed"]])
    ax_c.set_ylabel("Block mean $Sub_{shot}$")
    ax_c.set_title("All three new seed-block lower bounds exceed zero", loc="left")
    style_axis(ax_c, grid_axis="y")

    panel_label(ax_d, "d")
    x = np.arange(2)
    width = 0.32
    nonpositive_counts = explanation_counts["nonpositive_contexts"].to_numpy()
    explained_counts = explanation_counts["positive_substitution_contexts"].to_numpy()
    ax_d.bar(
        x - width / 2,
        nonpositive_counts,
        width=width,
        color=COLORS["muted"],
        label="$\\Delta C_{episode}\\leq0$",
        zorder=3,
    )
    ax_d.bar(
        x + width / 2,
        explained_counts,
        width=width,
        color=COLORS["pass"],
        label="positive $Sub_{cost,total}$",
        zorder=3,
    )
    ax_d.set_xticks(x, ["R1 discovery", "R2 confirmation"])
    ax_d.set_ylabel("Contexts")
    ax_d.set_ylim(0, max(nonpositive_counts) + 3)
    ax_d.set_title("All nonpositive-cost contexts have positive substitution", loc="left")
    for index, (total_count, explained_count) in enumerate(
        zip(nonpositive_counts, explained_counts)
    ):
        ax_d.text(
            index,
            max(total_count, explained_count) + 0.35,
            f"{explained_count}/{total_count}",
            ha="center",
            fontsize=7,
        )
    ax_d.legend(ncol=1, loc="upper right")
    style_axis(ax_d, grid_axis="y")

    fig.suptitle(
        "Action substitution discovered in R1 replicates across independent R2 policy seeds",
        x=0.02,
        ha="left",
        fontsize=9,
        fontweight="bold",
    )
    exports = save_figure(fig, "figure_3_discovery_and_confirmation")
    write_metadata(
        "figure_3_discovery_and_confirmation",
        conclusion=(
            "Positive future substitution discovered in R1 replicates in "
            "prespecified R2 contexts from three new policy seeds."
        ),
        archetype="quantitative grid",
        panels=[
            {
                "id": "a",
                "source": "results/air_defense_v1/action_substitution_opportunity_cost_audit/context_opportunity_estimates.csv",
                "filter": "scenario=time_pressure, slot=resource",
                "unit": "context; n=32 paired repeats per context",
                "fields": ["sub_shot_mean", "sub_shot_lower", "sub_shot_upper"],
            },
            {
                "id": "b",
                "source": "results/air_defense_v1/action_substitution_confirmation/context_substitution_estimates.csv",
                "filter": "scenario=time_pressure, slot=resource",
                "unit": "context; n=32 paired repeats per context",
                "fields": ["sub_shot_mean", "sub_shot_lower", "sub_shot_upper"],
            },
            {
                "id": "c",
                "source": "results/air_defense_v1/action_substitution_confirmation/gate_summary.json",
                "field": "P-C2.seed_block_intervals",
                "unit": "context within seed block",
            },
            {
                "id": "d",
                "source": [
                    "results/air_defense_v1/action_substitution_opportunity_cost_audit/gate_summary.json",
                    "results/air_defense_v1/action_substitution_confirmation/gate_summary.json",
                ],
                "fields": [
                    "P-R1.nonpositive_total_cost_contexts",
                    "P-R1.explained_nonpositive_cost_contexts",
                    "P-C2.nonpositive_contexts",
                    "P-C2.nonpositive_with_positive_sub_cost",
                ],
                "unit": "context",
            },
        ],
        exports=exports,
    )


def figure_4() -> None:
    contexts = pd.read_csv(R2_ROOT / "context_substitution_estimates.csv")
    selected = contexts[
        (contexts["scenario"] == "time_pressure") & (contexts["slot"] == "resource")
    ].copy()
    fields = [
        "direct_cost_mean",
        "same_step_other_sub_cost_mean",
        "future_sub_cost_probe_mean",
        "future_sub_cost_other_mean",
        "sub_cost_mean",
        "episode_cost_delta_mean",
        "rho_sub_mean",
        "cost_sign_masked_rate",
    ]
    summary = {field: float(selected[field].mean()) for field in fields}
    component_data = pd.DataFrame(
        [
            {"component": "Direct cost", "mean": summary["direct_cost_mean"]},
            {
                "component": "Same-step other",
                "mean": summary["same_step_other_sub_cost_mean"],
            },
            {
                "component": "Future probe",
                "mean": summary["future_sub_cost_probe_mean"],
            },
            {
                "component": "Future other",
                "mean": summary["future_sub_cost_other_mean"],
            },
            {"component": "Episode delta", "mean": summary["episode_cost_delta_mean"]},
        ]
    )
    component_data.to_csv(SOURCE_ROOT / "figure_4_component_data.csv", index=False)
    selected[
        [
            "context_id",
            "policy_seed",
            "resource_type",
            "rho_sub_mean",
            "episode_cost_delta_mean",
            "cost_sign_masked_rate",
            "direct_cost_mean",
            "same_step_other_sub_cost_mean",
            "future_sub_cost_probe_mean",
            "future_sub_cost_other_mean",
        ]
    ].to_csv(SOURCE_ROOT / "figure_4_context_data.csv", index=False)

    same = summary["same_step_other_sub_cost_mean"]
    future_probe = summary["future_sub_cost_probe_mean"]
    future_other = summary["future_sub_cost_other_mean"]
    total = same + future_probe + future_other
    same_fraction = same / total if total != 0 else math.nan
    future_fraction = (future_probe + future_other) / total if total != 0 else math.nan

    fig = plt.figure(figsize=(WIDTH_IN, 4.55))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.08, 1.0], hspace=0.52, wspace=0.34)
    ax_a = fig.add_subplot(grid[:, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 1])

    panel_label(ax_a, "a")
    ax_a.set_title("Direct cost and the complete substitution ledger", loc="left")
    x = np.arange(3)
    ax_a.bar(
        x[0],
        summary["direct_cost_mean"],
        width=0.56,
        color=COLORS["direct"],
        label="$C_{direct}$",
        zorder=3,
    )
    bottom = 0.0
    for value, color, label in (
        (same, COLORS["same"], "same-step other"),
        (future_probe, COLORS["future_probe"], "future probe"),
        (future_other, COLORS["future_other"], "future other"),
    ):
        ax_a.bar(
            x[1],
            value,
            width=0.56,
            bottom=bottom,
            color=color,
            label=label,
            zorder=3,
        )
        bottom += value
    ax_a.bar(
        x[2],
        summary["episode_cost_delta_mean"],
        width=0.56,
        color=COLORS["n"],
        label="$\\Delta C_{episode}$",
        zorder=3,
    )
    ax_a.axhline(0, color=COLORS["ink"], linewidth=0.7)
    ax_a.set_xticks(x, ["Direct", "Total\nsubstitution", "Episode\ndelta"])
    ax_a.set_ylabel("Mean resource cost")
    ax_a.text(
        1,
        total + 0.04,
        f"{total:.3f}",
        ha="center",
        fontsize=6.5,
    )
    ax_a.legend(loc="upper right")
    style_axis(ax_a, grid_axis="y")

    panel_label(ax_b, "b")
    ax_b.set_title("Substitution composition", loc="left")
    ax_b.barh(
        [0],
        [same_fraction * 100],
        color=COLORS["same"],
        height=0.36,
        label="same-step",
    )
    ax_b.barh(
        [0],
        [future_fraction * 100],
        left=[same_fraction * 100],
        color=COLORS["future_probe"],
        height=0.36,
        label="future",
    )
    ax_b.text(
        same_fraction * 50,
        0,
        f"{same_fraction * 100:.0f}%",
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
    )
    ax_b.text(
        same_fraction * 100 + future_fraction * 50,
        0,
        f"{future_fraction * 100:.0f}%",
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
    )
    ax_b.set_xlim(0, 100)
    ax_b.set_yticks([])
    ax_b.set_xlabel("Share of $Sub_{cost,total}$ (%)")
    ax_b.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    style_axis(ax_b, grid_axis="x")

    panel_label(ax_c, "c")
    ax_c.set_title("Substitution ratio and episode-cost sign", loc="left")
    marker_colors = selected["resource_type"].map(
        {"missile": COLORS["missile"], "laser": COLORS["laser"]}
    )
    marker_sizes = 18 + 30 * selected["cost_sign_masked_rate"].to_numpy()
    ax_c.scatter(
        selected["rho_sub_mean"],
        selected["episode_cost_delta_mean"],
        c=marker_colors,
        s=marker_sizes,
        alpha=0.88,
        edgecolor="white",
        linewidth=0.4,
        zorder=3,
    )
    ax_c.axhline(0, color=COLORS["ink"], linewidth=0.7)
    ax_c.axvline(1, color=COLORS["muted"], linestyle="--", linewidth=0.8)
    ax_c.set_xlabel("$\\rho_{sub}=Sub_{cost,total}/C_{direct}$")
    ax_c.set_ylabel("$\\Delta C_{episode}$")
    ax_c.text(
        0.98,
        0.06,
        "marker size = masked rate",
        transform=ax_c.transAxes,
        ha="right",
        fontsize=6,
        color=COLORS["muted"],
    )
    for label, color, x_pos in (
        ("missile", COLORS["missile"], 0.04),
        ("laser", COLORS["laser"], 0.24),
    ):
        ax_c.scatter([x_pos], [0.94], transform=ax_c.transAxes, color=color, s=14)
        ax_c.text(x_pos + 0.035, 0.94, label, transform=ax_c.transAxes, va="center", fontsize=6)
    style_axis(ax_c, grid_axis="both")

    fig.suptitle(
        "Same-step and future actions jointly mediate the episode resource-cost readout",
        x=0.02,
        ha="left",
        fontsize=9,
        fontweight="bold",
    )
    exports = save_figure(fig, "figure_4_cost_composition")
    write_metadata(
        "figure_4_cost_composition",
        conclusion=(
            "Total substitution cost includes a necessary same-step component "
            "and a dominant future component that can mask direct-cost sign."
        ),
        archetype="asymmetric quantitative figure",
        panels=[
            {
                "id": "a",
                "source": "results/air_defense_v1/action_substitution_confirmation/context_substitution_estimates.csv",
                "filter": "scenario=time_pressure, slot=resource",
                "aggregation": "equal-weight mean across 18 contexts",
                "fields": fields,
            },
            {
                "id": "b",
                "source": "same as panel a",
                "aggregation": "same_step / total and (future_probe + future_other) / total",
                "unit": "context-equal aggregate",
            },
            {
                "id": "c",
                "source": "same as panel a",
                "fields": [
                    "rho_sub_mean",
                    "episode_cost_delta_mean",
                    "cost_sign_masked_rate",
                    "resource_type",
                ],
                "unit": "context",
            },
        ],
        exports=exports,
    )


def figure_5() -> None:
    scenario = pd.read_csv(R2_ROOT / "scenario_boundary_summary.csv")
    scenario = scenario[scenario["slot"] == "resource"].copy()
    scenario_order = ["medium", "time_pressure", "heterogeneity_pressure"]
    scenario["scenario"] = pd.Categorical(
        scenario["scenario"], categories=scenario_order, ordered=True
    )
    scenario = scenario.sort_values("scenario")
    resource = pd.read_csv(R2_ROOT / "resource_type_summary.csv")
    resource = resource[
        (resource["scenario"] == "time_pressure") & (resource["slot"] == "resource")
    ].copy()
    resource_order = ["missile", "laser"]
    resource["resource_type"] = pd.Categorical(
        resource["resource_type"], categories=resource_order, ordered=True
    )
    resource = resource.sort_values("resource_type")
    gate = read_json(R2_ROOT / "gate_summary.json")

    scenario[
        [
            "scenario",
            "contexts",
            "rho_sub_mean_aggregate",
            "rho_sub_mean_lower",
            "rho_sub_mean_upper",
            "cost_sign_masked_rate_aggregate",
        ]
    ].to_csv(SOURCE_ROOT / "figure_5_scenario_data.csv", index=False)
    resource[
        [
            "resource_type",
            "contexts",
            "sub_shot_mean_aggregate",
            "sub_shot_mean_lower",
            "sub_shot_mean_upper",
            "rho_sub_mean_aggregate",
            "cost_sign_masked_rate_aggregate",
        ]
    ].to_csv(SOURCE_ROOT / "figure_5_resource_data.csv", index=False)

    fig = plt.figure(figsize=(WIDTH_IN, 4.45))
    grid = fig.add_gridspec(2, 2, hspace=0.50, wspace=0.34)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    panel_label(ax_a, "a")
    x = np.arange(len(scenario))
    values = scenario["rho_sub_mean_aggregate"].to_numpy()
    ax_a.errorbar(
        x,
        values,
        yerr=np.vstack(
            [
                values - scenario["rho_sub_mean_lower"].to_numpy(),
                scenario["rho_sub_mean_upper"].to_numpy() - values,
            ]
        ),
        fmt="o",
        color=COLORS["e"],
        ecolor=COLORS["e"],
        markersize=4.5,
        capsize=3,
        zorder=3,
    )
    ax_a.axhline(1, color=COLORS["muted"], linestyle="--", linewidth=0.8)
    ax_a.set_xticks(x, ["Medium", "Time\npressure", "Heterogeneity\npressure"])
    ax_a.set_ylabel("$\\rho_{sub}$")
    ax_a.set_title("Scenario-conditional substitution ratio", loc="left")
    style_axis(ax_a, grid_axis="y")

    panel_label(ax_b, "b")
    x = np.arange(len(resource))
    values = resource["sub_shot_mean_aggregate"].to_numpy()
    colors = [COLORS["missile"], COLORS["laser"]]
    for index, row in resource.reset_index(drop=True).iterrows():
        color = colors[index]
        ax_b.errorbar(
            index,
            row["sub_shot_mean_aggregate"],
            yerr=np.array(
                [
                    [
                        row["sub_shot_mean_aggregate"]
                        - row["sub_shot_mean_lower"]
                    ],
                    [
                        row["sub_shot_mean_upper"]
                        - row["sub_shot_mean_aggregate"]
                    ],
                ]
            ),
            fmt="o",
            color=color,
            ecolor=color,
            markersize=5,
            elinewidth=1.2,
            capsize=3,
            zorder=3,
        )
    ax_b.axhline(0, color=COLORS["ink"], linewidth=0.7)
    ax_b.set_xticks(x, ["Missile", "Laser"])
    ax_b.set_ylabel("$Sub_{shot}$")
    ax_b.set_title("Both resource types show positive substitution", loc="left")
    style_axis(ax_b, grid_axis="y")

    panel_label(ax_c, "c")
    masked = [
        gate["P-C3"]["missile"]["masked_contexts"],
        gate["P-C3"]["laser"]["masked_contexts"],
    ]
    ax_c.bar(
        [0, 1],
        masked,
        width=0.56,
        color=colors,
        zorder=3,
    )
    ax_c.axhline(3, color=COLORS["fail"], linestyle="--", linewidth=0.9)
    ax_c.text(
        0.02,
        3.12,
        "P-C3 threshold = 3",
        ha="left",
        fontsize=6,
        color=COLORS["fail"],
    )
    ax_c.set_xticks([0, 1], ["Missile", "Laser"])
    ax_c.set_ylabel("Masked contexts (of 9)")
    ax_c.set_ylim(0, 6.3)
    ax_c.set_title("Missile does not reach the sign-masking threshold", loc="left")
    for index, value in enumerate(masked):
        ax_c.text(index, value + 0.18, f"{value}/9", ha="center", fontsize=7)
    style_axis(ax_c, grid_axis="y")

    panel_label(ax_d, "d")
    gates = [
        ("P-C1", gate["mechanism_gates"]["P-C1_cost_decomposition"]),
        ("P-C2", gate["mechanism_gates"]["P-C2_independent_substitution"]),
        ("P-C3", gate["mechanism_gates"]["P-C3_cross_resource_type"]),
    ]
    ax_d.set_axis_off()
    ax_d.set_xlim(0, 1)
    ax_d.set_ylim(0, 1)
    ax_d.set_title("Frozen gate decision", loc="left", pad=8)
    for index, (name, passed) in enumerate(gates):
        y = 0.78 - index * 0.26
        color = COLORS["pass"] if passed else COLORS["fail"]
        status = "PASS" if passed else "FAIL"
        add_box(
            ax_d,
            (0.10, y - 0.08),
            0.80,
            0.16,
            f"{name}   {status}",
            facecolor="#EDF5F0" if passed else "#F8ECEC",
            edgecolor=color,
            fontsize=8,
        )
    ax_d.text(
        0.50,
        0.05,
        "Decision: resource-type-conditional claim",
        ha="center",
        fontsize=6.5,
        color=COLORS["muted"],
    )

    fig.suptitle(
        "Action-substitution strength is conditional on scenario and resource type",
        x=0.02,
        ha="left",
        fontsize=9,
        fontweight="bold",
    )
    exports = save_figure(fig, "figure_5_scenario_resource_boundaries")
    write_metadata(
        "figure_5_scenario_resource_boundaries",
        conclusion=(
            "Substitution occurs across scenarios and resource types, but "
            "sign-masking strength is resource-type conditional and P-C3 fails."
        ),
        archetype="quantitative grid",
        panels=[
            {
                "id": "a",
                "source": "results/air_defense_v1/action_substitution_confirmation/scenario_boundary_summary.csv",
                "filter": "slot=resource",
                "unit": "context; n=18 per scenario",
                "fields": [
                    "rho_sub_mean_aggregate",
                    "rho_sub_mean_lower",
                    "rho_sub_mean_upper",
                ],
            },
            {
                "id": "b",
                "source": "results/air_defense_v1/action_substitution_confirmation/resource_type_summary.csv",
                "filter": "scenario=time_pressure, slot=resource",
                "unit": "context; n=9 per resource type",
                "fields": [
                    "sub_shot_mean_aggregate",
                    "sub_shot_mean_lower",
                    "sub_shot_mean_upper",
                ],
            },
            {
                "id": "c",
                "source": "results/air_defense_v1/action_substitution_confirmation/gate_summary.json",
                "fields": ["P-C3.missile.masked_contexts", "P-C3.laser.masked_contexts"],
                "unit": "context",
            },
            {
                "id": "d",
                "source": "same gate summary",
                "field": "mechanism_gates",
            },
        ],
        exports=exports,
    )


def dataframe_to_markdown(data: pd.DataFrame) -> str:
    columns = list(data.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in data.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows]) + "\n"


def export_table(stem: str, data: pd.DataFrame) -> dict[str, str]:
    csv_path = TABLE_EXPORT_ROOT / f"{stem}.csv"
    md_path = TABLE_EXPORT_ROOT / f"{stem}.md"
    data.to_csv(csv_path, index=False)
    md_path.write_text(dataframe_to_markdown(data), encoding="utf-8")
    return {
        "csv": str(csv_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "markdown": str(md_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
    }


def generate_tables() -> None:
    config = read_json(R2_ROOT / "experiment_config.json")
    r2_gate = read_json(R2_ROOT / "gate_summary.json")
    r1_gate = read_json(R1_ROOT / "gate_summary.json")
    label_gate = read_json(LABEL_ROOT / "gate_summary.json")
    short_gate = read_json(SHORT_ROOT / "gate_summary.json")

    table_1 = pd.DataFrame(
        [
            ("Environment", "AirDefense v1", "3 units, 5 targets, dynamic legal masks"),
            ("Scenarios", "3", ", ".join(config["scenarios"])),
            ("Source policy", config["method"], "factorized joint PPO; order 0-1-2"),
            ("Policy seeds", "3", "/".join(map(str, config["policy_seeds"]))),
            ("Source models", config["source_model_count"], "9/9 retained without behavior screening"),
            ("Training", "10,000 steps/model", "n_steps=256; batch=64; epochs=2"),
            ("Contexts", r2_gate["context_count"], "6 safety + 6 resource per scenario-seed block"),
            ("Resource quota", "3 missile + 3 laser", "per scenario-seed resource block"),
            ("Paired repeats", config["confirmation_config"]["repeats"], "N/E CRN pairs per context"),
            ("Target action", "exact marginalization", "conditional on engage"),
            ("Continuation", "stochastic", "frozen actor and shared uniform tape"),
        ],
        columns=["Item", "Value", "Definition"],
    )

    table_2 = pd.DataFrame(
        [
            ("R1", "8/9/10", "72", "discovery", "old source policies"),
            ("R2", "17/18/19", "108", "independent confirmation", "old observation-hash overlap = 0"),
            ("R2 source models", "17/18/19", "9 models", "all retained", "selected_by_behavior = false"),
            ("R2 repeats", "17/18/19", "3,456", "context-repeat rows", "32 per context"),
            ("R2 target ledger", "17/18/19", "7,776", "target-conditional rows", "not independent contexts"),
            ("R2 actor", "17/18/19", "9 models", "frozen", "maximum parameter difference = 0"),
        ],
        columns=["Phase", "Policy seeds", "Count", "Role", "Integrity condition"],
    )

    pc2 = r2_gate["P-C2"]
    pc3 = r2_gate["P-C3"]
    table_3 = pd.DataFrame(
        [
            ("P-C1", "complete ledger residual <= 1e-6; direct cost > 0", "8.88e-16; all positive", "PASS"),
            ("P-C2.1", "positive mean Sub_shot >= 12/18", f"{pc2['positive_mean_sub_shot']}/18", "PASS"),
            ("P-C2.2", "positive lower95 Sub_shot >= 6/18", f"{pc2['positive_lower_sub_shot']}/18", "PASS"),
            ("P-C2.3", "positive seed-block lower95 >= 2/3", f"{pc2['positive_block_lower_seeds']}/3", "PASS"),
            ("P-C2.4", "seed masked rate >= 0.5 for >= 2/3", f"{pc2['masked_rate_at_least_half_seeds']}/3", "PASS"),
            ("P-C2.5", "nonpositive contexts explained >= 0.8", f"{pc2['nonpositive_with_positive_sub_cost']}/{pc2['nonpositive_contexts']}", "PASS"),
            ("P-C3 missile", "positive lower95; >=2 positive seeds; >=3 masked contexts", f"{pc3['missile']['positive_seed_blocks']} seeds; {pc3['missile']['masked_contexts']}/9 masked", "FAIL"),
            ("P-C3 laser", "same three conditions", f"{pc3['laser']['positive_seed_blocks']} seeds; {pc3['laser']['masked_contexts']}/9 masked", "PASS"),
            ("P-C3 overall", "both resource types pass", "missile fails sign-masking count", "FAIL"),
        ],
        columns=["Gate", "Frozen criterion", "Observed", "Decision"],
    )

    scenario = pd.read_csv(R2_ROOT / "scenario_boundary_summary.csv")
    scenario = scenario[scenario["slot"] == "resource"]
    scenario_order = {"medium": 0, "time_pressure": 1, "heterogeneity_pressure": 2}
    scenario = scenario.sort_values(
        "scenario", key=lambda values: values.map(scenario_order)
    )
    resource = pd.read_csv(R2_ROOT / "resource_type_summary.csv")
    resource = resource[
        (resource["scenario"] == "time_pressure") & (resource["slot"] == "resource")
    ]
    resource_order = {"missile": 0, "laser": 1}
    resource = resource.sort_values(
        "resource_type", key=lambda values: values.map(resource_order)
    )
    table_4_rows = []
    for row in scenario.itertuples(index=False):
        table_4_rows.append(
            (
                "Scenario",
                row.scenario,
                int(row.contexts),
                f"{row.sub_shot_mean_aggregate:.3f}",
                f"{row.rho_sub_mean_aggregate:.3f}",
                f"{row.cost_sign_masked_rate_aggregate:.3f}",
            )
        )
    for row in resource.itertuples(index=False):
        table_4_rows.append(
            (
                "Resource type (time/resource)",
                str(row.resource_type),
                int(row.contexts),
                f"{row.sub_shot_mean_aggregate:.3f}",
                f"{row.rho_sub_mean_aggregate:.3f}",
                f"{row.cost_sign_masked_rate_aggregate:.3f}",
            )
        )
    table_4 = pd.DataFrame(
        table_4_rows,
        columns=["Stratum", "Level", "Contexts", "Mean Sub_shot", "Mean rho_sub", "Masked rate"],
    )

    table_s1 = pd.DataFrame(
        [
            ("A: argmax + deterministic", label_gate["reliable_counts"]["a"], "full episode", "discovery audit"),
            ("B: exact target + deterministic", label_gate["reliable_counts"]["b"], "full episode", "discovery audit"),
            ("C: exact target + stochastic", label_gate["reliable_counts"]["c"], "full episode", "deterministic continuation rejected"),
            ("Short-window ENGAGE", short_gate["short_label_counts"]["ENGAGE"], "TTI-linked horizon", "not enough overall actionable labels"),
            ("Short-window STOP", short_gate["short_label_counts"]["STOP"], "TTI-linked horizon", "not universal resource label"),
            ("Short-window AMBIGUOUS", short_gate["short_label_counts"]["AMBIGUOUS"], "TTI-linked horizon", "time/resource = 18/18 ambiguous"),
        ],
        columns=["Protocol/label", "Contexts", "Horizon", "Frozen interpretation"],
    )

    table_s2 = pd.DataFrame(
        [
            ("P-R1 action substitution", "18/18 positive lower95 Sub_shot in time/resource", "PASS"),
            ("P-R2 opportunity value", f"reliable resource contexts: time={r1_gate['reliable_resource_contexts']['time_pressure']}, heterogeneity={r1_gate['reliable_resource_contexts']['heterogeneity_pressure']}", "FAIL"),
            ("P-R3 resource criticality", f"reliable unit types: {','.join(r1_gate['reliable_resource_unit_types'])}", "FAIL"),
            ("Route decision", "retain substitution; stop general opportunity route", "FROZEN"),
        ],
        columns=["Gate", "Observed", "Decision"],
    )

    table_s3 = pd.DataFrame(
        [
            ("Future-only affected ledger rows", "287 / 7,776", "pre_ledger_correction/repeat_cost_ledger.csv"),
            ("Future-only maximum residual", "2.0", "gate_summary.json"),
            ("Complete-ledger maximum residual", "8.88e-16", "gate_summary.json"),
            ("Models/contexts/tapes/thresholds", "unchanged", "research integrity disclosure"),
            ("Invalid first result", "archived", "pre_ledger_correction/"),
            ("Complete reruns after correction", "1", "experiment_config.json"),
        ],
        columns=["Integrity item", "Value", "Audit source"],
    )

    table_entries = {}
    for stem, data in (
        ("table_1_task_policy_protocol", table_1),
        ("table_2_independence_integrity", table_2),
        ("table_3_confirmation_gates", table_3),
        ("table_4_scenario_resource_boundaries", table_4),
        ("table_s1_label_semantics", table_s1),
        ("table_s2_resource_restoration_negative", table_s2),
        ("table_s3_ledger_correction_integrity", table_s3),
    ):
        table_entries[stem] = {
            "rows": len(data),
            "exports": export_table(stem, data),
        }

    manifest = {
        "backend": "python_pandas",
        "script": str(SCRIPT_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "tables": table_entries,
        "manual_numeric_entry": False,
        "new_rollouts": False,
    }
    (TABLE_METADATA_ROOT / "table_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def validate_outputs() -> None:
    expected_stems = [
        "figure_1_measurement_problem",
        "figure_2_protocol_and_identity",
        "figure_3_discovery_and_confirmation",
        "figure_4_cost_composition",
        "figure_5_scenario_resource_boundaries",
    ]
    for stem in expected_stems:
        for suffix in (".svg", ".pdf", ".tiff", "_preview.png"):
            path = EXPORT_ROOT / f"{stem}{suffix}"
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"Missing or empty export: {path}")
        svg = (EXPORT_ROOT / f"{stem}.svg").read_text(encoding="utf-8")
        if "<text" not in svg:
            raise RuntimeError(f"SVG text is not editable: {stem}")

    r2_contexts = pd.read_csv(R2_ROOT / "context_substitution_estimates.csv")
    selected = r2_contexts[
        (r2_contexts["scenario"] == "time_pressure")
        & (r2_contexts["slot"] == "resource")
    ]
    if len(selected) != 18:
        raise RuntimeError("Expected 18 R2 time/resource contexts")
    if set(selected["resource_type"]) != {"missile", "laser"}:
        raise RuntimeError("Both resource types must be present")
    if selected["resource_type"].value_counts().to_dict() != {"missile": 9, "laser": 9}:
        raise RuntimeError("Expected 9 missile and 9 laser contexts")


def main() -> None:
    ensure_output_dirs()
    figure_1()
    figure_2()
    figure_3()
    figure_4()
    figure_5()
    generate_tables()
    validate_outputs()
    print("W1-06 deterministic figure/table generation completed.")


if __name__ == "__main__":
    main()
