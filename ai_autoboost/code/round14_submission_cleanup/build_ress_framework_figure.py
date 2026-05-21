"""Build a RESS-style system reliability framework figure."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "round14_submission_cleanup"
OUT.mkdir(parents=True, exist_ok=True)


def box(ax, xy, w, h, text, fc, ec="#333333", size=10):
    patch = FancyBboxPatch(
        xy, w, h,
        boxstyle="round,pad=0.02,rounding_size=0.035",
        linewidth=1.1,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=size)
    return patch


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, linewidth=1.1, color="#333333"))


def main() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.965, "Climate-modulated seismic lifeline reliability screening workflow",
            ha="center", va="top", fontsize=16, weight="bold")

    # Layer bands
    ax.add_patch(FancyBboxPatch((0.035, 0.64), 0.93, 0.22, boxstyle="round,pad=0.015,rounding_size=0.02",
                                facecolor="#f1f5f9", edgecolor="#cbd5e1", linewidth=0.9))
    ax.text(0.055, 0.835, "Physical screening modules", fontsize=11, weight="bold", color="#334155")

    ax.add_patch(FancyBboxPatch((0.035, 0.34), 0.93, 0.22, boxstyle="round,pad=0.015,rounding_size=0.02",
                                facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=0.9))
    ax.text(0.055, 0.535, "Network reliability module", fontsize=11, weight="bold", color="#334155")

    ax.add_patch(FancyBboxPatch((0.035, 0.08), 0.93, 0.16, boxstyle="round,pad=0.015,rounding_size=0.02",
                                facecolor="#fff7ed", edgecolor="#fed7aa", linewidth=0.9))
    ax.text(0.055, 0.205, "Validation, uncertainty and surrogate layer", fontsize=11, weight="bold", color="#7c2d12")

    # Physical layer boxes
    b1 = box(ax, (0.08, 0.69), 0.16, 0.095, "Scenario forcing\nsoil-state perturbation", "#dbeafe")
    b2 = box(ax, (0.29, 0.69), 0.16, 0.095, "Ground motion\nBSSA14 PGA", "#dcfce7")
    b3 = box(ax, (0.50, 0.69), 0.16, 0.095, "Liquefaction\ntriggering", "#ede9fe")
    b4 = box(ax, (0.71, 0.69), 0.16, 0.095, "Fragility\nensemble", "#fae8ff")
    for start, end in [((0.24, 0.738), (0.29, 0.738)), ((0.45, 0.738), (0.50, 0.738)), ((0.66, 0.738), (0.71, 0.738))]:
        arrow(ax, start, end)

    # Network layer
    b5 = box(ax, (0.18, 0.39), 0.19, 0.095, "Lifeline dependency\ngraph", "#e0f2fe")
    b6 = box(ax, (0.43, 0.39), 0.19, 0.095, "Cascade propagation\nbounded damage", "#ccfbf1")
    b7 = box(ax, (0.68, 0.39), 0.19, 0.095, "Reliability-loss\nmetrics", "#fef3c7")
    arrow(ax, (0.79, 0.69), (0.76, 0.485))
    arrow(ax, (0.37, 0.438), (0.43, 0.438))
    arrow(ax, (0.62, 0.438), (0.68, 0.438))

    # Validation/surrogate layer
    b8 = box(ax, (0.09, 0.115), 0.18, 0.07, "BCa intervals\nrobustness sweeps", "#ffedd5", size=9.5)
    b9 = box(ax, (0.32, 0.115), 0.18, 0.07, "Historical\nenvelope checks", "#ffedd5", size=9.5)
    b10 = box(ax, (0.55, 0.115), 0.18, 0.07, "GraphSAGE\nsurrogate", "#ffedd5", size=9.5)
    b11 = box(ax, (0.78, 0.115), 0.13, 0.07, "Supplementary\ndiagnostics", "#ffedd5", size=9.2)
    arrow(ax, (0.765, 0.39), (0.64, 0.185))
    arrow(ax, (0.76, 0.39), (0.84, 0.185))

    ax.text(0.5, 0.035,
            "Interpretation boundary: scenario-conditioned screening evidence, not city-specific loss or outage prediction.",
            ha="center", va="bottom", fontsize=10.5, color="#334155")

    fig.tight_layout()
    out = OUT / "Fig_R14_RESS_framework.png"
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
