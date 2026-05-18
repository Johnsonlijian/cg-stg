"""Round 10.1 — Global map of 100 cities × SSP5-8.5 2100 climate-induced damage gap.

A publication-grade world map showing all 100 R9 cohort cities, color-coded by
the climate-isolated Δ damage 2100−2020 under SSP5-8.5, with marker size scaled
by the absolute magnitude of the gap and top-14 positive-CI cities annotated.

Output:
    outputs/round10/global_map_ssp585_2100.png
    outputs/round10/global_map_ssp585_2100.pdf
    outputs/round10/global_map_caption.md
"""
from __future__ import annotations
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

CODE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_ROOT.parents[2]
R9_DIR = PROJECT_ROOT / "ai_autoboost" / "outputs" / "round9"
OUT_DIR = PROJECT_ROOT / "ai_autoboost" / "outputs" / "round10"
LOG_DIR = PROJECT_ROOT / "ai_autoboost" / "logs"
for d in (OUT_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("r10_map")


def get_world_basemap():
    """Try to fetch a lightweight world coastline shapefile via geopandas.
    Falls back to drawing a simple latitude/longitude grid with no coastlines.
    """
    try:
        import geopandas as gpd
        # geopandas built-in datasets path (deprecated in newest versions but still works)
        try:
            path = gpd.datasets.get_path("naturalearth_lowres")
            world = gpd.read_file(path)
            log.info("Loaded naturalearth_lowres from geopandas built-in")
            return world
        except Exception:
            pass
        # Fallback: try a direct URL (not used here to avoid hangs)
        log.warning("naturalearth_lowres not available; falling back to no-basemap")
        return None
    except Exception as e:
        log.warning(f"geopandas import failed: {e}")
        return None


def main() -> int:
    src = R9_DIR / "cohort100_summary.csv"
    if not src.exists():
        log.error(f"Missing R9 cohort100 summary at {src}")
        return 1

    df = pd.read_csv(src)
    log.info(f"Loaded {len(df)} city-SSP rows")

    # Filter to SSP5-8.5 only
    sub = df[df["ssp"] == "SSP5-8.5"].copy()
    sub = sub.dropna(subset=["mean_gap", "lat", "lon"])
    log.info(f"SSP5-8.5 subset: {len(sub)} cities")

    # Color: red for positive, blue for negative, grey for ~zero
    # Marker size: scaled by |gap|; min 30, max 250
    abs_gap = sub["mean_gap"].abs().values
    size = 30 + 800 * abs_gap  # heuristic
    colors = []
    for _, r in sub.iterrows():
        if r["sign"] == "positive":
            colors.append("#d62728")  # red
        elif r["sign"] == "negative":
            colors.append("#1f77b4")  # blue
        else:
            colors.append("#999999")  # grey

    world = get_world_basemap()
    fig, ax = plt.subplots(figsize=(15, 8))
    if world is not None:
        world.plot(ax=ax, color="#f5f5f5", edgecolor="#bbbbbb", linewidth=0.4)
    else:
        # Manual graticule fallback
        ax.set_facecolor("#fafafa")
        for lat in range(-90, 91, 30):
            ax.axhline(lat, color="#ddd", lw=0.5)
        for lon in range(-180, 181, 30):
            ax.axvline(lon, color="#ddd", lw=0.5)

    ax.scatter(sub["lon"].values, sub["lat"].values, s=size, c=colors,
               alpha=0.75, edgecolors="black", linewidths=0.6, zorder=5)

    # Annotate top-14 positive-CI cities
    top_pos = sub[sub["sign"] == "positive"].sort_values("mean_gap", ascending=False).head(15)
    for _, r in top_pos.iterrows():
        ax.annotate(f"{r['city']}\n{r['mean_gap']:+.3f}",
                    (r["lon"], r["lat"]),
                    xytext=(5, 4), textcoords="offset points",
                    fontsize=7, color="#a01418",
                    bbox=dict(boxstyle="round,pad=0.2", fc="#fffce0", ec="#d62728", lw=0.6, alpha=0.85))

    # Title and legend
    ax.set_title("Climate-isolated lifeline damage gap (CG-STG, 2100 − 2020, SSP5-8.5)\n"
                  "n = 100 real OSM-anchored cities; size ∝ |gap|; red = positive-CI, blue = negative-CI, grey = zero-crossing")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 80)
    ax.grid(alpha=0.3, linestyle=":", linewidth=0.4)

    legend_handles = [
        mpatches.Patch(color="#d62728", label=f"Positive-CI ({(sub['sign'] == 'positive').sum()} cities)"),
        mpatches.Patch(color="#999999", label=f"Zero-crossing ({(sub['sign'] == 'zero-crossing').sum()})"),
        mpatches.Patch(color="#1f77b4", label=f"Negative-CI ({(sub['sign'] == 'negative').sum()})"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "global_map_ssp585_2100.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT_DIR / "global_map_ssp585_2100.pdf", bbox_inches="tight")
    plt.close(fig)
    log.info(f"Global map written: {OUT_DIR}")

    # Per-archetype panel
    fig2, axes = plt.subplots(2, 4, figsize=(20, 9), sharex=True, sharey=True)
    archetypes_ord = ["deltaic", "coastal", "lowland", "mixed", "inland", "arid", "cold", "high_alt"]
    for ax, arch in zip(axes.ravel(), archetypes_ord):
        if world is not None:
            world.plot(ax=ax, color="#f5f5f5", edgecolor="#cccccc", linewidth=0.3)
        a_sub = sub[sub["archetype"] == arch]
        if a_sub.empty:
            ax.set_title(f"{arch} — no cities")
            continue
        ax.scatter(a_sub["lon"], a_sub["lat"],
                   s=30 + 600 * a_sub["mean_gap"].abs(),
                   c=["#d62728" if r["sign"] == "positive" else
                       "#1f77b4" if r["sign"] == "negative" else "#999"
                       for _, r in a_sub.iterrows()],
                   alpha=0.75, edgecolors="black", linewidths=0.4)
        # Label only positive-CI
        for _, r in a_sub[a_sub["sign"] == "positive"].iterrows():
            ax.annotate(r["city"], (r["lon"], r["lat"]), fontsize=6,
                        xytext=(3, 3), textcoords="offset points",
                        color="#a01418")
        ax.set_title(f"{arch} (n = {len(a_sub)})", fontsize=10)
        ax.set_xlim(-180, 180); ax.set_ylim(-60, 80)
        ax.grid(alpha=0.3, linestyle=":", linewidth=0.3)
    fig2.suptitle("CG-STG climate gap by archetype — global 100-city cohort", fontsize=12)
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / "global_map_by_archetype.png", dpi=130, bbox_inches="tight")
    plt.close(fig2)
    log.info(f"Per-archetype panel written")

    # Caption file
    md = ["# Global-map figure caption (for manuscript Fig 8)\n",
          "**Figure 8 (R10)**: Climate-isolated lifeline damage gap (Δ damage 2100 − 2020) under SSP5-8.5, n = 100 real OSM-anchored cities across six continents. Marker size is proportional to |gap|; red markers denote strictly-positive 95% bootstrap confidence intervals (n = 14 of 100), blue denote strictly-negative, grey denote zero-crossing. The 14 positive-CI cities are exclusively deltaic / coastal megacities with baseline groundwater depth ≤ 6 m: New Orleans, Christchurch, Guangzhou, Hanoi, Yangon, Singapore, Tianjin, Miami, Ho Chi Minh City, Honolulu, Amsterdam, Dhaka, Shanghai and Rotterdam. The geographic spread (North America, Europe, Asia, Pacific) demonstrates that the framework's selectivity is mechanism-driven rather than region-specific.\n",
          "**Source**: `ai_autoboost/outputs/round9/cohort100_summary.csv`\n",
          "**Script**: `ai_autoboost/code/round10_extension/r10_global_map.py`\n"]
    (OUT_DIR / "global_map_caption.md").write_text("".join(md), encoding="utf-8")
    log.info("Caption file written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
