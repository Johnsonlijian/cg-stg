"""Round 8.2 — Decadal acceleration analysis.

For each of the top-10 R7 cities × each SSP, fit a quadratic (and linear) trend
to the 2020-2100 climate-gap trajectory and report the acceleration coefficient.
This quantifies whether the per-decade gap INCREASES with time (positive
acceleration; consistent with non-linear soil-state response) or stays linear
(constant decadal gain).

Output:
    outputs/round8/decadal_acceleration.csv
    outputs/round8/decadal_acceleration.png
"""
from __future__ import annotations
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_ROOT.parents[2]
R7_DIR = PROJECT_ROOT / "ai_autoboost" / "outputs" / "round7"
OUT_DIR = PROJECT_ROOT / "ai_autoboost" / "outputs" / "round8"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("r8_accel")


def main() -> int:
    src = R7_DIR / "decadal_summary.csv"
    if not src.exists():
        log.error(f"Missing R7 decadal_summary at {src}")
        return 1

    df = pd.read_csv(src)
    log.info(f"Loaded {len(df)} decadal records")

    # Fit per (city, ssp) a quadratic: gap = a + b * t + c * t^2 where t = (epoch - 2020)/10
    rows = []
    for (city, ssp), sub in df.groupby(["city", "ssp"]):
        sub_sorted = sub.sort_values("epoch")
        if len(sub_sorted) < 4:
            continue
        t = (sub_sorted["epoch"].values - 2020) / 10.0
        g = sub_sorted["mean_gap"].values
        # Quadratic fit
        coeffs = np.polyfit(t, g, 2)  # [a2, a1, a0] in p(t) = a2*t^2 + a1*t + a0
        a2, a1, a0 = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
        # Linear fit for comparison
        lin = np.polyfit(t, g, 1)
        l1, l0 = float(lin[0]), float(lin[1])
        # R2 of quadratic vs linear
        ss_total = np.sum((g - g.mean()) ** 2)
        ss_res_q = np.sum((g - np.polyval(coeffs, t)) ** 2)
        ss_res_l = np.sum((g - np.polyval(lin, t)) ** 2)
        r2_q = 1 - ss_res_q / max(ss_total, 1e-10)
        r2_l = 1 - ss_res_l / max(ss_total, 1e-10)
        # Acceleration: 2 * a2 = d²(gap)/dt² in (per decade)²
        accel = 2 * a2
        # 2020→2050 decade-rate vs 2050→2100 decade-rate
        rate_early = (np.polyval(coeffs, 3) - np.polyval(coeffs, 0)) / 3.0
        rate_late = (np.polyval(coeffs, 8) - np.polyval(coeffs, 5)) / 3.0
        rate_ratio = rate_late / max(abs(rate_early), 1e-10) if abs(rate_early) > 1e-10 else float("nan")
        rows.append({
            "city": city, "ssp": ssp,
            "n_epochs": int(len(sub_sorted)),
            "linear_slope": round(l1, 6),
            "linear_intercept": round(l0, 6),
            "linear_R2": round(r2_l, 4),
            "quadratic_a2": round(a2, 6),
            "quadratic_a1": round(a1, 6),
            "quadratic_a0": round(a0, 6),
            "quadratic_R2": round(r2_q, 4),
            "acceleration": round(accel, 6),
            "decade_rate_2020_2050": round(rate_early, 5),
            "decade_rate_2050_2100": round(rate_late, 5),
            "rate_ratio_late_over_early": round(rate_ratio, 3),
            "quadratic_improves_R2_by": round(r2_q - r2_l, 4),
        })

    with (OUT_DIR / "decadal_acceleration.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # Aggregate per SSP
    agg = []
    for ssp, sub_ssp in pd.DataFrame(rows).groupby("ssp"):
        agg.append({
            "ssp": ssp,
            "n_cities": int(len(sub_ssp)),
            "mean_acceleration": round(float(sub_ssp["acceleration"].mean()), 6),
            "median_rate_ratio_late_over_early": round(float(sub_ssp["rate_ratio_late_over_early"].median()), 3),
            "mean_quadratic_R2": round(float(sub_ssp["quadratic_R2"].mean()), 4),
            "mean_linear_R2": round(float(sub_ssp["linear_R2"].mean()), 4),
        })
    with (OUT_DIR / "decadal_acceleration_agg.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
        w.writeheader(); w.writerows(agg)
    for r in agg:
        log.info(f"{r['ssp']}: mean_accel={r['mean_acceleration']}, "
                 f"median rate_late/rate_early = {r['median_rate_ratio_late_over_early']}, "
                 f"quad-vs-lin R² gain = {r['mean_quadratic_R2'] - r['mean_linear_R2']:.3f}")

    # Plot acceleration per city × SSP
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        df_r = pd.DataFrame(rows)
        cities = sorted(df_r["city"].unique())
        ssps = ["Control-NoCC", "SSP1-2.6", "SSP2-4.5", "SSP5-8.5"]
        x = np.arange(len(cities))
        width = 0.21
        ssp_colors = {"Control-NoCC": "#666", "SSP1-2.6": "#1f77b4",
                       "SSP2-4.5": "#ff7f0e", "SSP5-8.5": "#d62728"}
        fig, ax = plt.subplots(figsize=(13, 5))
        for i, ssp in enumerate(ssps):
            sub = df_r[df_r["ssp"] == ssp].set_index("city").reindex(cities)
            vals = sub["acceleration"].values
            ax.bar(x + (i - 1.5) * width, vals, width=width, color=ssp_colors[ssp], label=ssp)
        ax.set_xticks(x)
        ax.set_xticklabels(cities, rotation=20, ha="right")
        ax.set_ylabel("Quadratic acceleration (d²gap/dt²) per decade²")
        ax.set_title("Decadal acceleration of climate-induced damage gap (top-10 R7 cities × 4 SSPs)")
        ax.axhline(0, color="grey", lw=0.5)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "decadal_acceleration.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        log.warning(f"plot fail: {e}")

    log.info("R8 decadal acceleration done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
