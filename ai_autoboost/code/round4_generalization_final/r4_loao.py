"""Round 4.2 — Leave-one-archetype-out (and leave-one-city-out) regression stability.

For each held-out city / archetype, recompute the cohort regression of climate gap
on inverse-GW depth and report R² + Pearson r + p. If R² remains stable across
holdouts, the headline R² is not dominated by a single city.

Output:
    outputs/round4/loao_R2_stability.csv
    outputs/round4/loao_R2_stability.png
"""
from __future__ import annotations

import csv
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_ROOT.parents[2]
OUT_DIR = PROJECT_ROOT / "ai_autoboost" / "outputs" / "round4"
LOG_DIR = PROJECT_ROOT / "ai_autoboost" / "logs"
for d in (OUT_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / f"r4_loao_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.log",
                            encoding="utf-8"),
    ],
)
log = logging.getLogger("r4_loao")


def cohort_regression(x: np.ndarray, y: np.ndarray) -> Dict:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.linear_model import LinearRegression
    r_p, p_p = pearsonr(x, y)
    r_s, p_s = spearmanr(x, y)
    m = LinearRegression().fit(x.reshape(-1, 1), y)
    return {
        "n": int(x.size),
        "pearson_r": round(float(r_p), 4),
        "pearson_p": round(float(p_p), 6),
        "spearman_r": round(float(r_s), 4),
        "spearman_p": round(float(p_s), 6),
        "linear_R2": round(float(m.score(x.reshape(-1, 1), y)), 4),
        "slope": round(float(m.coef_[0]), 5),
        "intercept": round(float(m.intercept_), 5),
    }


def main() -> int:
    t0 = datetime.utcnow()
    summary_csv = OUT_DIR / "expanded_cohort_summary.csv"
    if not summary_csv.exists():
        log.error(f"Required input not found: {summary_csv}")
        return 1
    df = pd.read_csv(summary_csv)
    log.info(f"Loaded cohort: {df.shape[0]} cities")

    # Baseline: full cohort regression
    inv_gw = 1.0 / df["gw_base_m"].values
    gap = df["mean_gap"].values
    base = cohort_regression(inv_gw, gap)
    base["holdout_type"] = "FULL"
    base["holdout_id"] = "—"
    log.info(f"Full-cohort baseline: n={base['n']}  Pearson r={base['pearson_r']}  R²={base['linear_R2']}  p={base['pearson_p']}")

    rows: List[Dict] = [base]

    # Leave-one-city-out (LOCO)
    for i in range(df.shape[0]):
        keep = np.ones(df.shape[0], dtype=bool)
        keep[i] = False
        sub_x = inv_gw[keep]; sub_y = gap[keep]
        r = cohort_regression(sub_x, sub_y)
        r["holdout_type"] = "LOCO"
        r["holdout_id"] = df.iloc[i]["city"]
        r["holdout_archetype"] = df.iloc[i]["archetype"]
        r["holdout_gw_base_m"] = float(df.iloc[i]["gw_base_m"])
        r["holdout_gap"] = float(df.iloc[i]["mean_gap"])
        rows.append(r)

    # Leave-one-archetype-out (LOAO)
    for arch in sorted(df["archetype"].unique()):
        keep = df["archetype"].values != arch
        if keep.sum() < 3:
            continue
        sub_x = inv_gw[keep]; sub_y = gap[keep]
        r = cohort_regression(sub_x, sub_y)
        r["holdout_type"] = "LOAO"
        r["holdout_id"] = arch
        r["n_cities_held_out"] = int((~keep).sum())
        rows.append(r)

    # Persist
    fields = sorted({k for r in rows for k in r.keys()})
    fields = ["holdout_type", "holdout_id", "n"] + [f for f in fields if f not in ("holdout_type", "holdout_id", "n")]
    for r in rows:
        for f in fields:
            r.setdefault(f, "")
    with (OUT_DIR / "loao_R2_stability.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # Diagnostics
    loco = [r for r in rows if r["holdout_type"] == "LOCO"]
    loao = [r for r in rows if r["holdout_type"] == "LOAO"]
    loco_R2 = np.array([r["linear_R2"] for r in loco])
    loao_R2 = np.array([r["linear_R2"] for r in loao])
    log.info(f"LOCO R² across {loco_R2.size} city holdouts: mean={loco_R2.mean():.3f}  std={loco_R2.std(ddof=1):.3f}  min={loco_R2.min():.3f}  max={loco_R2.max():.3f}")
    log.info(f"LOAO R² across {loao_R2.size} archetype holdouts: {loao_R2.tolist()}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        # LOCO
        ax = axes[0]
        order = np.argsort([r["linear_R2"] for r in loco])
        labels = [loco[i]["holdout_id"] for i in order]
        vals = [loco[i]["linear_R2"] for i in order]
        ax.barh(range(len(loco)), vals, color="#1f77b4")
        ax.axvline(base["linear_R2"], color="red", lw=1.5, linestyle="--",
                   label=f"Full cohort R²={base['linear_R2']}")
        ax.set_yticks(range(len(loco)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("R² after holding out this city")
        ax.set_title(f"Leave-one-city-out (n={base['n']})")
        ax.legend()
        ax.grid(alpha=0.3, axis="x")
        # LOAO
        ax = axes[1]
        labels = [r["holdout_id"] for r in loao]
        vals = [r["linear_R2"] for r in loao]
        bars = ax.bar(range(len(loao)), vals, color="#d62728")
        ax.axhline(base["linear_R2"], color="black", lw=1.5, linestyle="--",
                   label=f"Full cohort R²={base['linear_R2']}")
        ax.set_xticks(range(len(loao)))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("R² after holding out archetype")
        ax.set_title(f"Leave-one-archetype-out")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.2f}",
                    ha="center", fontsize=8)
        fig.suptitle("Round 4 — cohort regression R² stability under holdout")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "loao_R2_stability.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        log.warning(f"plot fail: {e}")

    log.info(f"R4 LOAO done in {(datetime.utcnow() - t0).total_seconds():.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
