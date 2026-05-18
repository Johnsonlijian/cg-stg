"""Round 7.1 — High-precision rerun of 50-city × 4-SSP cohort.

Rationale: R6 used n_seeds=4 × n_mc=8 = 32 samples per cell to fit 50 cities × 4 SSPs
× 3 epochs in 7 minutes. The trade-off was wider per-city CIs (0/50 strictly positive
under SSP5-8.5 vs R5's 8/30). R7 reruns with R5 precision (6 × 12 = 72 samples per
cell) but drops the 2050 epoch (R7.2 decadal handles trajectory) to stay within
~ 15 min budget.

Plan: 50 cities × 4 SSPs × 2 epochs (2020, 2100) × 6 seeds × 12 MC × 2 methods
    = 57,600 records, ~ 12 min compute + ~ 3 min OSM cache.

Output:
    outputs/round7/cohort50_hires_raw.csv
    outputs/round7/cohort50_hires_summary.csv
    outputs/round7/cohort50_hires_regression.csv
    outputs/round7/cohort50_hires_regression.png
    outputs/round7/cohort50_hires_top_positive_cities.csv
"""
from __future__ import annotations
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

CODE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_ROOT.parents[2]
OUT_DIR = PROJECT_ROOT / "ai_autoboost" / "outputs" / "round7"
LOG_DIR = PROJECT_ROOT / "ai_autoboost" / "logs"
for d in (OUT_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / f"r7_hires_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.log",
                            encoding="utf-8"),
    ],
)
log = logging.getLogger("r7_hires")

sys.path.insert(0, str(CODE_ROOT.parent / "round2_baselines_ablation"))
sys.path.insert(0, str(CODE_ROOT.parent / "round3_mechanism_error"))
sys.path.insert(0, str(CODE_ROOT.parent / "round6_extension"))
import r2_lib as L
from r2_main import METHODS, run_one
from r3_osm_pipeline import fetch_osm_graph
from r6_cohort50 import cohort50, sample_dGW_r6


SSPS = ["Control-NoCC", "SSP1-2.6", "SSP2-4.5", "SSP5-8.5"]
EPOCHS = (2020, 2100)


def run_city_hires(cg: L.CityGraph, n_seeds: int = 6, n_mc: int = 12,
                    Mw: float = 6.5, R_km: float = 25.0) -> List[Dict]:
    rows = []
    for seed in range(n_seeds):
        for ssp in SSPS:
            for epoch in EPOCHS:
                ssp_rng = np.random.default_rng(abs(seed * 100_000 + hash(ssp) % 9973 + epoch))
                for mc in range(n_mc):
                    dGW = sample_dGW_r6(ssp_rng, ssp, epoch, cg.archetype) if cg.archetype != "unknown" else 0.0
                    for method_name in ("B0_static_hazus", "B4_cgstg_full"):
                        cfg = METHODS[method_name]
                        _, d_final, _, _, _, _ = run_one(cg, Mw, R_km, dGW, cfg, ssp_rng)
                        rows.append({
                            "city": getattr(cg, "city_name", "?"),
                            "archetype": cg.archetype,
                            "seed": seed, "ssp": ssp, "epoch": epoch, "mc": mc,
                            "dGW": float(dGW),
                            "method": method_name,
                            "mean_dmg_final": float(d_final),
                        })
    return rows


def per_city_per_ssp_gap_hires(rows: List[Dict], city: str, ssp: str) -> Dict:
    sub = [r for r in rows if r["city"] == city and r["method"] == "B4_cgstg_full" and r["ssp"] == ssp]
    g_2020 = [r["mean_dmg_final"] for r in sub if r["epoch"] == 2020]
    g_2100 = [r["mean_dmg_final"] for r in sub if r["epoch"] == 2100]
    if not g_2020 or not g_2100:
        return {"city": city, "ssp": ssp, "n_pairs": 0,
                "mean_gap": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    a = np.array(g_2100); b = np.array(g_2020)
    diff = a - b
    lo, hi = (L.bca_ci(diff, n_resamples=2999, rng_seed=hash((city, ssp)) & 0xFFFF)
              if diff.size >= 5 else (float("nan"), float("nan")))
    return {
        "city": city, "ssp": ssp, "n_pairs": int(diff.size),
        "mean_gap": float(diff.mean()),
        "std_gap": float(diff.std(ddof=1)) if diff.size > 1 else 0.0,
        "ci_lo": float(lo), "ci_hi": float(hi),
        "sign": "positive" if lo > 0 else ("negative" if hi < 0 else "zero-crossing"),
    }


def main() -> int:
    t0 = datetime.utcnow()
    cohort = cohort50()
    log.info(f"R7 hires start: {len(cohort)} cities; per-cell samples = 72 (6 seeds × 12 MC)")

    all_rows: List[Dict] = []
    failed, successful = [], []
    for city in cohort:
        log.info(f"Loading {city.name} ...")
        d = {"name": city.name, "lat": city.lat, "lon": city.lon,
             "gw_base": city.gw_base_m, "vs30_mu": city.vs30_mu,
             "archetype_match": city.archetype}
        cg = fetch_osm_graph(d, dist_m=1000, sample_n=100)
        if cg is None:
            failed.append(city.name); continue
        object.__setattr__(cg, "city_name", city.name)
        rows = run_city_hires(cg, n_seeds=6, n_mc=12)
        all_rows.extend(rows)
        successful.append(city.name)

    log.info(f"loaded {len(successful)}/{len(cohort)}")

    with (OUT_DIR / "cohort50_hires_raw.csv").open("w", newline="", encoding="utf-8") as f:
        if all_rows:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader(); w.writerows(all_rows)

    # Summary per city × SSP
    summary_rows = []
    for city in cohort:
        if city.name not in successful:
            continue
        for ssp in SSPS:
            s = per_city_per_ssp_gap_hires(all_rows, city.name, ssp)
            s.update({"archetype": city.archetype, "country": city.country,
                       "gw_base_m": city.gw_base_m, "lat": city.lat, "lon": city.lon})
            summary_rows.append(s)
    cols = ["city", "country", "archetype", "lat", "lon", "gw_base_m",
            "ssp", "n_pairs", "mean_gap", "std_gap", "ci_lo", "ci_hi", "sign"]
    with (OUT_DIR / "cohort50_hires_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in summary_rows:
            r = {k: round(v, 5) if isinstance(v, float) else v for k, v in r.items()}
            w.writerow({k: r.get(k, "") for k in cols})

    # Per-SSP cohort regression
    from scipy.stats import pearsonr
    from sklearn.linear_model import LinearRegression
    reg_rows = []
    for ssp in SSPS:
        sub = [r for r in summary_rows if r["ssp"] == ssp and not np.isnan(r["mean_gap"])]
        x = np.array([1.0 / r["gw_base_m"] for r in sub])
        y = np.array([r["mean_gap"] for r in sub])
        r_p, p_p = pearsonr(x, y)
        m = LinearRegression().fit(x.reshape(-1, 1), y)
        r2 = float(m.score(x.reshape(-1, 1), y))
        reg_rows.append({
            "ssp": ssp, "n_cities": len(sub),
            "pearson_r": round(r_p, 4), "pearson_p": round(p_p, 6),
            "linear_R2": round(r2, 4),
            "slope": round(float(m.coef_[0]), 5),
            "intercept": round(float(m.intercept_), 5),
            "n_strictly_positive_CI": int(sum(r["sign"] == "positive" for r in sub)),
            "n_strictly_negative_CI": int(sum(r["sign"] == "negative" for r in sub)),
            "n_zero_crossing": int(sum(r["sign"] == "zero-crossing" for r in sub)),
        })
        log.info(f"{ssp}: n={len(sub)}, r={reg_rows[-1]['pearson_r']}, p={reg_rows[-1]['pearson_p']:.4g}, R²={reg_rows[-1]['linear_R2']}, positive_CI={reg_rows[-1]['n_strictly_positive_CI']}/{len(sub)}")
    with (OUT_DIR / "cohort50_hires_regression.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(reg_rows[0].keys()))
        w.writeheader(); w.writerows(reg_rows)

    # Top-positive cities (CI strictly above zero) under SSP5-8.5
    top_pos = sorted(
        [r for r in summary_rows if r["ssp"] == "SSP5-8.5" and r["sign"] == "positive"],
        key=lambda r: r["mean_gap"], reverse=True)
    with (OUT_DIR / "cohort50_hires_top_positive_cities.csv").open("w", newline="", encoding="utf-8") as f:
        if top_pos:
            cols2 = ["city", "country", "archetype", "gw_base_m", "mean_gap", "ci_lo", "ci_hi"]
            w = csv.DictWriter(f, fieldnames=cols2)
            w.writeheader()
            for r in top_pos:
                w.writerow({k: round(r[k], 5) if isinstance(r[k], float) else r[k] for k in cols2 if k in r})
    log.info(f"SSP5-8.5 strictly-positive-CI cities: {len(top_pos)}/{len(successful)}")
    if top_pos:
        log.info(f"  Top: {[(r['city'], round(r['mean_gap'], 4)) for r in top_pos[:10]]}")

    # Plot per-SSP regression with hi-res
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
        for i, ssp in enumerate(SSPS):
            sub = [r for r in summary_rows if r["ssp"] == ssp and not np.isnan(r["mean_gap"])]
            x = np.array([1.0 / r["gw_base_m"] for r in sub])
            y = np.array([r["mean_gap"] for r in sub])
            ax = axes[i]
            colors = ["#d62728" if r["sign"] == "positive" else
                       "#1f77b4" if r["sign"] == "negative" else "#999"
                       for r in sub]
            ax.scatter(x, y, s=40, c=colors, edgecolors="black")
            if len(x) >= 5:
                m = LinearRegression().fit(x.reshape(-1, 1), y)
                xs = np.linspace(x.min(), x.max(), 50)
                ax.plot(xs, m.predict(xs.reshape(-1, 1)), "k--", lw=1)
                r_p, p_p = pearsonr(x, y)
                r2_local = float(m.score(x.reshape(-1, 1), y))
                pos_count = sum(r["sign"] == "positive" for r in sub)
                ax.set_title(f"{ssp}\nR² = {r2_local:.3f}, p = {p_p:.2e}\npos-CI = {pos_count}/{len(sub)}")
            ax.axhline(0, color="grey", lw=0.5)
            ax.set_xlabel("1 / baseline GW depth (m⁻¹)")
            if i == 0:
                ax.set_ylabel("Mean Δ damage 2100−2020")
            ax.grid(alpha=0.3)
        fig.suptitle("R7 hi-res per-SSP cohort regression — n=50 real OSM cities, 72 samples/cell")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "cohort50_hires_per_ssp.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        log.warning(f"plot fail: {e}")

    meta = {
        "elapsed_seconds": round((datetime.utcnow() - t0).total_seconds(), 2),
        "n_cities": len(cohort),
        "n_successful": len(successful),
        "samples_per_cell": 72,
        "n_records": len(all_rows),
        "regression": {r["ssp"]: {"R2": r["linear_R2"], "p": r["pearson_p"],
                                    "positive_CI_count": r["n_strictly_positive_CI"]}
                        for r in reg_rows},
    }
    (OUT_DIR / "cohort50_hires_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info(f"R7 hires done in {meta['elapsed_seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
