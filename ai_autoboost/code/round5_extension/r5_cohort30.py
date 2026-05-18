"""Round 5.1 — Cohort expansion 18 → 30 real OSM cities.

Adds 12 new anchor cities spanning Pacific NW, South Asia, Central Asia, high-altitude
Andean piedmont, and cold-zone Atlantic. Reuses R3/R4 OSM + CG-STG pipeline.

Output:
    outputs/round5/cohort30_raw.csv
    outputs/round5/cohort30_summary.csv
    outputs/round5/cohort30_regression.csv
    outputs/round5/cohort30_regression.png
    outputs/round5/cohort30_climate_gap.png
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

CODE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_ROOT.parents[2]
OUT_DIR = PROJECT_ROOT / "ai_autoboost" / "outputs" / "round5"
LOG_DIR = PROJECT_ROOT / "ai_autoboost" / "logs"
for d in (OUT_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / f"r5_cohort_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.log",
                            encoding="utf-8"),
    ],
)
log = logging.getLogger("r5_cohort")

sys.path.insert(0, str(CODE_ROOT.parent / "round2_baselines_ablation"))
sys.path.insert(0, str(CODE_ROOT.parent / "round3_mechanism_error"))
sys.path.insert(0, str(CODE_ROOT.parent / "round4_generalization_final"))
import r2_lib as L
from r2_main import METHODS, sample_dGW, run_one
from r3_osm_pipeline import fetch_osm_graph
from r4_cohort_anchors import COHORT_R4, CityAnchor
from r4_expanded_cohort import run_city, per_city_climate_gap


# 12 additional cities to make 30-city cohort.
ADDITIONS = [
    CityAnchor("Vancouver",     "Canada",        49.2827, -123.1207, "coastal",   5.0, 270.0, "Fraser delta margin"),
    CityAnchor("Seattle",       "USA",           47.6062, -122.3321, "mixed",     8.0, 320.0, "Puget Sound mixed"),
    CityAnchor("Portland",      "USA",           45.5152, -122.6784, "mixed",     7.0, 310.0, "Willamette river plain"),
    CityAnchor("Tehran",        "Iran",          35.6892,   51.3890, "arid",     15.0, 360.0, "Alborz piedmont, arid"),
    CityAnchor("Karachi",       "Pakistan",      24.8607,   67.0011, "coastal",   4.5, 240.0, "Indus delta margin"),
    CityAnchor("Dhaka",         "Bangladesh",    23.8103,   90.4125, "deltaic",   1.5, 210.0, "Bengal delta, extreme subsidence"),
    CityAnchor("Shanghai",      "China",         31.2304,  121.4737, "deltaic",   2.5, 230.0, "Yangtze delta"),
    CityAnchor("HoChiMinh",     "Vietnam",       10.8231,  106.6297, "deltaic",   2.0, 220.0, "Mekong delta margin"),
    CityAnchor("Tashkent",      "Uzbekistan",    41.2995,   69.2401, "arid",     12.0, 340.0, "Central Asia arid"),
    CityAnchor("Quito",         "Ecuador",       -0.1807,  -78.4678, "high_alt",  6.0, 280.0, "Andes 2850m"),
    CityAnchor("LaPaz",         "Bolivia",      -16.5000,  -68.1500, "high_alt", 10.0, 320.0, "Altiplano 3600m"),
    CityAnchor("Reykjavik",     "Iceland",       64.1466,  -21.9426, "cold",      5.0, 350.0, "North Atlantic basalt"),
]


def cohort30() -> List[CityAnchor]:
    return list(COHORT_R4) + list(ADDITIONS)


def main() -> int:
    t0 = datetime.utcnow()
    cohort = cohort30()
    log.info(f"R5 cohort30 start: {len(cohort)} cities (= R4 18 + R5 {len(ADDITIONS)} additions)")
    log.info(f"Archetypes: {sorted(set(c.archetype for c in cohort))}")

    all_rows: List[Dict] = []
    successful: List[str] = []
    failed: List[str] = []

    for city in cohort:
        log.info(f"Loading {city.name} ({city.country}) ...")
        d = {"name": city.name, "lat": city.lat, "lon": city.lon,
             "gw_base": city.gw_base_m, "vs30_mu": city.vs30_mu,
             "archetype_match": city.archetype}
        cg = fetch_osm_graph(d, dist_m=1000, sample_n=120)
        if cg is None:
            failed.append(city.name)
            log.warning(f"  SKIP {city.name}")
            continue
        object.__setattr__(cg, "city_name", city.name)
        object.__setattr__(cg, "country", city.country)
        object.__setattr__(cg, "lat", city.lat)
        object.__setattr__(cg, "lon", city.lon)

        rows = run_city(cg, n_seeds=6, n_mc=12)
        all_rows.extend(rows)
        successful.append(city.name)
        log.info(f"  {city.name}: {len(rows)} records, n_nodes={cg.n_nodes}")

    log.info(f"Loaded {len(successful)}/{len(cohort)}; failed: {failed}")

    # Persist raw
    with (OUT_DIR / "cohort30_raw.csv").open("w", newline="", encoding="utf-8") as f:
        if all_rows:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)

    # Per-city summary
    summary_rows = []
    for city in cohort:
        if city.name not in successful:
            continue
        s = per_city_climate_gap(all_rows, city.name)
        s.update({
            "archetype": city.archetype, "country": city.country,
            "gw_base_m": city.gw_base_m, "vs30_mu": city.vs30_mu,
            "lat": city.lat, "lon": city.lon,
        })
        summary_rows.append(s)
    summary_rows.sort(key=lambda r: r["mean_gap"], reverse=True)
    if summary_rows:
        cols = ["city", "country", "archetype", "lat", "lon", "gw_base_m", "vs30_mu",
                "n_pairs", "mean_gap", "ci_lo", "ci_hi", "sign"]
        with (OUT_DIR / "cohort30_summary.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in summary_rows:
                r = {k: round(v, 5) if isinstance(v, float) else v for k, v in r.items()}
                w.writerow({k: r.get(k, "") for k in cols})

    # Cohort regression
    if summary_rows:
        from scipy.stats import pearsonr, spearmanr
        from sklearn.linear_model import LinearRegression
        x = np.array([1.0 / r["gw_base_m"] for r in summary_rows])
        y = np.array([r["mean_gap"] for r in summary_rows])
        r_p, p_p = pearsonr(x, y)
        r_s, p_s = spearmanr(x, y)
        m = LinearRegression().fit(x.reshape(-1, 1), y)
        r2 = float(m.score(x.reshape(-1, 1), y))
        reg = {
            "n_cities": len(summary_rows),
            "pearson_r": round(r_p, 4),
            "pearson_p": round(p_p, 6),
            "spearman_r": round(r_s, 4),
            "spearman_p": round(p_s, 6),
            "linear_R2": round(r2, 4),
            "slope": round(float(m.coef_[0]), 5),
            "intercept": round(float(m.intercept_), 5),
        }
        with (OUT_DIR / "cohort30_regression.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(reg.keys()))
            w.writeheader()
            w.writerow(reg)
        log.info(f"COHORT REGRESSION n={reg['n_cities']}, Pearson r={reg['pearson_r']}, p={reg['pearson_p']}, R²={reg['linear_R2']}")

        # Plots
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # Scatter + OLS
            fig, ax = plt.subplots(figsize=(8, 5.5))
            ax.scatter(x, y, s=70, c="#d62728", edgecolors="black")
            for r in summary_rows:
                ax.annotate(r["city"], (1.0 / r["gw_base_m"], r["mean_gap"]),
                            fontsize=6.5, alpha=0.7, xytext=(4, 0), textcoords="offset points")
            xs = np.linspace(x.min(), x.max(), 100)
            ys = m.predict(xs.reshape(-1, 1))
            ax.plot(xs, ys, "k--", lw=1,
                    label=f"OLS y = {reg['slope']:.4f} x {'+' if reg['intercept'] >= 0 else ''}{reg['intercept']:.5f}\nR² = {reg['linear_R2']:.3f}, Pearson p = {reg['pearson_p']:.4f}")
            ax.axhline(0, color="grey", lw=0.5)
            ax.set_xlabel("1 / baseline groundwater depth (m⁻¹)")
            ax.set_ylabel("Mean climate-isolated Δ damage (2100−2020, SSP5-8.5)")
            ax.set_title(f"Round 5: cohort regression on inv-GW depth (n={len(summary_rows)} real OSM cities)")
            ax.legend(loc="lower right", fontsize=9)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(OUT_DIR / "cohort30_regression.png", dpi=130, bbox_inches="tight")
            plt.close(fig)

            # Bar of per-city
            fig, ax = plt.subplots(figsize=(15, 6))
            sorted_rows = sorted(summary_rows, key=lambda r: r["mean_gap"], reverse=True)
            xb = np.arange(len(sorted_rows))
            means = [r["mean_gap"] for r in sorted_rows]
            yerr = np.array([[m_ - r["ci_lo"], r["ci_hi"] - m_] for m_, r in zip(means, sorted_rows)]).T
            colors = ["#d62728" if r["sign"] == "positive" else
                       "#1f77b4" if r["sign"] == "negative" else "#aaaaaa"
                       for r in sorted_rows]
            ax.bar(xb, means, yerr=yerr, capsize=3, color=colors)
            ax.set_xticks(xb)
            labels = [f"{r['city']}\n({r['archetype']}, {r['gw_base_m']:.1f}m)" for r in sorted_rows]
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
            ax.set_ylabel("Climate-isolated Δ damage (2100−2020, SSP5-8.5)")
            ax.set_title(f"Round 5: per-city climate gap, n={len(sorted_rows)} real OSM cities (red=positive CI, blue=negative, grey=zero-crossing)")
            ax.axhline(0, color="grey", lw=0.5)
            ax.grid(alpha=0.3, axis="y")
            fig.tight_layout()
            fig.savefig(OUT_DIR / "cohort30_climate_gap.png", dpi=130, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            log.warning(f"plot fail: {e}")

    meta = {
        "started_utc": t0.isoformat() + "Z",
        "elapsed_seconds": round((datetime.utcnow() - t0).total_seconds(), 2),
        "n_cities": len(cohort),
        "n_successful": len(successful),
        "failed": failed,
        "n_records": len(all_rows),
    }
    (OUT_DIR / "cohort30_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                                  encoding="utf-8")
    log.info(f"R5 cohort done in {meta['elapsed_seconds']:.1f}s")
    print(f"\n=== R5 cohort expanded to {len(successful)}/{len(cohort)} cities, {len(all_rows)} records ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
