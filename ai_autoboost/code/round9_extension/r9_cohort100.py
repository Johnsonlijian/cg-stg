"""Round 9.1 — 100-city cohort: add 50 new anchors to the R6 50-city cohort
and rerun cross-SSP regression at 72 samples/cell precision (matching R7).

Output:
    outputs/round9/cohort100_raw.csv
    outputs/round9/cohort100_summary.csv
    outputs/round9/cohort100_regression.csv
    outputs/round9/cohort100_regression.png
    outputs/round9/cohort100_top_positive.csv
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
OUT_DIR = PROJECT_ROOT / "ai_autoboost" / "outputs" / "round9"
LOG_DIR = PROJECT_ROOT / "ai_autoboost" / "logs"
for d in (OUT_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / f"r9_cohort_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.log",
                            encoding="utf-8"),
    ],
)
log = logging.getLogger("r9_cohort")

sys.path.insert(0, str(CODE_ROOT.parent / "round2_baselines_ablation"))
sys.path.insert(0, str(CODE_ROOT.parent / "round3_mechanism_error"))
sys.path.insert(0, str(CODE_ROOT.parent / "round4_generalization_final"))
sys.path.insert(0, str(CODE_ROOT.parent / "round6_extension"))
sys.path.insert(0, str(CODE_ROOT.parent / "round7_extension"))
import r2_lib as L
from r2_main import METHODS, run_one
from r3_osm_pipeline import fetch_osm_graph
from r4_cohort_anchors import CityAnchor
from r6_cohort50 import cohort50, sample_dGW_r6


# 50 additional anchor cities (R9 round), grouped roughly by region.
R9_ADDITIONS: List[CityAnchor] = [
    # North America
    CityAnchor("LosAngeles",   "USA",          34.0522, -118.2437, "mixed",     8.0, 330.0, "LA basin, mixed sediment"),
    CityAnchor("SanDiego",     "USA",          32.7157, -117.1611, "coastal",   6.0, 290.0, "Pacific coastal"),
    CityAnchor("Phoenix",      "USA",          33.4484, -112.0740, "arid",     30.0, 380.0, "Sonoran desert"),
    CityAnchor("Houston",      "USA",          29.7604,  -95.3698, "coastal",   2.5, 220.0, "Gulf coast deltaic margin"),
    CityAnchor("Miami",        "USA",          25.7617,  -80.1918, "coastal",   1.5, 200.0, "Limestone aquifer near surface"),
    CityAnchor("Boston",       "USA",          42.3601,  -71.0589, "mixed",     5.0, 280.0, "Glacial-marine"),
    CityAnchor("NewYork",      "USA",          40.7128,  -74.0060, "mixed",     6.0, 300.0, "Manhattan schist + Hudson fill"),
    CityAnchor("Toronto",      "Canada",       43.6532,  -79.3832, "mixed",     7.0, 310.0, "Lake Ontario margin"),
    CityAnchor("Montreal",     "Canada",       45.5017,  -73.5673, "cold",      6.0, 320.0, "Saint Lawrence valley"),
    CityAnchor("Chicago",      "USA",          41.8781,  -87.6298, "lowland",   4.0, 250.0, "Lake Michigan margin"),
    CityAnchor("Memphis",      "USA",          35.1495,  -90.0490, "lowland",   2.0, 220.0, "New Madrid Seismic Zone"),
    # Europe
    CityAnchor("London",       "UK",           51.5074,   -0.1278, "mixed",     8.0, 310.0, "Thames basin clay"),
    CityAnchor("Paris",        "France",       48.8566,    2.3522, "lowland",   5.0, 260.0, "Seine basin"),
    CityAnchor("Berlin",       "Germany",      52.5200,   13.4050, "lowland",   4.5, 250.0, "North European Plain"),
    CityAnchor("Madrid",       "Spain",        40.4168,   -3.7038, "mixed",    12.0, 330.0, "Iberian plateau"),
    CityAnchor("Rome",         "Italy",        41.9028,   12.4964, "mixed",    10.0, 320.0, "Tiber valley + volcanic"),
    CityAnchor("Bucharest",    "Romania",      44.4268,   26.1025, "mixed",     8.0, 310.0, "Vrancea seismic zone"),
    CityAnchor("Sofia",        "Bulgaria",     42.6977,   23.3219, "mixed",    11.0, 320.0, "Balkan piedmont"),
    CityAnchor("Belgrade",     "Serbia",       44.7866,   20.4489, "mixed",     6.0, 290.0, "Sava-Danube confluence"),
    CityAnchor("Stockholm",    "Sweden",       59.3293,   18.0686, "coastal",   3.0, 350.0, "Baltic Sea archipelago"),
    CityAnchor("Helsinki",     "Finland",      60.1699,   24.9384, "coastal",   2.0, 340.0, "Baltic granite"),
    CityAnchor("Oslo",         "Norway",       59.9139,   10.7522, "coastal",   4.0, 330.0, "Fjord head"),
    CityAnchor("Copenhagen",   "Denmark",      55.6761,   12.5683, "coastal",   2.5, 240.0, "Glacial moraine + Oresund"),
    CityAnchor("Amsterdam",    "Netherlands",  52.3676,    4.9041, "deltaic",   1.0, 200.0, "Rhine-Meuse delta below sea level"),
    CityAnchor("Rotterdam",    "Netherlands",  51.9244,    4.4777, "deltaic",   1.0, 200.0, "Rhine delta below sea level"),
    CityAnchor("Hamburg",      "Germany",      53.5511,    9.9937, "lowland",   3.0, 240.0, "Elbe river port"),
    CityAnchor("StPetersburg", "Russia",       59.9343,   30.3351, "coastal",   2.5, 240.0, "Neva delta"),
    # Middle East / Africa
    CityAnchor("Beirut",       "Lebanon",      33.8938,   35.5018, "coastal",   6.0, 290.0, "Eastern Mediterranean"),
    CityAnchor("TelAviv",      "Israel",       32.0853,   34.7818, "coastal",   5.0, 260.0, "Levantine coast"),
    CityAnchor("Dubai",        "UAE",          25.2048,   55.2708, "arid",      4.0, 270.0, "Arabian Gulf, brackish shallow GW"),
    CityAnchor("Riyadh",       "SaudiArabia",  24.7136,   46.6753, "arid",     35.0, 400.0, "Najd plateau"),
    CityAnchor("AddisAbaba",   "Ethiopia",      9.0250,   38.7469, "high_alt",  9.0, 310.0, "Ethiopian highlands 2300m"),
    CityAnchor("Nairobi",      "Kenya",        -1.2921,   36.8219, "mixed",     8.0, 310.0, "African highlands"),
    CityAnchor("Khartoum",     "Sudan",        15.5007,   32.5599, "arid",     10.0, 350.0, "Blue/White Nile confluence"),
    CityAnchor("Tunis",        "Tunisia",      36.8065,   10.1815, "coastal",   8.0, 320.0, "Mediterranean coast"),
    CityAnchor("Marrakech",    "Morocco",      31.6295,   -7.9811, "arid",     15.0, 350.0, "Atlas piedmont"),
    # Asia
    CityAnchor("Seoul",        "SouthKorea",   37.5665,  126.9780, "mixed",     7.0, 310.0, "Han River basin"),
    CityAnchor("Pyongyang",    "NorthKorea",   39.0392,  125.7625, "mixed",     8.0, 320.0, "Taedong River valley"),
    CityAnchor("Hanoi",        "Vietnam",      21.0285,  105.8542, "deltaic",   2.0, 210.0, "Red River delta"),
    CityAnchor("KualaLumpur",  "Malaysia",      3.1390,  101.6869, "mixed",     7.0, 280.0, "Klang valley"),
    CityAnchor("Singapore",    "Singapore",     1.3521,  103.8198, "coastal",   3.0, 260.0, "Tropical island, reclaimed land"),
    CityAnchor("Yangon",       "Myanmar",      16.8409,   96.1735, "deltaic",   2.0, 220.0, "Irrawaddy delta"),
    CityAnchor("Bishkek",      "Kyrgyzstan",   42.8746,   74.5698, "mixed",    10.0, 310.0, "Tian Shan piedmont"),
    CityAnchor("Ulaanbaatar",  "Mongolia",     47.8864,  106.9057, "cold",      9.0, 310.0, "Tuul River valley, permafrost margin"),
    # South America
    CityAnchor("Caracas",      "Venezuela",    10.4806,  -66.9036, "mixed",     8.0, 310.0, "Coastal range"),
    CityAnchor("Medellin",     "Colombia",      6.2476,  -75.5658, "high_alt",  7.0, 290.0, "Aburrá Valley 1495m"),
    CityAnchor("Rio",          "Brazil",      -22.9068,  -43.1729, "coastal",   3.5, 250.0, "Guanabara Bay"),
    CityAnchor("SaoPaulo",     "Brazil",      -23.5505,  -46.6333, "mixed",     8.0, 310.0, "Atlantic Plateau 760m"),
    CityAnchor("Brasilia",     "Brazil",      -15.8267,  -47.9218, "mixed",    11.0, 340.0, "Central highlands 1170m"),
    # Oceania
    CityAnchor("Honolulu",     "USA",          21.3069, -157.8583, "coastal",   3.0, 270.0, "Volcanic island"),
]


def cohort100() -> List[CityAnchor]:
    return list(cohort50()) + list(R9_ADDITIONS)


SSPS = ["Control-NoCC", "SSP1-2.6", "SSP2-4.5", "SSP5-8.5"]
EPOCHS = (2020, 2100)


def run_city_72(cg: L.CityGraph, n_seeds: int = 6, n_mc: int = 12,
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


def per_city_per_ssp_gap(rows: List[Dict], city: str, ssp: str) -> Dict:
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
    return {"city": city, "ssp": ssp, "n_pairs": int(diff.size),
            "mean_gap": float(diff.mean()),
            "ci_lo": float(lo), "ci_hi": float(hi),
            "sign": "positive" if lo > 0 else ("negative" if hi < 0 else "zero-crossing")}


def main() -> int:
    t0 = datetime.utcnow()
    cohort = cohort100()
    log.info(f"R9 cohort100 start: {len(cohort)} cities (R6 50 + R9 50)")

    all_rows: List[Dict] = []
    failed, successful = [], []
    for city in cohort:
        log.info(f"Loading {city.name} ({city.country}) ...")
        d = {"name": city.name, "lat": city.lat, "lon": city.lon,
             "gw_base": city.gw_base_m, "vs30_mu": city.vs30_mu,
             "archetype_match": city.archetype}
        cg = fetch_osm_graph(d, dist_m=1000, sample_n=100)
        if cg is None:
            failed.append(city.name); continue
        object.__setattr__(cg, "city_name", city.name)
        rows = run_city_72(cg, n_seeds=6, n_mc=12)
        all_rows.extend(rows)
        successful.append(city.name)
        log.info(f"  {city.name}: {len(rows)} records, n_nodes={cg.n_nodes}")

    log.info(f"Loaded {len(successful)}/{len(cohort)}; failed: {failed}")

    # Persist raw + summary
    with (OUT_DIR / "cohort100_raw.csv").open("w", newline="", encoding="utf-8") as f:
        if all_rows:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader(); w.writerows(all_rows)

    summary_rows = []
    for city in cohort:
        if city.name not in successful:
            continue
        for ssp in SSPS:
            s = per_city_per_ssp_gap(all_rows, city.name, ssp)
            s.update({"archetype": city.archetype, "country": city.country,
                       "gw_base_m": city.gw_base_m, "lat": city.lat, "lon": city.lon})
            summary_rows.append(s)
    cols = ["city", "country", "archetype", "lat", "lon", "gw_base_m",
            "ssp", "n_pairs", "mean_gap", "ci_lo", "ci_hi", "sign"]
    with (OUT_DIR / "cohort100_summary.csv").open("w", newline="", encoding="utf-8") as f:
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
            "pearson_r": round(r_p, 4), "pearson_p": round(p_p, 10),
            "linear_R2": round(r2, 4),
            "slope": round(float(m.coef_[0]), 5),
            "intercept": round(float(m.intercept_), 5),
            "n_strictly_positive_CI": int(sum(r["sign"] == "positive" for r in sub)),
            "n_strictly_negative_CI": int(sum(r["sign"] == "negative" for r in sub)),
            "n_zero_crossing": int(sum(r["sign"] == "zero-crossing" for r in sub)),
        })
        log.info(f"{ssp}: n={reg_rows[-1]['n_cities']}, r={reg_rows[-1]['pearson_r']}, p={reg_rows[-1]['pearson_p']:.4g}, R²={reg_rows[-1]['linear_R2']}, positive_CI={reg_rows[-1]['n_strictly_positive_CI']}")
    with (OUT_DIR / "cohort100_regression.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(reg_rows[0].keys()))
        w.writeheader(); w.writerows(reg_rows)

    # Top positive cities under SSP5-8.5
    top_pos = sorted(
        [r for r in summary_rows if r["ssp"] == "SSP5-8.5" and r["sign"] == "positive"],
        key=lambda r: r["mean_gap"], reverse=True)
    if top_pos:
        with (OUT_DIR / "cohort100_top_positive.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["city", "country", "archetype", "gw_base_m",
                                              "mean_gap", "ci_lo", "ci_hi"])
            w.writeheader()
            for r in top_pos:
                w.writerow({k: round(r[k], 5) if isinstance(r[k], float) else r[k]
                            for k in ["city", "country", "archetype", "gw_base_m",
                                      "mean_gap", "ci_lo", "ci_hi"]})
    log.info(f"SSP5-8.5 strictly-positive-CI cities: {len(top_pos)}/{len(successful)}")
    log.info(f"  Top: {[(r['city'], round(r['mean_gap'], 4)) for r in top_pos[:15]]}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 4, figsize=(22, 5), sharey=True)
        for i, ssp in enumerate(SSPS):
            sub = [r for r in summary_rows if r["ssp"] == ssp and not np.isnan(r["mean_gap"])]
            x = np.array([1.0 / r["gw_base_m"] for r in sub])
            y = np.array([r["mean_gap"] for r in sub])
            ax = axes[i]
            colors = ["#d62728" if r["sign"] == "positive" else
                       "#1f77b4" if r["sign"] == "negative" else "#999"
                       for r in sub]
            ax.scatter(x, y, s=24, c=colors, edgecolors="black", linewidths=0.4)
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
        fig.suptitle(f"R9 per-SSP cohort regression — n={len(successful)} real OSM cities, 72 samples/cell")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "cohort100_regression.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        log.warning(f"plot fail: {e}")

    meta = {
        "elapsed_seconds": round((datetime.utcnow() - t0).total_seconds(), 2),
        "n_cities_targeted": len(cohort),
        "n_successful": len(successful),
        "n_records": len(all_rows),
        "regression": {r["ssp"]: {"R2": r["linear_R2"], "p": r["pearson_p"],
                                    "n_positive_CI": r["n_strictly_positive_CI"]}
                        for r in reg_rows},
    }
    (OUT_DIR / "cohort100_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info(f"R9 cohort100 done in {meta['elapsed_seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
