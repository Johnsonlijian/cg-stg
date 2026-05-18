"""Round 6 — 50-city cohort + 4-SSP cross-scenario scaling + policy ranking.

Adds 20 more anchor cities to R5's 30-city cohort (total 50), and extends the
climate-driver sampler to include SSP1-2.6 (low-emission baseline) for
cross-scenario damage-gap scaling analysis.

Output:
    outputs/round6/cohort50_raw.csv          (~30K records)
    outputs/round6/cohort50_summary.csv      (per-city per-SSP gap)
    outputs/round6/cohort50_regression.csv   (per-SSP cohort regression)
    outputs/round6/cohort50_scaling.csv      (SSP-scaling analysis)
    outputs/round6/cohort50_ranking.csv      (policy ranking 2100 × 4 SSPs)
    outputs/round6/cohort50_scaling.png
    outputs/round6/cohort50_ranking.png
"""
from __future__ import annotations
import csv
import json
import logging
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

CODE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_ROOT.parents[2]
OUT_DIR = PROJECT_ROOT / "ai_autoboost" / "outputs" / "round6"
LOG_DIR = PROJECT_ROOT / "ai_autoboost" / "logs"
for d in (OUT_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / f"r6_cohort_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.log",
                            encoding="utf-8"),
    ],
)
log = logging.getLogger("r6_cohort")

sys.path.insert(0, str(CODE_ROOT.parent / "round2_baselines_ablation"))
sys.path.insert(0, str(CODE_ROOT.parent / "round3_mechanism_error"))
sys.path.insert(0, str(CODE_ROOT.parent / "round4_generalization_final"))
sys.path.insert(0, str(CODE_ROOT.parent / "round5_extension"))
import r2_lib as L
from r2_main import METHODS, run_one
from r3_osm_pipeline import fetch_osm_graph
from r4_cohort_anchors import COHORT_R4, CityAnchor
from r5_cohort30 import ADDITIONS as R5_ADDITIONS


# 20 additional cities to make 50-city cohort, spanning more continents.
R6_ADDITIONS: List[CityAnchor] = [
    CityAnchor("Lisbon",       "Portugal",      38.7223,  -9.1393, "coastal",   6.0, 290.0, "Tagus delta, 1755 fault history"),
    CityAnchor("Naples",       "Italy",         40.8518,  14.2681, "coastal",   8.0, 320.0, "Campanian volcanic-sediment"),
    CityAnchor("Athens",       "Greece",        37.9838,  23.7275, "mixed",    11.0, 340.0, "Attican Plain"),
    CityAnchor("Algiers",      "Algeria",       36.7372,   3.0866, "coastal",   7.0, 300.0, "Mitidja basin"),
    CityAnchor("Casablanca",   "Morocco",       33.5731,  -7.5898, "coastal",   8.0, 320.0, "Atlantic coast"),
    CityAnchor("CapeTown",     "SouthAfrica",  -33.9249,  18.4241, "coastal",   9.0, 360.0, "Table Mountain piedmont"),
    CityAnchor("BuenosAires",  "Argentina",    -34.6037, -58.3816, "coastal",   3.5, 230.0, "Río de la Plata delta"),
    CityAnchor("Santiago",     "Chile",        -33.4489, -70.6693, "mixed",    10.0, 330.0, "Andean piedmont, 1985 Valparaíso aftershock zone"),
    CityAnchor("Bogota",       "Colombia",       4.7110, -74.0721, "high_alt",  6.0, 280.0, "Andes 2640m, paleo-lakebed"),
    CityAnchor("Auckland",     "NewZealand",   -36.8485, 174.7633, "coastal",   5.0, 260.0, "Volcanic isthmus"),
    CityAnchor("Sydney",       "Australia",    -33.8688, 151.2093, "coastal",   6.0, 290.0, "Sandstone + Eastern Beaches sand"),
    CityAnchor("Melbourne",    "Australia",    -37.8136, 144.9631, "coastal",   7.0, 300.0, "Port Phillip Bay margin"),
    CityAnchor("Wuhan",        "China",         30.5928, 114.3055, "lowland",   3.5, 230.0, "Yangtze interior delta"),
    CityAnchor("Guangzhou",    "China",         23.1291, 113.2644, "deltaic",   2.5, 220.0, "Pearl River delta"),
    CityAnchor("Bushehr",      "Iran",          28.9684,  50.8385, "coastal",   4.5, 240.0, "Persian Gulf coast"),
    CityAnchor("Baku",         "Azerbaijan",    40.4093,  49.8671, "coastal",   8.0, 320.0, "Caspian Sea margin"),
    CityAnchor("Anchorage",    "USA",           61.2181, -149.9003, "cold",      8.0, 310.0, "Cook Inlet glacial"),
    CityAnchor("Almaty",       "Kazakhstan",    43.2389,  76.8897, "mixed",    14.0, 350.0, "Tian Shan piedmont"),
    CityAnchor("Yerevan",      "Armenia",       40.1792,  44.4991, "mixed",    13.0, 340.0, "Caucasus highlands"),
    CityAnchor("Asuncion",     "Paraguay",     -25.2637, -57.5759, "lowland",   4.0, 240.0, "Paraguay river floodplain"),
]


def cohort50() -> List[CityAnchor]:
    """R4 18 + R5 12 + R6 20 = 50 cities."""
    return list(COHORT_R4) + list(R5_ADDITIONS) + list(R6_ADDITIONS)


# Climate-driver sampler that supports SSP1-2.6 in addition to SSP2-4.5 / SSP5-8.5 / Control
def sample_dGW_r6(rng: np.random.Generator, ssp: str, epoch: int, archetype: str) -> float:
    if epoch == 2020:
        return 0.0
    base = {
        "SSP1-2.6":     {2050: -0.10, 2100: -0.25},     # low-emission baseline
        "SSP2-4.5":     {2050: -0.25, 2100: -0.50},
        "SSP5-8.5":     {2050: -0.50, 2100: -1.00},
        "Control-NoCC": {2050:   0.0, 2100:   0.0},
    }[ssp][epoch]
    sd = {"SSP1-2.6": 0.10, "SSP2-4.5": 0.15, "SSP5-8.5": 0.20, "Control-NoCC": 0.10}[ssp]
    arch_amp = {
        "deltaic":  1.4, "coastal":  1.2, "lowland":  1.0, "mixed":    0.85,
        "inland":   0.70, "arid":    0.40, "cold":    1.10, "high_alt": 0.95,
    }[archetype]
    return float(rng.normal(base, sd) * arch_amp)


# ---------------------------------------------------------------------------
# Run city × SSP × epoch sweep
# ---------------------------------------------------------------------------

def run_city_4ssp(cg: L.CityGraph, n_seeds: int, n_mc: int,
                   Mw: float = 6.5, R_km: float = 25.0) -> List[Dict]:
    rows = []
    ssps = ["Control-NoCC", "SSP1-2.6", "SSP2-4.5", "SSP5-8.5"]
    for seed in range(n_seeds):
        for ssp in ssps:
            for epoch in (2020, 2050, 2100):
                ssp_rng = np.random.default_rng(abs(seed * 100_000 + hash(ssp) % 9973 + epoch))
                for mc in range(n_mc):
                    dGW = sample_dGW_r6(ssp_rng, ssp, epoch, cg.archetype) if cg.archetype != "unknown" else 0.0
                    for method_name in ("B0_static_hazus", "B4_cgstg_full"):
                        cfg = METHODS[method_name]
                        _, d_final, p_liq, pga, _, _ = run_one(cg, Mw, R_km, dGW, cfg, ssp_rng)
                        rows.append({
                            "city": getattr(cg, "city_name", "?"),
                            "country": getattr(cg, "country", ""),
                            "archetype": cg.archetype,
                            "lat": float(getattr(cg, "lat", 0.0)),
                            "lon": float(getattr(cg, "lon", 0.0)),
                            "seed": seed, "ssp": ssp, "epoch": epoch, "mc": mc,
                            "dGW": float(dGW),
                            "method": method_name,
                            "mean_dmg_final": float(d_final),
                            "mean_p_liq": float(p_liq),
                            "mean_pga_g": float(pga),
                            "gw_base_m": float(cg.GW_2020.mean()),
                        })
    return rows


def per_city_per_ssp_gap(rows: List[Dict], city_name: str, ssp: str) -> Dict:
    sub = [r for r in rows if r["city"] == city_name and r["method"] == "B4_cgstg_full" and r["ssp"] == ssp]
    g_2020 = [r["mean_dmg_final"] for r in sub if r["epoch"] == 2020]
    g_2100 = [r["mean_dmg_final"] for r in sub if r["epoch"] == 2100]
    if not g_2020 or not g_2100:
        return {"city": city_name, "ssp": ssp, "n_pairs": 0,
                "mean_gap": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    a = np.array(g_2100); b = np.array(g_2020)
    diff = a - b
    lo, hi = (L.bca_ci(diff, n_resamples=1999, rng_seed=hash((city_name, ssp)) & 0xFFFF)
              if diff.size >= 5 else (float("nan"), float("nan")))
    return {
        "city": city_name, "ssp": ssp, "n_pairs": int(diff.size),
        "mean_gap": float(diff.mean()),
        "ci_lo": float(lo), "ci_hi": float(hi),
        "sign": "positive" if lo > 0 else ("negative" if hi < 0 else "zero-crossing"),
    }


def main() -> int:
    t0 = datetime.utcnow()
    cohort = cohort50()
    log.info(f"R6 cohort50 start: {len(cohort)} cities (R4=18 + R5=12 + R6=20)")
    log.info(f"Archetypes: {sorted(set(c.archetype for c in cohort))}")

    all_rows: List[Dict] = []
    failed: List[str] = []
    successful: List[str] = []
    for city in cohort:
        log.info(f"Loading {city.name} ({city.country}) ...")
        d = {"name": city.name, "lat": city.lat, "lon": city.lon,
             "gw_base": city.gw_base_m, "vs30_mu": city.vs30_mu,
             "archetype_match": city.archetype}
        cg = fetch_osm_graph(d, dist_m=1000, sample_n=100)
        if cg is None:
            failed.append(city.name)
            log.warning(f"  SKIP {city.name}")
            continue
        object.__setattr__(cg, "city_name", city.name)
        object.__setattr__(cg, "country", city.country)
        object.__setattr__(cg, "lat", city.lat)
        object.__setattr__(cg, "lon", city.lon)
        rows = run_city_4ssp(cg, n_seeds=4, n_mc=8)
        all_rows.extend(rows)
        successful.append(city.name)
        log.info(f"  {city.name}: {len(rows)} records")

    log.info(f"Loaded {len(successful)}/{len(cohort)}; failed: {failed}")

    # Persist raw
    with (OUT_DIR / "cohort50_raw.csv").open("w", newline="", encoding="utf-8") as f:
        if all_rows:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)

    # Per-city per-SSP summary
    summary_rows = []
    ssps = ["Control-NoCC", "SSP1-2.6", "SSP2-4.5", "SSP5-8.5"]
    for city in cohort:
        if city.name not in successful:
            continue
        for ssp in ssps:
            s = per_city_per_ssp_gap(all_rows, city.name, ssp)
            s.update({"archetype": city.archetype, "country": city.country,
                      "gw_base_m": city.gw_base_m, "vs30_mu": city.vs30_mu,
                      "lat": city.lat, "lon": city.lon})
            summary_rows.append(s)
    cols = ["city", "country", "archetype", "lat", "lon", "gw_base_m", "vs30_mu",
            "ssp", "n_pairs", "mean_gap", "ci_lo", "ci_hi", "sign"]
    with (OUT_DIR / "cohort50_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in summary_rows:
            r = {k: round(v, 5) if isinstance(v, float) else v for k, v in r.items()}
            w.writerow({k: r.get(k, "") for k in cols})

    # Per-SSP cohort regression
    from scipy.stats import pearsonr, spearmanr
    from sklearn.linear_model import LinearRegression
    reg_rows = []
    for ssp in ssps:
        sub = [r for r in summary_rows if r["ssp"] == ssp and not np.isnan(r["mean_gap"])]
        if len(sub) < 5:
            continue
        x = np.array([1.0 / r["gw_base_m"] for r in sub])
        y = np.array([r["mean_gap"] for r in sub])
        r_p, p_p = pearsonr(x, y)
        r_s, p_s = spearmanr(x, y)
        m = LinearRegression().fit(x.reshape(-1, 1), y)
        r2 = float(m.score(x.reshape(-1, 1), y))
        reg_rows.append({
            "ssp": ssp, "n_cities": len(sub),
            "pearson_r": round(r_p, 4), "pearson_p": round(p_p, 6),
            "spearman_r": round(r_s, 4), "spearman_p": round(p_s, 6),
            "linear_R2": round(r2, 4),
            "slope": round(float(m.coef_[0]), 5),
            "intercept": round(float(m.intercept_), 5),
        })
        log.info(f"{ssp}: n={len(sub)}, r={reg_rows[-1]['pearson_r']}, p={reg_rows[-1]['pearson_p']}, R²={reg_rows[-1]['linear_R2']}, slope={reg_rows[-1]['slope']}")
    with (OUT_DIR / "cohort50_regression.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(reg_rows[0].keys()))
        w.writeheader()
        w.writerows(reg_rows)

    # SSP scaling analysis: how does mean climate gap scale with emission scenario?
    scaling_rows = []
    for ssp in ssps:
        sub = [r for r in summary_rows if r["ssp"] == ssp and not np.isnan(r["mean_gap"])]
        gaps = np.array([r["mean_gap"] for r in sub])
        scaling_rows.append({
            "ssp": ssp, "n_cities": len(sub),
            "mean_gap_across_cohort": round(float(gaps.mean()), 5),
            "median_gap": round(float(np.median(gaps)), 5),
            "p10_gap": round(float(np.percentile(gaps, 10)), 5),
            "p90_gap": round(float(np.percentile(gaps, 90)), 5),
            "max_gap_city": sub[int(np.argmax(gaps))]["city"],
            "max_gap_value": round(float(gaps.max()), 5),
            "n_strictly_positive_CI": int(sum(r["sign"] == "positive" for r in sub)),
            "n_zero_crossing": int(sum(r["sign"] == "zero-crossing" for r in sub)),
        })
        log.info(f"{ssp}: cohort mean gap = {scaling_rows[-1]['mean_gap_across_cohort']}, "
                 f"max = {scaling_rows[-1]['max_gap_value']} ({scaling_rows[-1]['max_gap_city']}), "
                 f"n_positive_CI = {scaling_rows[-1]['n_strictly_positive_CI']}")
    with (OUT_DIR / "cohort50_scaling.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(scaling_rows[0].keys()))
        w.writeheader()
        w.writerows(scaling_rows)

    # Policy ranking: per-city 2100 climate gap × 4 SSPs (CSV for community use)
    ranking_rows = []
    for city in cohort:
        if city.name not in successful:
            continue
        row = {"city": city.name, "country": city.country, "archetype": city.archetype,
               "lat": city.lat, "lon": city.lon, "gw_base_m": city.gw_base_m}
        for ssp in ssps:
            sub = next((r for r in summary_rows if r["city"] == city.name and r["ssp"] == ssp), None)
            if sub:
                row[f"gap_{ssp}"] = round(sub["mean_gap"], 5)
                row[f"gap_{ssp}_ci_lo"] = round(sub["ci_lo"], 5)
                row[f"gap_{ssp}_ci_hi"] = round(sub["ci_hi"], 5)
                row[f"sign_{ssp}"] = sub["sign"]
        ranking_rows.append(row)
    # Sort by SSP5-8.5 gap descending (the worst-case climate scenario)
    ranking_rows.sort(key=lambda r: r.get("gap_SSP5-8.5", -1), reverse=True)
    # Add a column for rank
    for i, r in enumerate(ranking_rows):
        r["rank_under_SSP5-8.5"] = i + 1
    fields_order = ["rank_under_SSP5-8.5", "city", "country", "archetype", "lat", "lon", "gw_base_m"]
    for ssp in ssps:
        fields_order.extend([f"gap_{ssp}", f"gap_{ssp}_ci_lo", f"gap_{ssp}_ci_hi", f"sign_{ssp}"])
    for r in ranking_rows:
        for f_ in fields_order:
            r.setdefault(f_, "")
    with (OUT_DIR / "cohort50_ranking.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields_order)
        w.writeheader()
        w.writerows(ranking_rows)

    # Plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # SSP scaling: mean and p10-p90 box per SSP
        fig, ax = plt.subplots(figsize=(8, 5))
        ssp_order = ["Control-NoCC", "SSP1-2.6", "SSP2-4.5", "SSP5-8.5"]
        cohort_gaps_per_ssp = {ssp: [r["mean_gap"] for r in summary_rows
                                       if r["ssp"] == ssp and not np.isnan(r["mean_gap"])]
                                for ssp in ssp_order}
        bp = ax.boxplot([cohort_gaps_per_ssp[s] for s in ssp_order], labels=ssp_order,
                         patch_artist=True, showmeans=True)
        for patch, c in zip(bp["boxes"], ["#666", "#1f77b4", "#ff7f0e", "#d62728"]):
            patch.set_facecolor(c); patch.set_alpha(0.7)
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_ylabel("Per-city climate-isolated Δ damage (2100 − 2020)")
        ax.set_title(f"Cross-SSP scaling of climate-induced damage gap (n={len(successful)} real OSM cities)")
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "cohort50_scaling.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

        # Per-SSP regression scatter
        fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
        for i, ssp in enumerate(ssp_order):
            sub = [r for r in summary_rows if r["ssp"] == ssp and not np.isnan(r["mean_gap"])]
            x = np.array([1.0 / r["gw_base_m"] for r in sub])
            y = np.array([r["mean_gap"] for r in sub])
            ax = axes[i]
            ax.scatter(x, y, s=40, c="#d62728" if i > 0 else "#666", edgecolors="black")
            if len(x) >= 5:
                m = LinearRegression().fit(x.reshape(-1, 1), y)
                xs = np.linspace(x.min(), x.max(), 50)
                ax.plot(xs, m.predict(xs.reshape(-1, 1)), "k--", lw=1)
                r_p, p_p = pearsonr(x, y)
                ax.set_title(f"{ssp}\nR² = {m.score(x.reshape(-1, 1), y):.3f}, p = {p_p:.2e}")
            ax.axhline(0, color="grey", lw=0.5)
            ax.set_xlabel("1 / baseline GW depth (m⁻¹)")
            if i == 0:
                ax.set_ylabel("Mean Δ damage 2100−2020")
            ax.grid(alpha=0.3)
        fig.suptitle("Per-SSP cohort regression (n=50 real OSM cities)")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "cohort50_per_ssp_regression.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

        # Top-20 cities under SSP5-8.5
        top20 = ranking_rows[:20]
        fig, ax = plt.subplots(figsize=(13, 5.5))
        x = np.arange(len(top20))
        ssp_colors = {"Control-NoCC": "#666", "SSP1-2.6": "#1f77b4",
                       "SSP2-4.5": "#ff7f0e", "SSP5-8.5": "#d62728"}
        width = 0.21
        for i, ssp in enumerate(ssp_order):
            vals = [r.get(f"gap_{ssp}", 0) for r in top20]
            ax.bar(x + (i - 1.5) * width, vals, width=width,
                   color=ssp_colors[ssp], label=ssp)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r['city']}\n({r['archetype']}, {r['gw_base_m']:.1f}m)" for r in top20],
                            rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Climate-isolated Δ damage (2100 − 2020)")
        ax.set_title("Top-20 cities ranked by SSP5-8.5 climate gap — comparison across 4 SSPs")
        ax.axhline(0, color="grey", lw=0.5)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "cohort50_ranking.png", dpi=130, bbox_inches="tight")
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
        "ssp_per_ssp_R2": {r["ssp"]: r["linear_R2"] for r in reg_rows},
    }
    (OUT_DIR / "cohort50_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                                  encoding="utf-8")
    elapsed = (datetime.utcnow() - t0).total_seconds()
    log.info(f"R6 cohort50 done in {elapsed:.1f}s")
    print(f"\n=== R6 cohort50 done: {len(successful)}/{len(cohort)} cities, {len(all_rows)} records ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
