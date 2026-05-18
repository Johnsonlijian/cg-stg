"""Round 4.1 — Expanded cohort: 18 real cities via OSM + CG-STG end-to-end.

Pipeline:
    For each city in COHORT_R4:
        1. Fetch OSM road graph + buildings (1 km radius around centre)
        2. Build CityGraph with city-specific Vs30 + baseline GW
        3. Run B0_static_hazus and B4_cgstg_full at Mw=6.5, R=25 km
           across (SSP5-8.5 + Control-NoCC) × (2020 + 2100) × n_seeds × n_mc
        4. Compute climate-isolated gap (B4 2100 vs B4 2020 SSP5-8.5)

Output:
    outputs/round4/expanded_cohort_raw.csv
    outputs/round4/expanded_cohort_summary.csv     (per-city climate gap + CI)
    outputs/round4/expanded_cohort_regression.csv  (cohort R² on inv-GW)
    outputs/round4/expanded_cohort_climate_gap.png
"""
from __future__ import annotations

import csv
import json
import logging
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

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
        logging.FileHandler(LOG_DIR / f"r4_cohort_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.log",
                            encoding="utf-8"),
    ],
)
log = logging.getLogger("r4_cohort")

# Add code paths
sys.path.insert(0, str(CODE_ROOT.parent / "round2_baselines_ablation"))
sys.path.insert(0, str(CODE_ROOT.parent / "round3_mechanism_error"))
import r2_lib as L
from r2_main import METHODS, sample_dGW, run_one
from r3_osm_pipeline import fetch_osm_graph
from r4_cohort_anchors import COHORT_R4


def run_city(cg: L.CityGraph, n_seeds: int = 6, n_mc: int = 12,
              Mw: float = 6.5, R_km: float = 25.0) -> List[Dict]:
    """Run B0 + B4 for SSP5-8.5 + Control-NoCC × epochs 2020 + 2100."""
    rows = []
    for seed in range(n_seeds):
        for ssp in ("SSP5-8.5", "Control-NoCC"):
            for epoch in (2020, 2100):
                ssp_rng = np.random.default_rng(abs(seed * 100_000 + hash(ssp) % 9973 + epoch))
                for mc in range(n_mc):
                    dGW = sample_dGW(ssp_rng, ssp, epoch, cg.archetype) if cg.archetype != "unknown" else 0.0
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
                            "vs30_mu": float(cg.Vs30.mean()),
                            "n_nodes_graph": int(cg.n_nodes),
                        })
    return rows


def per_city_climate_gap(rows: List[Dict], city_name: str) -> Dict:
    """Compute climate-isolated gap (B4 SSP5-8.5 2100 − B4 SSP5-8.5 2020) with BCa CI."""
    sub = [r for r in rows if r["city"] == city_name and r["method"] == "B4_cgstg_full" and r["ssp"] == "SSP5-8.5"]
    g_2020 = [r["mean_dmg_final"] for r in sub if r["epoch"] == 2020]
    g_2100 = [r["mean_dmg_final"] for r in sub if r["epoch"] == 2100]
    if not g_2020 or not g_2100:
        return {"city": city_name, "n_pairs": 0, "mean_gap": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    # Pair by seed and mc
    a = np.array(g_2100)
    b = np.array(g_2020)
    diff = a - b
    lo, hi = L.bca_ci(diff, n_resamples=2999, rng_seed=hash(city_name) & 0xFFFF) if diff.size >= 5 else (float("nan"), float("nan"))
    return {
        "city": city_name,
        "n_pairs": int(diff.size),
        "mean_gap": float(diff.mean()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "sign": "positive" if lo > 0 else ("negative" if hi < 0 else "zero-crossing"),
    }


def main() -> int:
    t0 = datetime.utcnow()
    log.info(f"R4 cohort start: {len(COHORT_R4)} cities")
    log.info(f"Archetypes covered: {sorted(set(c.archetype for c in COHORT_R4))}")

    all_rows: List[Dict] = []
    failed_cities: List[str] = []
    successful: List[str] = []

    for city in COHORT_R4:
        log.info(f"Loading OSM for {city.name} ({city.country}) @ ({city.lat:.4f}, {city.lon:.4f}) ...")
        city_dict = {"name": city.name, "lat": city.lat, "lon": city.lon,
                     "gw_base": city.gw_base_m, "vs30_mu": city.vs30_mu,
                     "archetype_match": city.archetype}
        cg = fetch_osm_graph(city_dict, dist_m=1000, sample_n=120)
        if cg is None:
            log.warning(f"  SKIP {city.name}")
            failed_cities.append(city.name)
            continue
        object.__setattr__(cg, "city_name", city.name)
        object.__setattr__(cg, "country", city.country)
        object.__setattr__(cg, "lat", city.lat)
        object.__setattr__(cg, "lon", city.lon)

        rows = run_city(cg, n_seeds=6, n_mc=12)
        all_rows.extend(rows)
        successful.append(city.name)
        log.info(f"  {city.name}: {len(rows)} records, n_nodes_graph={cg.n_nodes}")

    log.info(f"Cohort complete: {len(successful)} ok, {len(failed_cities)} failed: {failed_cities}")

    # Persist raw
    with (OUT_DIR / "expanded_cohort_raw.csv").open("w", newline="", encoding="utf-8") as f:
        if all_rows:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)

    # Per-city summary
    summary_rows = []
    for city in COHORT_R4:
        if city.name not in successful:
            continue
        s = per_city_climate_gap(all_rows, city.name)
        s.update({"archetype": city.archetype, "country": city.country,
                  "gw_base_m": city.gw_base_m, "vs30_mu": city.vs30_mu,
                  "lat": city.lat, "lon": city.lon})
        summary_rows.append(s)
    # Sort by mean_gap desc
    summary_rows.sort(key=lambda r: r["mean_gap"], reverse=True)
    with (OUT_DIR / "expanded_cohort_summary.csv").open("w", newline="", encoding="utf-8") as f:
        if summary_rows:
            cols = ["city", "country", "archetype", "lat", "lon", "gw_base_m", "vs30_mu",
                    "n_pairs", "mean_gap", "ci_lo", "ci_hi", "sign"]
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in summary_rows:
                r = {k: round(v, 5) if isinstance(v, float) else v for k, v in r.items()}
                w.writerow({k: r.get(k, "") for k in cols})

    # Cohort regression of climate gap on inverse-GW depth
    if summary_rows:
        from scipy.stats import pearsonr, spearmanr
        from sklearn.linear_model import LinearRegression
        x = np.array([1.0 / r["gw_base_m"] for r in summary_rows])
        y = np.array([r["mean_gap"] for r in summary_rows])
        r_p, p_p = pearsonr(x, y)
        r_s, p_s = spearmanr(x, y)
        model = LinearRegression().fit(x.reshape(-1, 1), y)
        r2 = float(model.score(x.reshape(-1, 1), y))
        reg = {
            "n_cities": len(summary_rows),
            "pearson_r": round(r_p, 4),
            "pearson_p": round(p_p, 6),
            "spearman_r": round(r_s, 4),
            "spearman_p": round(p_s, 6),
            "linear_R2": round(r2, 4),
            "slope": round(float(model.coef_[0]), 4),
            "intercept": round(float(model.intercept_), 5),
        }
        with (OUT_DIR / "expanded_cohort_regression.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(reg.keys()))
            w.writeheader()
            w.writerow(reg)
        log.info(f"COHORT REGRESSION on 1/GW: n={reg['n_cities']}, Pearson r={reg['pearson_r']}, p={reg['pearson_p']}, R²={reg['linear_R2']}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 5.5))
        cities_sorted = sorted(summary_rows, key=lambda r: r["mean_gap"], reverse=True)
        x = np.arange(len(cities_sorted))
        means = [r["mean_gap"] for r in cities_sorted]
        yerr = [[m - r["ci_lo"], r["ci_hi"] - m] for m, r in zip(means, cities_sorted)]
        yerr = np.array(yerr).T
        colors = []
        for r in cities_sorted:
            if r["sign"] == "positive": colors.append("#d62728")
            elif r["sign"] == "negative": colors.append("#1f77b4")
            else: colors.append("#aaaaaa")
        ax.bar(x, means, yerr=yerr, capsize=3, color=colors)
        ax.set_xticks(x)
        labels = [f"{r['city']}\n({r['archetype']}, gw={r['gw_base_m']:.1f}m)" for r in cities_sorted]
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Climate-isolated Δ damage (2100 − 2020, SSP5-8.5)")
        ax.set_title(f"Round 4 expanded cohort: per-city climate gap (n={len(cities_sorted)} real OSM cities)\n"
                     f"Red = CI positive; Blue = CI negative; Grey = zero-crossing")
        ax.axhline(0, color="grey", lw=0.5)
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "expanded_cohort_climate_gap.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

        # Scatter inv-GW vs climate gap
        fig, ax = plt.subplots(figsize=(7.5, 5))
        x_plot = np.array([1.0 / r["gw_base_m"] for r in summary_rows])
        y_plot = np.array([r["mean_gap"] for r in summary_rows])
        ax.scatter(x_plot, y_plot, s=60, c="#d62728" if r2 > 0 else "#1f77b4", edgecolors="black")
        for r in summary_rows:
            ax.annotate(r["city"], (1.0 / r["gw_base_m"], r["mean_gap"]),
                        fontsize=7, alpha=0.7,
                        xytext=(5, 0), textcoords="offset points")
        # regression line
        xs_line = np.linspace(x_plot.min(), x_plot.max(), 100)
        ys_line = model.predict(xs_line.reshape(-1, 1))
        ax.plot(xs_line, ys_line, "k--", lw=1, alpha=0.6,
                label=f"OLS: y = {reg['slope']:.4f} x {'+' if reg['intercept'] >= 0 else ''}{reg['intercept']:.4f}\nR² = {reg['linear_R2']:.3f}, Pearson p = {reg['pearson_p']:.4f}")
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_xlabel("1 / baseline groundwater depth (m⁻¹)")
        ax.set_ylabel("Mean climate-isolated Δ damage (2100−2020, SSP5-8.5)")
        ax.set_title(f"Round 4 cohort regression on inverse-GW depth (n={len(summary_rows)} real cities)")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "expanded_cohort_regression.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        log.warning(f"plot fail: {e}")

    meta = {
        "started_utc": t0.isoformat() + "Z",
        "elapsed_seconds": round((datetime.utcnow() - t0).total_seconds(), 2),
        "n_cities_targeted": len(COHORT_R4),
        "n_cities_successful": len(successful),
        "successful_cities": successful,
        "failed_cities": failed_cities,
        "n_records": len(all_rows),
    }
    (OUT_DIR / "expanded_cohort_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                                       encoding="utf-8")
    elapsed = (datetime.utcnow() - t0).total_seconds()
    log.info(f"R4 cohort done in {elapsed:.1f}s")
    print(f"\n=== R4 cohort expanded to {len(successful)}/{len(COHORT_R4)} cities, {len(all_rows)} records ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
