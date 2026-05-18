"""Round 7.2 — Decadal climate-gap trajectories for top-10 cities × 4 SSPs.

For each of the top-10 SSP5-8.5 ranked cities (from R7 hires), compute the
climate-isolated damage gap (B4 epoch − B4 2020) at 9 decadal slices
(2020, 2030, ..., 2100) under all 4 SSPs.

This produces a time-domain picture: how does the climate-induced damage
accumulate decade-by-decade? Useful for adaptation planning.

Output:
    outputs/round7/decadal_trajectories.csv
    outputs/round7/decadal_trajectories.png
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
        logging.FileHandler(LOG_DIR / f"r7_decadal_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.log",
                            encoding="utf-8"),
    ],
)
log = logging.getLogger("r7_decadal")

sys.path.insert(0, str(CODE_ROOT.parent / "round2_baselines_ablation"))
sys.path.insert(0, str(CODE_ROOT.parent / "round3_mechanism_error"))
sys.path.insert(0, str(CODE_ROOT.parent / "round6_extension"))
import r2_lib as L
from r2_main import METHODS, run_one
from r3_osm_pipeline import fetch_osm_graph
from r6_cohort50 import cohort50


SSPS = ["Control-NoCC", "SSP1-2.6", "SSP2-4.5", "SSP5-8.5"]
EPOCHS_DECADAL = (2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100)

# Top-10 from R7 hires
TOP10 = ["Shanghai", "Christchurch", "Dhaka", "NewOrleans", "Guangzhou",
          "Manila", "Wuhan", "Tianjin", "Bangkok", "HoChiMinh"]


def sample_dGW_decadal(rng: np.random.Generator, ssp: str, epoch: int, archetype: str) -> float:
    """Linear interpolation of dGW between R6 anchor epochs (2020, 2050, 2100) at decadal granularity."""
    if epoch == 2020:
        return 0.0
    anchors = {
        "Control-NoCC": {2020: 0.0, 2050: 0.0, 2100: 0.0},
        "SSP1-2.6":     {2020: 0.0, 2050: -0.10, 2100: -0.25},
        "SSP2-4.5":     {2020: 0.0, 2050: -0.25, 2100: -0.50},
        "SSP5-8.5":     {2020: 0.0, 2050: -0.50, 2100: -1.00},
    }[ssp]
    if epoch <= 2050:
        f = (epoch - 2020) / 30.0
        base = (1 - f) * anchors[2020] + f * anchors[2050]
    else:
        f = (epoch - 2050) / 50.0
        base = (1 - f) * anchors[2050] + f * anchors[2100]
    sd = {"Control-NoCC": 0.05, "SSP1-2.6": 0.08, "SSP2-4.5": 0.12, "SSP5-8.5": 0.18}[ssp]
    arch_amp = {
        "deltaic": 1.4, "coastal": 1.2, "lowland": 1.0, "mixed": 0.85,
        "inland": 0.70, "arid": 0.40, "cold": 1.10, "high_alt": 0.95,
    }[archetype]
    return float(rng.normal(base, sd) * arch_amp)


def run_decadal_for_city(cg: L.CityGraph, n_seeds: int = 6, n_mc: int = 12) -> List[Dict]:
    cfg = METHODS["B4_cgstg_full"]
    rows = []
    for seed in range(n_seeds):
        for ssp in SSPS:
            for epoch in EPOCHS_DECADAL:
                ssp_rng = np.random.default_rng(abs(seed * 100_000 + hash(ssp) % 9973 + epoch))
                for mc in range(n_mc):
                    dGW = sample_dGW_decadal(ssp_rng, ssp, epoch, cg.archetype)
                    _, d_final, _, _, _, _ = run_one(cg, 6.5, 25.0, dGW, cfg, ssp_rng)
                    rows.append({
                        "city": getattr(cg, "city_name", "?"),
                        "archetype": cg.archetype,
                        "seed": seed, "ssp": ssp, "epoch": epoch, "mc": mc,
                        "dGW": float(dGW),
                        "mean_dmg_final": float(d_final),
                    })
    return rows


def main() -> int:
    t0 = datetime.utcnow()
    cohort = cohort50()
    target_anchors = {c.name: c for c in cohort if c.name in TOP10}
    log.info(f"R7 decadal start: {len(target_anchors)} top cities × {len(SSPS)} SSPs × {len(EPOCHS_DECADAL)} epochs × 72 samples")

    all_rows: List[Dict] = []
    for name, city in target_anchors.items():
        log.info(f"Loading {name} ...")
        d = {"name": name, "lat": city.lat, "lon": city.lon,
             "gw_base": city.gw_base_m, "vs30_mu": city.vs30_mu,
             "archetype_match": city.archetype}
        cg = fetch_osm_graph(d, dist_m=1000, sample_n=100)
        if cg is None:
            continue
        object.__setattr__(cg, "city_name", name)
        rows = run_decadal_for_city(cg, n_seeds=6, n_mc=12)
        all_rows.extend(rows)
        log.info(f"  {name}: {len(rows)} records")

    # Persist
    with (OUT_DIR / "decadal_raw.csv").open("w", newline="", encoding="utf-8") as f:
        if all_rows:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader(); w.writerows(all_rows)

    # Per-(city, ssp, epoch) mean and CI of (B4 epoch − B4 2020)
    summary = []
    by_city_ssp_seed_2020 = {}
    for r in all_rows:
        if r["epoch"] == 2020:
            key = (r["city"], r["ssp"], r["seed"])
            by_city_ssp_seed_2020.setdefault(key, []).append(r["mean_dmg_final"])

    for city in TOP10:
        for ssp in SSPS:
            for epoch in EPOCHS_DECADAL:
                if epoch == 2020:
                    continue
                # gap = mean[B4 city,ssp,epoch] - mean[B4 city,ssp,2020] across same seed
                gaps_per_seed = []
                seeds_in = sorted({r["seed"] for r in all_rows if r["city"] == city and r["ssp"] == ssp})
                for s in seeds_in:
                    e2020_mean = np.mean(by_city_ssp_seed_2020.get((city, ssp, s), [np.nan]))
                    e_now = [r["mean_dmg_final"] for r in all_rows
                             if r["city"] == city and r["ssp"] == ssp and r["seed"] == s and r["epoch"] == epoch]
                    if not e_now:
                        continue
                    gaps_per_seed.append(np.mean(e_now) - e2020_mean)
                gaps_per_seed = np.array(gaps_per_seed)
                if gaps_per_seed.size < 3:
                    continue
                ci_lo, ci_hi = L.bca_ci(gaps_per_seed, n_resamples=2999, rng_seed=abs(hash((city, ssp, epoch))) & 0xFFFF)
                summary.append({
                    "city": city, "ssp": ssp, "epoch": int(epoch),
                    "n_seeds": int(gaps_per_seed.size),
                    "mean_gap": round(float(gaps_per_seed.mean()), 5),
                    "ci_lo": round(ci_lo, 5),
                    "ci_hi": round(ci_hi, 5),
                    "sign": "positive" if ci_lo > 0 else ("negative" if ci_hi < 0 else "zero-crossing"),
                })
    cols = ["city", "ssp", "epoch", "n_seeds", "mean_gap", "ci_lo", "ci_hi", "sign"]
    with (OUT_DIR / "decadal_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(summary)

    # Plot: one panel per top-10 city, x=epoch, y=gap, lines per SSP
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 5, figsize=(22, 8), sharex=True, sharey=True)
        axes_flat = axes.ravel()
        ssp_colors = {"Control-NoCC": "#666666", "SSP1-2.6": "#1f77b4",
                       "SSP2-4.5": "#ff7f0e", "SSP5-8.5": "#d62728"}
        for i, city in enumerate(TOP10):
            ax = axes_flat[i]
            for ssp in SSPS:
                sub = [r for r in summary if r["city"] == city and r["ssp"] == ssp]
                if not sub:
                    continue
                sub.sort(key=lambda r: r["epoch"])
                xs = [r["epoch"] for r in sub]
                ys = [r["mean_gap"] for r in sub]
                lo = [r["ci_lo"] for r in sub]
                hi = [r["ci_hi"] for r in sub]
                ax.plot(xs, ys, marker="o", color=ssp_colors[ssp], label=ssp, linewidth=1.5)
                ax.fill_between(xs, lo, hi, color=ssp_colors[ssp], alpha=0.18)
            ax.axhline(0, color="grey", lw=0.5)
            ax.set_title(city, fontsize=10)
            ax.grid(alpha=0.3)
            if i % 5 == 0:
                ax.set_ylabel("Δ damage (epoch − 2020)")
            if i >= 5:
                ax.set_xlabel("Year")
        axes_flat[0].legend(loc="upper left", fontsize=7)
        fig.suptitle("Decadal climate-isolated damage trajectories — top-10 CG-STG-ranked cities × 4 SSPs", fontsize=12)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "decadal_trajectories.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        log.warning(f"plot fail: {e}")

    meta = {
        "elapsed_seconds": round((datetime.utcnow() - t0).total_seconds(), 2),
        "n_cities": len(target_anchors),
        "n_records": len(all_rows),
        "n_summary_rows": len(summary),
    }
    (OUT_DIR / "decadal_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info(f"R7 decadal done in {meta['elapsed_seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
