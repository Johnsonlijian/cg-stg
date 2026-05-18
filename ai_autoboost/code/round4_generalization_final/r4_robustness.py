"""Round 4.3 — Robustness sweep.

For each perturbation type, recompute per-city climate gap + cohort regression:
    (a) PGA noise ε ~ LogNormal(0, σ ∈ {0.05, 0.10, 0.20})  → 3 levels
    (b) Soil-moisture noise ε ~ Normal(0, σ ∈ {0.05, 0.10, 0.15})  → 3 levels
    (c) Random dependency-graph perturbation: with prob {0.05, 0.10, 0.20}, rewire
        each non-building edge to a random other class-matched node  → 3 levels

For each perturbation cell: 4 seeds × 6 cities (subset, for speed) × 8 MC.

Output:
    outputs/round4/robustness_summary.csv
    outputs/round4/robustness_heatmap.png
"""
from __future__ import annotations

import csv
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

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
        logging.FileHandler(LOG_DIR / f"r4_rob_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.log",
                            encoding="utf-8"),
    ],
)
log = logging.getLogger("r4_rob")

sys.path.insert(0, str(CODE_ROOT.parent / "round2_baselines_ablation"))
sys.path.insert(0, str(CODE_ROOT))
import r2_lib as L
from r2_main import sample_dGW, run_one, METHODS
from r4_cohort_anchors import COHORT_R4


# Pick a representative subset of 6 cities spanning archetypes
SUBSET = ["Christchurch", "Tianjin", "Bangkok", "Lima", "Tokyo", "Cairo"]


def perturb_graph(cg: L.CityGraph, rewire_prob: float, rng: np.random.Generator) -> L.CityGraph:
    """Return a copy of the graph with non-building edges rewired with probability."""
    A_new = cg.adjacency.copy()
    n = cg.n_nodes
    # find non-building → non-building edges (i.e., class != 0)
    for i in range(n):
        if cg.asset_class[i] == 0:
            continue
        for j in range(n):
            if A_new[i, j] != 0 and cg.asset_class[j] != 0:
                if rng.random() < rewire_prob:
                    # rewire to a random other node of same class as j
                    cand = np.where((cg.asset_class == cg.asset_class[j]) & (np.arange(n) != j))[0]
                    if cand.size > 0:
                        k = int(rng.choice(cand))
                        A_new[i, j], A_new[i, k] = 0.0, A_new[i, j]
    cg_new = L.CityGraph(
        n_nodes=cg.n_nodes, Vs30=cg.Vs30.copy(), GW_2020=cg.GW_2020.copy(),
        asset_class=cg.asset_class.copy(), x_km=cg.x_km.copy(), y_km=cg.y_km.copy(),
        adjacency=A_new, archetype=cg.archetype,
    )
    return cg_new


def perturbed_run(cg: L.CityGraph, dGW: float, Mw: float, R_km: float,
                  pga_noise_sigma: float, soil_noise_sigma: float,
                  rng: np.random.Generator) -> float:
    """One CG-STG run with PGA + soil noise applied. Returns mean final damage."""
    cfg = METHODS["B4_cgstg_full"]
    R_vec = np.full(cg.n_nodes, R_km)
    pga = L.bssa14_pga(Mw, R_vec, cg.Vs30, fault="SS")
    # Base BSSA14 sigma + perturbation
    pga = pga * np.exp(rng.normal(0.0, 0.72, size=cg.n_nodes))
    if pga_noise_sigma > 0:
        pga = pga * np.exp(rng.normal(0.0, pga_noise_sigma, size=cg.n_nodes))
    GW_t = np.clip(cg.GW_2020 + dGW, 0.3, None)
    # Soil-moisture surrogate noise via dGW perturbation
    if soil_noise_sigma > 0:
        GW_t = np.clip(GW_t + rng.normal(0, soil_noise_sigma * 5.0, size=cg.n_nodes), 0.3, None)
    p_liq = L.liquefaction_probability(Mw, pga, cg.Vs30, GW_t, depth_m=3.0)
    dmg_init, _, _ = L.damage_ensemble(pga, cg.asset_class, p_liq)
    d_final, _ = L.physics_cascading(dmg_init, cg, n_steps=8,
                                       transmission_kappa=0.15, recovery_threshold=0.10)
    return float(d_final.mean())


def make_city_graphs() -> Dict[str, L.CityGraph]:
    sys.path.insert(0, str(CODE_ROOT.parent / "round3_mechanism_error"))
    from r3_osm_pipeline import fetch_osm_graph
    cities = {}
    for a in COHORT_R4:
        if a.name not in SUBSET:
            continue
        d = {"name": a.name, "lat": a.lat, "lon": a.lon, "gw_base": a.gw_base_m,
             "vs30_mu": a.vs30_mu, "archetype_match": a.archetype}
        cg = fetch_osm_graph(d, dist_m=1000, sample_n=100)
        if cg is None:
            continue
        object.__setattr__(cg, "city_name", a.name)
        object.__setattr__(cg, "country", a.country)
        cities[a.name] = cg
        log.info(f"  loaded {a.name}: n_nodes={cg.n_nodes} archetype={cg.archetype}")
    return cities


def compute_gap(cg: L.CityGraph, n_seeds: int, n_mc: int,
                pga_noise_sigma: float, soil_noise_sigma: float,
                rewire_prob: float, rng: np.random.Generator) -> float:
    """Compute climate-isolated gap (2100 − 2020 under SSP5-8.5) under given noise."""
    if rewire_prob > 0:
        cg_use = perturb_graph(cg, rewire_prob, rng)
    else:
        cg_use = cg
    gaps = []
    for seed in range(n_seeds):
        seed_rng = np.random.default_rng(abs(seed * 100_000 + hash(cg.archetype) % 9973))
        per_mc_2020 = []
        per_mc_2100 = []
        for _ in range(n_mc):
            dGW_2020 = sample_dGW(seed_rng, "SSP5-8.5", 2020, cg.archetype)
            dGW_2100 = sample_dGW(seed_rng, "SSP5-8.5", 2100, cg.archetype)
            per_mc_2020.append(perturbed_run(cg_use, dGW_2020, 6.5, 25.0,
                                               pga_noise_sigma, soil_noise_sigma, seed_rng))
            per_mc_2100.append(perturbed_run(cg_use, dGW_2100, 6.5, 25.0,
                                               pga_noise_sigma, soil_noise_sigma, seed_rng))
        gaps.append(np.mean(per_mc_2100) - np.mean(per_mc_2020))
    return float(np.mean(gaps))


def main() -> int:
    t0 = datetime.utcnow()
    log.info("R4 robustness start")
    cities = make_city_graphs()
    log.info(f"Loaded {len(cities)} city graphs for robustness subset")

    perturbations = [
        ("baseline", 0.0, 0.0, 0.0),
        ("PGA_sigma_0.05", 0.05, 0.0, 0.0),
        ("PGA_sigma_0.10", 0.10, 0.0, 0.0),
        ("PGA_sigma_0.20", 0.20, 0.0, 0.0),
        ("soil_sigma_0.05", 0.0, 0.05, 0.0),
        ("soil_sigma_0.10", 0.0, 0.10, 0.0),
        ("soil_sigma_0.15", 0.0, 0.15, 0.0),
        ("graph_rewire_0.05", 0.0, 0.0, 0.05),
        ("graph_rewire_0.10", 0.0, 0.0, 0.10),
        ("graph_rewire_0.20", 0.0, 0.0, 0.20),
        ("combined_0.20+0.15+0.10", 0.20, 0.15, 0.10),
    ]

    n_seeds = 4
    n_mc = 8

    rng_master = np.random.default_rng(2026_05_15)
    rows = []
    for pert_name, pga_s, soil_s, rewire in perturbations:
        log.info(f"  perturbation: {pert_name}")
        city_gaps = {}
        for name, cg in cities.items():
            rng = np.random.default_rng(rng_master.integers(0, 1 << 31))
            gap = compute_gap(cg, n_seeds, n_mc, pga_s, soil_s, rewire, rng)
            city_gaps[name] = gap
            log.info(f"    {name}: gap = {gap:+.5f}")
        # cohort regression on inv-GW
        x = np.array([1.0 / next(c.gw_base_m for c in COHORT_R4 if c.name == name)
                       for name in city_gaps])
        y = np.array(list(city_gaps.values()))
        from scipy.stats import pearsonr
        from sklearn.linear_model import LinearRegression
        r_p, p_p = pearsonr(x, y)
        m = LinearRegression().fit(x.reshape(-1, 1), y)
        rows.append({
            "perturbation": pert_name, "pga_sigma": pga_s, "soil_sigma": soil_s,
            "rewire_prob": rewire,
            "pearson_r": round(float(r_p), 4), "pearson_p": round(float(p_p), 6),
            "R2": round(float(m.score(x.reshape(-1, 1), y)), 4),
            **{f"gap_{name}": round(g, 5) for name, g in city_gaps.items()},
        })
        log.info(f"    R²={rows[-1]['R2']} Pearson r={rows[-1]['pearson_r']} p={rows[-1]['pearson_p']}")

    # Persist
    fields = sorted({k for r in rows for k in r.keys()})
    leading = ["perturbation", "pga_sigma", "soil_sigma", "rewire_prob", "R2", "pearson_r", "pearson_p"]
    fields = leading + [f for f in fields if f not in leading]
    for r in rows:
        for f in fields:
            r.setdefault(f, "")
    with (OUT_DIR / "robustness_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # Heatmap
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        df = pd.DataFrame(rows)
        baseline_r2 = float(df[df["perturbation"] == "baseline"]["R2"].values[0])
        fig, ax = plt.subplots(figsize=(10, 4.5))
        labels = df["perturbation"].values
        vals = df["R2"].values
        bars = ax.bar(range(len(df)), vals, color="#1f77b4")
        ax.axhline(baseline_r2, color="red", lw=1.5, linestyle="--",
                   label=f"Baseline R² = {baseline_r2}")
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("R² of climate gap vs inv-GW depth (subset 6 cities)")
        ax.set_title("Round 4 robustness — R² stability under input perturbations")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.2f}",
                    ha="center", fontsize=8)
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "robustness_heatmap.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        log.warning(f"plot fail: {e}")

    elapsed = (datetime.utcnow() - t0).total_seconds()
    log.info(f"R4 robustness done in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
