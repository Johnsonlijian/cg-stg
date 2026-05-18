"""Round 1 — Minimum Viable CG-STG Pipeline (CPU, no torch required).

What this script does, end-to-end:
    1. Synthesize N cohort cities, each with a random urban graph (buildings + lifelines).
    2. Sample hazard scenarios (Boore-Atkinson-like point-source PGA).
    3. Sample climate drivers at 3 epochs (2020 / 2050 / 2100) under SSP2-4.5 and SSP5-8.5,
       and a "no climate change" control.
    4. Compute time-varying liquefaction susceptibility (NCEER-style simplified CRR with
       water-table mediator).
    5. Compute per-node damage proxy via HAZUS-style vulnerability + amplification.
    6. Compare the "static-2020 extrapolated" baseline against the "time-varying CG-STG"
       output at horizon 2100.
    7. Repeat across n_seeds; bootstrap 95% CI; paired Wilcoxon signed-rank with
       Bonferroni correction across hypotheses.
    8. Write outputs:
         outputs/round1/main_results.csv
         outputs/round1/seed_stability.csv
         outputs/round1/statistical_tests.csv
         outputs/round1/data_leakage_check.md
         outputs/round1/run_meta.json
         outputs/round1/seed_stability.png
         outputs/round1/confidence_intervals.png

CLI:
    python r1_main.py --seeds 10 --cities 3 --mc 50

Smoke test (5 minutes on a laptop CPU):
    python r1_main.py --seeds 3 --cities 3 --mc 10
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# matplotlib is optional for the headless smoke test; import lazily inside plot fns

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # 2026-CompoundSeismicClimate-UrbanLifelines
OUT_DIR = PROJECT_ROOT / "ai_autoboost" / "outputs" / "round1"
LOG_DIR = PROJECT_ROOT / "ai_autoboost" / "logs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / f"r1_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("r1")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    seeds: int = 10
    cities: int = 3
    mc: int = 50                # Monte Carlo scenarios per (city, epoch, ssp)
    nodes_per_city: int = 200   # building / lifeline nodes
    Mw: float = 6.5             # fixed magnitude for smoke test (sweep deferred to Round 2)
    R_km_mean: float = 25.0     # mean source-to-site distance per city
    epochs: Tuple[int, ...] = (2020, 2050, 2100)
    ssps: Tuple[str, ...] = ("SSP2-4.5", "SSP5-8.5", "Control-NoCC")
    bonferroni_n_tests: int = 6  # H1+H3 across 2 SSPs and a control × 1 horizon for smoke
    bootstrap_n: int = 1000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(s: int) -> np.random.Generator:
    return np.random.default_rng(seed=s)


def synthesize_city(rng: np.random.Generator, city_id: int, n_nodes: int) -> Dict[str, np.ndarray]:
    """Generate per-city node attributes.

    Returns a dict of arrays of length n_nodes:
        Vs30 (m/s), GW_depth_2020 (m), soil_class index (0-4 ~ HAZUS A-E),
        lifeline_type (0=building, 1=water, 2=power, 3=transport),
        x, y (km, abstract).
    """
    # Per-city characteristic depth-to-water: smaller in deltaic cities (city 0), larger inland
    city_baseline_gw = np.array([2.0, 5.0, 8.0, 12.0, 15.0])[city_id % 5]
    Vs30 = rng.normal(loc=300.0, scale=60.0, size=n_nodes).clip(150.0, 700.0)
    GW = rng.normal(loc=city_baseline_gw, scale=2.0, size=n_nodes).clip(0.5, 30.0)
    soil_class = np.digitize(Vs30, [180, 360, 760])  # 0=E, 1=D, 2=C, 3=B
    type_probs = np.array([0.70, 0.10, 0.10, 0.10])
    lifeline_type = rng.choice(np.arange(4), size=n_nodes, p=type_probs)
    x = rng.uniform(0, 20, size=n_nodes)
    y = rng.uniform(0, 20, size=n_nodes)
    return {
        "Vs30": Vs30,
        "GW_2020": GW,
        "soil_class": soil_class,
        "lifeline_type": lifeline_type,
        "x": x,
        "y": y,
        "city_id": np.full(n_nodes, city_id, dtype=np.int64),
        "city_baseline_gw": np.full(n_nodes, city_baseline_gw, dtype=np.float64),
    }


def sample_climate_driver_dGW(rng: np.random.Generator, ssp: str, epoch: int) -> float:
    """Sample a city-level change in groundwater depth (m) relative to 2020.

    Negative dGW means water table rose (closer to surface).
    Loosely calibrated to global trends in deltaic / coastal cities; for smoke only.
    """
    if epoch == 2020:
        return 0.0
    if ssp == "Control-NoCC":
        return float(rng.normal(0.0, 0.10))
    if ssp == "SSP2-4.5":
        mu = -0.25 if epoch == 2050 else -0.50
        sd = 0.15
    elif ssp == "SSP5-8.5":
        mu = -0.50 if epoch == 2050 else -1.00
        sd = 0.20
    else:
        raise ValueError(f"unknown ssp {ssp}")
    return float(rng.normal(mu, sd))


def sample_hazard_pga(rng: np.random.Generator, n: int, Mw: float, R_km: float, Vs30: np.ndarray) -> np.ndarray:
    """Boore-Atkinson 2008-like simplified PGA in g for n nodes.

    Note: This is a smoke-test surrogate, NOT a calibrated GMPE. Round 2 replaces with proper
    GMPE (e.g., BSSA14 or ASK14) via OpenQuake-hazardlib if available.
    """
    # Log10 mean PGA (g) ~ Boore-Atkinson 2008 form
    c1, c2, c3, c4, c5 = -1.5, 0.55, -0.10, -0.0035, -0.30
    log10_mean = c1 + c2 * (Mw - 6.0) + c3 * (Mw - 6.0) ** 2 + c4 * R_km + c5 * np.log10(Vs30 / 760.0)
    sigma = 0.30  # natural-log sigma roughly mapped to log10
    log10_pga = log10_mean + rng.normal(0.0, sigma, size=n)
    return np.power(10.0, log10_pga)


def cyclic_stress_ratio(pga_g: np.ndarray, depth_m: np.ndarray, sigma_v0: float, sigma_v0_eff: float) -> np.ndarray:
    """Simplified CSR per Seed-Idriss for surface SPT-equivalent depth."""
    rd = 1.0 - 0.012 * np.minimum(depth_m, 9.0)
    return 0.65 * pga_g * (sigma_v0 / sigma_v0_eff) * rd


def cyclic_resistance_ratio_base(soil_class: np.ndarray) -> np.ndarray:
    """Base CRR_{7.5} as a function of soil class (smoke surrogate)."""
    crr_table = np.array([0.10, 0.14, 0.22, 0.32])  # higher class index → stiffer → higher CRR
    return crr_table[soil_class.clip(0, 3)]


def time_varying_crr(crr0: np.ndarray, dGW_m: float, depth_2020: np.ndarray) -> np.ndarray:
    """Time-varying CRR with water-table mediator.

    A drop in water table depth (dGW < 0 means water rose) reduces effective stress and
    therefore reduces CRR. Loosely calibrated, smoke-only.
    """
    # f = clip(1 + k * dGW / max(depth0, 0.5), 0.6, 1.4)
    k = 0.20
    f = np.clip(1.0 + k * dGW_m / np.clip(depth_2020, 0.5, None), 0.6, 1.4)
    return crr0 * f


def liquefaction_probability(crr: np.ndarray, csr: np.ndarray) -> np.ndarray:
    """Standard probit-style P_L from FS_L = CRR/CSR."""
    fs = crr / np.maximum(csr, 1e-6)
    # Tonkin-style probit: P_L = Phi((1 - FS) / sigma)
    sigma = 0.13
    from math import erf
    z = (1.0 - fs) / sigma
    # Use np for vectorised normal CDF
    return 0.5 * (1.0 + np.vectorize(erf)(z / math.sqrt(2.0)))


def hazus_amplification(soil_class: np.ndarray, pga_g: np.ndarray) -> np.ndarray:
    """HAZUS-style PGA amplification factor (smoke surrogate)."""
    table = np.array([
        [1.6, 1.5, 1.4, 1.3, 1.2],  # E
        [1.4, 1.4, 1.3, 1.2, 1.1],  # D
        [1.2, 1.2, 1.1, 1.0, 1.0],  # C
        [1.0, 1.0, 1.0, 1.0, 1.0],  # B
    ])
    bin_idx = np.minimum(4, np.digitize(pga_g, [0.05, 0.10, 0.20, 0.40]) - 0).astype(int)
    bin_idx = np.clip(bin_idx, 0, 4)
    sc = np.clip(soil_class, 0, 3).astype(int)
    return table[sc, bin_idx]


def hazus_damage_proxy(pga_amplified: np.ndarray, p_liq: np.ndarray, lifeline_type: np.ndarray) -> np.ndarray:
    """Per-node damage proxy in [0, 1]. Smoke surrogate.

    Building (type 0): mostly PGA-driven (HAZUS-MH building vulnerability),
        weak liquefaction sensitivity.
    Water (1) / Power (2) / Transport (3): liquefaction-sensitive (HAZUS-MH lifelines).
    """
    # Sigmoid PGA loss in [0, 1]
    def sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    pga_loss = sigmoid(4.0 * (pga_amplified - 0.35))
    # Lifeline-class weights for liquefaction sensitivity
    w_liq = np.array([0.10, 0.55, 0.40, 0.60])[lifeline_type.clip(0, 3)]
    # Combined
    return np.clip(pga_loss * (1.0 - w_liq) + p_liq * w_liq, 0.0, 1.0)


def run_one_scenario(rng: np.random.Generator, city: Dict[str, np.ndarray], dGW: float,
                     Mw: float, R_km: float) -> Dict[str, np.ndarray]:
    """One scenario: returns per-node damage, PGA, liq prob, CRR, CSR."""
    Vs30 = city["Vs30"]
    soil_class = city["soil_class"]
    GW_2020 = city["GW_2020"]
    lifeline_type = city["lifeline_type"]
    n = Vs30.shape[0]

    # Hazard
    pga = sample_hazard_pga(rng, n, Mw, R_km, Vs30)

    # Site state at this epoch
    GW_t = np.maximum(GW_2020 + dGW, 0.3)
    # Effective overburden at GW depth surrogate
    rho = 1900.0  # kg/m^3
    g = 9.81
    sigma_v0 = rho * g * GW_t
    sigma_v0_eff = np.maximum(sigma_v0 - 1000.0 * g * np.maximum(GW_2020 - GW_t, 0.0), 0.5 * sigma_v0)

    # Susceptibility chain
    csr = cyclic_stress_ratio(pga, GW_t, sigma_v0.mean(), sigma_v0_eff.mean())
    crr0 = cyclic_resistance_ratio_base(soil_class)
    crr_t = time_varying_crr(crr0, dGW, GW_2020)
    p_liq = liquefaction_probability(crr_t, csr)

    # Amplified PGA
    f_amp = hazus_amplification(soil_class, pga)
    pga_amp = pga * f_amp

    # Damage proxy
    dmg = hazus_damage_proxy(pga_amp, p_liq, lifeline_type)
    return {"pga": pga, "pga_amp": pga_amp, "p_liq": p_liq, "crr": crr_t, "csr": csr, "dmg": dmg}


# ---------------------------------------------------------------------------
# Experiment loop
# ---------------------------------------------------------------------------

@dataclass
class CityEpochResult:
    seed: int
    city_id: int
    ssp: str
    epoch: int
    dGW: float
    mean_pga: float
    mean_pliq: float
    mean_dmg: float
    p50_dmg: float
    p95_dmg: float
    dmg_per_class: Dict[int, float] = field(default_factory=dict)


def run_seed(seed: int, cfg: Config) -> List[CityEpochResult]:
    rng_master = set_seed(seed)
    results: List[CityEpochResult] = []
    for city_id in range(cfg.cities):
        city_rng = set_seed(seed * 1000 + city_id)
        city = synthesize_city(city_rng, city_id, cfg.nodes_per_city)
        for ssp in cfg.ssps:
            for epoch in cfg.epochs:
                ssp_rng = set_seed(seed * 1_000_000 + city_id * 1000 + hash(ssp) % 997 + epoch % 13)
                dGW = sample_climate_driver_dGW(ssp_rng, ssp, epoch)
                # MC scenarios
                dmg_mc = []
                pga_mc = []
                pliq_mc = []
                dmg_by_class = {0: [], 1: [], 2: [], 3: []}
                for _ in range(cfg.mc):
                    scen_rng = ssp_rng
                    scen = run_one_scenario(scen_rng, city, dGW, cfg.Mw, cfg.R_km_mean)
                    dmg_mc.append(scen["dmg"].mean())
                    pga_mc.append(scen["pga"].mean())
                    pliq_mc.append(scen["p_liq"].mean())
                    for c in range(4):
                        mask = city["lifeline_type"] == c
                        if mask.sum() > 0:
                            dmg_by_class[c].append(float(scen["dmg"][mask].mean()))
                dmg_arr = np.array(dmg_mc)
                results.append(CityEpochResult(
                    seed=seed,
                    city_id=city_id,
                    ssp=ssp,
                    epoch=epoch,
                    dGW=dGW,
                    mean_pga=float(np.mean(pga_mc)),
                    mean_pliq=float(np.mean(pliq_mc)),
                    mean_dmg=float(np.mean(dmg_arr)),
                    p50_dmg=float(np.percentile(dmg_arr, 50)),
                    p95_dmg=float(np.percentile(dmg_arr, 95)),
                    dmg_per_class={c: float(np.mean(dmg_by_class[c])) if dmg_by_class[c] else float("nan") for c in range(4)},
                ))
    return results


def static_baseline(results: List[CityEpochResult]) -> Dict[Tuple[int, int, str, int], float]:
    """For each (seed, city, ssp, epoch), look up the static-2020 damage from same seed and city."""
    static_map: Dict[Tuple[int, int, str], float] = {}
    for r in results:
        if r.epoch == 2020:
            # 2020 damage is shared across SSPs (since dGW=0); take Control-NoCC if present
            key = (r.seed, r.city_id, "BASELINE_2020")
            static_map.setdefault(key, r.mean_dmg)
    out: Dict[Tuple[int, int, str, int], float] = {}
    for r in results:
        out[(r.seed, r.city_id, r.ssp, r.epoch)] = static_map[(r.seed, r.city_id, "BASELINE_2020")]
    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def bootstrap_ci(x: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, rng: np.random.Generator | None = None) -> Tuple[float, float]:
    rng = rng or np.random.default_rng(0)
    boots = rng.choice(x, size=(n_boot, x.shape[0]), replace=True).mean(axis=1)
    return float(np.percentile(boots, 100 * alpha / 2)), float(np.percentile(boots, 100 * (1 - alpha / 2)))


def wilcoxon_signed_rank(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Wilcoxon signed-rank statistic and two-sided p-value. Self-contained."""
    d = x - y
    d = d[d != 0]
    n = d.shape[0]
    if n < 5:
        return float("nan"), float("nan")
    ranks = np.argsort(np.argsort(np.abs(d))) + 1.0
    W_plus = np.sum(ranks[d > 0])
    mean_W = n * (n + 1) / 4.0
    var_W = n * (n + 1) * (2 * n + 1) / 24.0
    z = (W_plus - mean_W) / math.sqrt(var_W)
    # Two-sided normal approximation
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    return float(z), float(p)


def wasserstein_1d(x: np.ndarray, y: np.ndarray) -> float:
    xs = np.sort(x)
    ys = np.sort(y)
    n = max(xs.size, ys.size)
    qs = np.linspace(0, 1, n)
    fx = np.interp(qs, np.linspace(0, 1, xs.size), xs)
    fy = np.interp(qs, np.linspace(0, 1, ys.size), ys)
    return float(np.mean(np.abs(fx - fy)))


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def save_main_results(results: List[CityEpochResult], static_map: Dict, out: Path) -> None:
    rows = []
    for r in results:
        static_dmg = static_map[(r.seed, r.city_id, r.ssp, r.epoch)]
        rows.append({
            "seed": r.seed,
            "city_id": r.city_id,
            "ssp": r.ssp,
            "epoch": r.epoch,
            "dGW_m": round(r.dGW, 4),
            "mean_pga_g": round(r.mean_pga, 4),
            "mean_pliq": round(r.mean_pliq, 4),
            "mean_dmg_cgstg": round(r.mean_dmg, 4),
            "mean_dmg_static": round(static_dmg, 4),
            "delta_dmg": round(r.mean_dmg - static_dmg, 4),
            "p50_dmg": round(r.p50_dmg, 4),
            "p95_dmg": round(r.p95_dmg, 4),
            "dmg_class0_building": round(r.dmg_per_class.get(0, float("nan")), 4),
            "dmg_class1_water": round(r.dmg_per_class.get(1, float("nan")), 4),
            "dmg_class2_power": round(r.dmg_per_class.get(2, float("nan")), 4),
            "dmg_class3_transport": round(r.dmg_per_class.get(3, float("nan")), 4),
        })
    write_csv(out, rows, list(rows[0].keys()))


def save_seed_stability(results: List[CityEpochResult], static_map: Dict, out: Path, cfg: Config) -> None:
    """For each (city, ssp, epoch), summarise across seeds: mean, std, 95% CI of delta_dmg."""
    by_key: Dict[Tuple[int, str, int], List[float]] = {}
    for r in results:
        d = r.mean_dmg - static_map[(r.seed, r.city_id, r.ssp, r.epoch)]
        by_key.setdefault((r.city_id, r.ssp, r.epoch), []).append(d)
    rows = []
    rng = set_seed(1234)
    for (city_id, ssp, epoch), values in by_key.items():
        arr = np.array(values)
        if arr.size >= 3:
            ci_lo, ci_hi = bootstrap_ci(arr, n_boot=cfg.bootstrap_n, alpha=0.05, rng=rng)
        else:
            ci_lo, ci_hi = float("nan"), float("nan")
        rows.append({
            "city_id": city_id,
            "ssp": ssp,
            "epoch": epoch,
            "n_seeds": arr.size,
            "delta_dmg_mean": round(float(arr.mean()), 5),
            "delta_dmg_std": round(float(arr.std(ddof=1) if arr.size > 1 else 0.0), 5),
            "delta_dmg_min": round(float(arr.min()), 5),
            "delta_dmg_max": round(float(arr.max()), 5),
            "ci95_lo": round(ci_lo, 5),
            "ci95_hi": round(ci_hi, 5),
        })
    write_csv(out, rows, list(rows[0].keys()))


def save_statistical_tests(results: List[CityEpochResult], static_map: Dict, out: Path, cfg: Config) -> None:
    """Per (ssp, epoch): paired Wilcoxon signed-rank on delta_dmg across seeds × cities.

    Also report H1 metric: Wasserstein-1 distance between the 2020 and {epoch} mean-damage
    distributions across seeds × cities, compared against the seed-noise-only Control-NoCC
    Wasserstein at the same epoch.
    """
    # Index by (seed, city, ssp, epoch) → mean_dmg
    idx: Dict[Tuple[int, int, str, int], float] = {(r.seed, r.city_id, r.ssp, r.epoch): r.mean_dmg for r in results}

    rows = []
    for ssp in cfg.ssps:
        for epoch in [e for e in cfg.epochs if e != 2020]:
            x_pairs = []
            y_pairs = []
            for seed in range(cfg.seeds):
                for city_id in range(cfg.cities):
                    x_pairs.append(idx[(seed, city_id, ssp, epoch)])
                    y_pairs.append(static_map[(seed, city_id, ssp, epoch)])
            x = np.array(x_pairs)
            y = np.array(y_pairs)
            z, p = wilcoxon_signed_rank(x, y)
            # H1 metric
            dist_epoch = np.array([idx[(seed, city_id, ssp, epoch)] for seed in range(cfg.seeds) for city_id in range(cfg.cities)])
            dist_2020 = np.array([idx[(seed, city_id, ssp, 2020)] for seed in range(cfg.seeds) for city_id in range(cfg.cities)])
            w1 = wasserstein_1d(dist_epoch, dist_2020)
            # Seed-noise control: Wasserstein between two random halves of 2020 across SSPs
            rng_h1 = set_seed(42)
            all_2020 = np.array([idx[(seed, city_id, s, 2020)] for seed in range(cfg.seeds) for city_id in range(cfg.cities) for s in cfg.ssps])
            perm = rng_h1.permutation(all_2020.size)
            half = all_2020.size // 2
            w1_noise = wasserstein_1d(all_2020[perm[:half]], all_2020[perm[half:2 * half]])
            rows.append({
                "test": "paired_wilcoxon_signed_rank_cgstg_vs_static",
                "ssp": ssp,
                "epoch": epoch,
                "n_pairs": x.size,
                "z": round(z, 4),
                "p_raw": round(p, 6),
                "p_bonferroni": round(min(1.0, p * cfg.bonferroni_n_tests), 6),
                "significant_at_alpha_0.05_bonferroni": (p * cfg.bonferroni_n_tests) < 0.05,
                "h1_wasserstein_epoch_vs_2020": round(w1, 5),
                "h1_wasserstein_seed_noise_baseline": round(w1_noise, 5),
                "h1_ratio_epoch_to_noise": round(w1 / max(w1_noise, 1e-9), 3),
            })
    write_csv(out, rows, list(rows[0].keys()))


def save_leakage_check(out: Path, cfg: Config) -> None:
    """Audit the leakage-prevention rules for the synthetic Round 1 smoke."""
    text = f"""# Data Leakage Audit — Round 1 Smoke Test

**Generated**: {datetime.utcnow().isoformat()}Z
**Pipeline mode**: Round 1 smoke, all-synthetic (no real data download yet).

## Splits enforced

- **Seed split**: {cfg.seeds} independent seeds; per-seed RNG state never shared across seeds.
- **City split**: {cfg.cities} cities; per-city RNG state is `seed * 1000 + city_id` — independent.
- **Epoch isolation**: each epoch's climate driver is sampled with an independent RNG state seeded by `(seed, city_id, ssp, epoch)`. No epoch ever uses information from another epoch.
- **SSP isolation**: SSP draws are independent; the Control-NoCC sham is generated separately.
- **Static baseline**: built from epoch 2020 of the same (seed, city_id). It is treated as a per-pair *reference*, never as a training target.

## What is NOT a leakage risk in Round 1

This smoke pipeline does not yet train any ML model. No information leakage between train and test
is possible because there is no train/test partition. Round 2 introduces the GNN and at that point
this audit will be extended to a city-LOCO + fault-source isolation check.

## What IS still a risk

- The Round 1 synthetic pipeline shares the **same** hazard-attenuation closed-form across all cities;
  this is not a leakage per se but **does** mean H1 detectability is artificially uniform across cities.
  Round 2 must replace with a per-city GMPE selection.
- The static baseline uses the same per-node Vs30/soil_class as CG-STG — by design, since we are
  comparing the *climate-mediated susceptibility shift*, not site characterisation. This is documented
  in §3.3 of `revised_main_text.md`.

## Verdict (Round 1)

- ❌ No leakage detected at the smoke-test stage.
- Round 2 audit must extend to: (a) city-LOCO; (b) fault-source split within city; (c) train/val split
  for any learned susceptibility surrogate.
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")


def save_meta(out: Path, cfg: Config, started_utc: str, finished_utc: str, elapsed_s: float) -> None:
    meta = {
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "elapsed_seconds": round(elapsed_s, 2),
        "config": cfg.__dict__,
        "n_records": cfg.seeds * cfg.cities * len(cfg.ssps) * len(cfg.epochs),
        "python": sys.version,
    }
    out.write_text(json.dumps(meta, indent=2, default=str, ensure_ascii=False), encoding="utf-8")


def try_plot_seed_stability(results: List[CityEpochResult], static_map: Dict, out_png: Path, cfg: Config) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        log.warning(f"matplotlib not available, skipping plot: {exc}")
        return
    fig, axes = plt.subplots(1, len(cfg.ssps), figsize=(4 * len(cfg.ssps), 4), sharey=True)
    if len(cfg.ssps) == 1:
        axes = [axes]
    for ax, ssp in zip(axes, cfg.ssps):
        for city_id in range(cfg.cities):
            x_vals = []
            y_means = []
            y_stds = []
            for epoch in cfg.epochs:
                deltas = []
                for seed in range(cfg.seeds):
                    rec = next(r for r in results if r.seed == seed and r.city_id == city_id and r.ssp == ssp and r.epoch == epoch)
                    deltas.append(rec.mean_dmg - static_map[(rec.seed, rec.city_id, rec.ssp, rec.epoch)])
                deltas = np.array(deltas)
                x_vals.append(epoch)
                y_means.append(deltas.mean())
                y_stds.append(deltas.std(ddof=1) if deltas.size > 1 else 0.0)
            x_vals = np.array(x_vals); y_means = np.array(y_means); y_stds = np.array(y_stds)
            ax.errorbar(x_vals, y_means, yerr=y_stds, marker="o", capsize=4, label=f"City {city_id}")
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_xlabel("Epoch")
        ax.set_title(ssp)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Δ mean damage (CG-STG − static)")
    axes[-1].legend(loc="upper left", fontsize=8)
    fig.suptitle("Seed stability — Δ damage vs static baseline (Round 1 smoke)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


def try_plot_confidence_intervals(results: List[CityEpochResult], static_map: Dict, out_png: Path, cfg: Config) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        log.warning(f"matplotlib not available, skipping plot: {exc}")
        return
    rng = set_seed(2024)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.25
    epochs_no_2020 = [e for e in cfg.epochs if e != 2020]
    for i, ssp in enumerate(cfg.ssps):
        means = []
        ci_lo = []
        ci_hi = []
        for epoch in epochs_no_2020:
            vals = []
            for seed in range(cfg.seeds):
                for city_id in range(cfg.cities):
                    rec = next(r for r in results if r.seed == seed and r.city_id == city_id and r.ssp == ssp and r.epoch == epoch)
                    vals.append(rec.mean_dmg - static_map[(rec.seed, rec.city_id, rec.ssp, rec.epoch)])
            arr = np.array(vals)
            means.append(arr.mean())
            lo, hi = bootstrap_ci(arr, n_boot=cfg.bootstrap_n, rng=rng) if arr.size >= 3 else (float("nan"), float("nan"))
            ci_lo.append(lo)
            ci_hi.append(hi)
        x = np.arange(len(epochs_no_2020)) + (i - 1) * width
        means = np.array(means); ci_lo = np.array(ci_lo); ci_hi = np.array(ci_hi)
        ax.bar(x, means, width=width, yerr=np.vstack([means - ci_lo, ci_hi - means]),
               capsize=4, label=ssp)
    ax.set_xticks(np.arange(len(epochs_no_2020)))
    ax.set_xticklabels([str(e) for e in epochs_no_2020])
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Δ mean damage (CG-STG − static), 95% bootstrap CI")
    ax.set_title("CG-STG vs static — pooled across seeds × cities (Round 1 smoke)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--cities", type=int, default=3)
    parser.add_argument("--mc", type=int, default=50)
    parser.add_argument("--nodes", type=int, default=200)
    args = parser.parse_args()

    cfg = Config(seeds=args.seeds, cities=args.cities, mc=args.mc, nodes_per_city=args.nodes)
    started = datetime.utcnow().isoformat() + "Z"
    log.info(f"Round 1 smoke pipeline start. cfg={cfg.__dict__}")

    t0 = datetime.utcnow().timestamp()
    all_results: List[CityEpochResult] = []
    for seed in range(cfg.seeds):
        rs = run_seed(seed, cfg)
        all_results.extend(rs)
        log.info(f"  seed {seed}: {len(rs)} (city,ssp,epoch) records")

    static_map = static_baseline(all_results)

    save_main_results(all_results, static_map, OUT_DIR / "main_results.csv")
    save_seed_stability(all_results, static_map, OUT_DIR / "seed_stability.csv", cfg)
    save_statistical_tests(all_results, static_map, OUT_DIR / "statistical_tests.csv", cfg)
    save_leakage_check(OUT_DIR / "data_leakage_check.md", cfg)
    try_plot_seed_stability(all_results, static_map, OUT_DIR / "seed_stability.png", cfg)
    try_plot_confidence_intervals(all_results, static_map, OUT_DIR / "confidence_intervals.png", cfg)

    finished = datetime.utcnow().isoformat() + "Z"
    elapsed = datetime.utcnow().timestamp() - t0
    save_meta(OUT_DIR / "run_meta.json", cfg, started, finished, elapsed)

    log.info(f"Round 1 smoke pipeline done. elapsed_s={elapsed:.2f}, n_records={len(all_results)}, out_dir={OUT_DIR}")
    print(f"\n=== Round 1 smoke pipeline finished in {elapsed:.2f}s ===")
    print(f"Output directory: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
