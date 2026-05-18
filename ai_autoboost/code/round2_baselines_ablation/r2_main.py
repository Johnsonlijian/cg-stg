"""Round 2 — 8-city cohort + BSSA14 + Boulanger-Idriss + fragility ensemble +
physics-anchored cascading + Mw×R sweep + B0–B4 baselines + A1–A5 ablations
+ BCa bootstrap confidence intervals.

GNN learning is in r2_gnn.py (separate); this script produces the physics-anchored
"target" against which the GNN is trained, AND establishes the headline statistical
result independently of any learned model.

CLI:
    python r2_main.py --seeds 10 --mc 30
    python r2_main.py --seeds 3 --mc 5 --smoke   # 30-second sanity
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

CODE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_ROOT.parents[2]
OUT_DIR = PROJECT_ROOT / "ai_autoboost" / "outputs" / "round2"
LOG_DIR = PROJECT_ROOT / "ai_autoboost" / "logs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / f"r2_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.log",
                            encoding="utf-8"),
    ],
)
log = logging.getLogger("r2")

sys.path.insert(0, str(CODE_ROOT))
import r2_lib as L


# ---------------------------------------------------------------------------
# 8-city cohort
# ---------------------------------------------------------------------------

ARCHETYPES = [
    ("deltaic",   "Tianjin-like"),       # GW ~ 2 m
    ("coastal",   "Bangkok-like"),       # GW ~ 4 m
    ("lowland",   "Jakarta-like"),       # GW ~ 7 m
    ("mixed",     "Beijing-like"),       # GW ~ 11 m
    ("inland",    "Tangshan-like"),      # GW ~ 16 m
    ("arid",      "Hohhot-like"),        # GW ~ 28 m
    ("cold",      "Anchorage-like"),     # GW ~ 9 m, permafrost
    ("high_alt",  "Mexico-City-like"),   # GW ~ 7 m, altitude
]


def build_cohort(rng: np.random.Generator, n_nodes: int = 200) -> List[L.CityGraph]:
    """Build the 8-city cohort. Each city uses the same n_nodes for fair pooling.

    NOTE: per ASSUMPTIONS.md A02 and A09, these are *archetypes*, not specific real
    cities — the "Tianjin-like" label denotes a deltaic archetype calibrated to
    public ground-water profiles for that class, NOT a literal Tianjin model.
    R3+ will optionally tighten with osmnx-loaded street/utility topologies.
    """
    cohort = []
    for i, (arch, label) in enumerate(ARCHETYPES):
        cg = L.synthesize_city_graph(np.random.default_rng(rng.integers(0, 1 << 31)),
                                      n_nodes=n_nodes, archetype=arch)
        cg.archetype = arch
        # we attach the label as an attribute by replacing the dataclass via a workaround:
        object.__setattr__(cg, "label", label)
        object.__setattr__(cg, "city_id", i)
        cohort.append(cg)
    return cohort


# ---------------------------------------------------------------------------
# Climate driver dGW (same form as R1, but per-archetype calibrated)
# ---------------------------------------------------------------------------

def sample_dGW(rng: np.random.Generator, ssp: str, epoch: int, archetype: str) -> float:
    if epoch == 2020:
        return 0.0
    base = {
        "SSP2-4.5": {2050: -0.25, 2100: -0.50},
        "SSP5-8.5": {2050: -0.50, 2100: -1.00},
        "Control-NoCC": {2050: 0.0, 2100: 0.0},
    }[ssp][epoch]
    sd = 0.20 if ssp == "SSP5-8.5" else (0.15 if ssp == "SSP2-4.5" else 0.10)
    # Archetype-specific amplification (deltaic cities are more sensitive to GW change)
    arch_amp = {
        "deltaic":  1.4,
        "coastal":  1.2,
        "lowland":  1.0,
        "mixed":    0.85,
        "inland":   0.70,
        "arid":     0.40,
        "cold":     1.10,
        "high_alt": 0.95,
    }[archetype]
    return float(rng.normal(base, sd) * arch_amp)


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------

@dataclass
class Config:
    seeds: int = 10
    mc: int = 30
    n_nodes: int = 200
    Mw_list: Tuple[float, ...] = (5.5, 6.0, 6.5, 7.0, 7.5)
    R_list_km: Tuple[float, ...] = (10.0, 25.0, 50.0, 100.0)
    main_Mw: float = 6.5
    main_R: float = 25.0
    epochs: Tuple[int, ...] = (2020, 2050, 2100)
    ssps: Tuple[str, ...] = ("SSP2-4.5", "SSP5-8.5", "Control-NoCC")
    cascading_steps: int = 8
    cascading_kappa: float = 0.15
    bootstrap_n: int = 4999


@dataclass
class MethodResult:
    seed: int
    city_id: int
    archetype: str
    label: str
    ssp: str
    epoch: int
    Mw: float
    R_km: float
    dGW: float
    method: str
    mean_dmg_initial: float       # before cascading
    mean_dmg_final: float         # after cascading
    p_liq_mean: float
    pga_mean_g: float
    epistemic_std_initial: float  # fragility-ensemble std
    dmg_by_class: Dict[int, float]


# ---------------------------------------------------------------------------
# Method definitions: B0–B4 + A1–A5
# ---------------------------------------------------------------------------

METHODS = {
    # B0: completely static, no climate, no cascading, single fragility (HAZUS only)
    "B0_static_hazus":
        dict(use_climate=False, use_ensemble=False, use_cascading=False, kappa=0.0, family=L.HAZUS),
    # B1: time-varying climate, no cascading, single fragility
    "B1_climate_hazus_no_cascade":
        dict(use_climate=True, use_ensemble=False, use_cascading=False, kappa=0.0, family=L.HAZUS),
    # B2: static, with cascading, single fragility
    "B2_static_cascade":
        dict(use_climate=False, use_ensemble=False, use_cascading=True, kappa=0.15, family=L.HAZUS),
    # B3: climate + cascading + single fragility (= "CG-STG no-ensemble")
    "B3_cgstg_hazus_only":
        dict(use_climate=True, use_ensemble=False, use_cascading=True, kappa=0.15, family=L.HAZUS),
    # B4: climate + cascading + fragility ensemble (= full physics-anchored CG-STG)
    "B4_cgstg_full":
        dict(use_climate=True, use_ensemble=True, use_cascading=True, kappa=0.15, family=None),
    # A1: ablate fragility ensemble (use GEM-like solo)
    "A1_cgstg_gem_only":
        dict(use_climate=True, use_ensemble=False, use_cascading=True, kappa=0.15, family=L.GEMlike),
    # A2: ablate cascading (kappa=0)
    "A2_cgstg_no_cascade":
        dict(use_climate=True, use_ensemble=True, use_cascading=False, kappa=0.0, family=None),
    # A3: ablate climate driver
    "A3_cgstg_no_climate":
        dict(use_climate=False, use_ensemble=True, use_cascading=True, kappa=0.15, family=None),
    # A4: cascading kappa halved
    "A4_cgstg_low_kappa":
        dict(use_climate=True, use_ensemble=True, use_cascading=True, kappa=0.075, family=None),
    # A5: cascading kappa doubled
    "A5_cgstg_high_kappa":
        dict(use_climate=True, use_ensemble=True, use_cascading=True, kappa=0.30, family=None),
}


# ---------------------------------------------------------------------------
# One-scenario evaluation
# ---------------------------------------------------------------------------

def run_one(graph: L.CityGraph, Mw: float, R_km: float, dGW_applied: float,
            method_cfg: dict, rng: np.random.Generator) -> Tuple[float, float, float, float, float, Dict[int, float]]:
    """Run a single scenario for one method on one city graph.

    Returns: (initial_mean, final_mean, p_liq_mean, pga_mean, epistemic_std, dmg_by_class)
    """
    # Hazard via BSSA14 — vectorized over nodes
    R_vec = np.full(graph.n_nodes, R_km, dtype=float)
    pga = L.bssa14_pga(Mw, R_vec, graph.Vs30, fault="SS")
    # Sample ground-motion sigma to add aleatory variability
    sigma_ln = 0.72
    pga = pga * np.exp(rng.normal(0.0, sigma_ln, size=graph.n_nodes))

    # Site state (climate-modulated if method enables it)
    if method_cfg["use_climate"]:
        GW_t = np.clip(graph.GW_2020 + dGW_applied, 0.3, None)
    else:
        GW_t = graph.GW_2020.copy()

    # Liquefaction probability per node
    p_liq = L.liquefaction_probability(Mw, pga, graph.Vs30, GW_t, depth_m=3.0)

    # Damage initial (fragility)
    if method_cfg["use_ensemble"]:
        dmg_init_mean, dmg_init_std, _stacked = L.damage_ensemble(pga, graph.asset_class, p_liq)
    else:
        dmg_init_mean = L.damage_state_probability(pga, graph.asset_class, method_cfg["family"], p_liq)
        dmg_init_std = np.zeros_like(dmg_init_mean)

    # Cascading
    if method_cfg["use_cascading"]:
        kappa_method = method_cfg["kappa"]
        d_final, _ = L.physics_cascading(dmg_init_mean, graph,
                                          n_steps=8,
                                          transmission_kappa=kappa_method,
                                          recovery_threshold=0.10)
    else:
        d_final = dmg_init_mean

    # Per-class aggregates
    dmg_by_class = {}
    for c in range(4):
        mask = graph.asset_class == c
        dmg_by_class[c] = float(d_final[mask].mean()) if mask.sum() > 0 else float("nan")

    return (float(dmg_init_mean.mean()), float(d_final.mean()),
            float(p_liq.mean()), float(pga.mean()),
            float(dmg_init_std.mean()), dmg_by_class)


# ---------------------------------------------------------------------------
# Main loops
# ---------------------------------------------------------------------------

def run_main_methods_grid(cohort: List[L.CityGraph], cfg: Config) -> List[MethodResult]:
    """For each method, run at the main config (Mw=6.5, R=25 km) across seeds × cities × ssp × epoch × MC."""
    results: List[MethodResult] = []
    for seed in range(cfg.seeds):
        rng_seed = np.random.default_rng(seed)
        for city in cohort:
            for ssp in cfg.ssps:
                for epoch in cfg.epochs:
                    ssp_rng = np.random.default_rng(seed * 10_000 + city.city_id * 1000 + hash(ssp) % 991 + epoch)
                    # MC over dGW draws + ground-motion aleatory
                    for mc in range(cfg.mc):
                        dGW = sample_dGW(ssp_rng, ssp, epoch, city.archetype)
                        for method_name, method_cfg in METHODS.items():
                            res = run_one(city, cfg.main_Mw, cfg.main_R, dGW, method_cfg, ssp_rng)
                            results.append(MethodResult(
                                seed=seed, city_id=city.city_id, archetype=city.archetype,
                                label=city.label, ssp=ssp, epoch=epoch,
                                Mw=cfg.main_Mw, R_km=cfg.main_R, dGW=dGW,
                                method=method_name,
                                mean_dmg_initial=res[0], mean_dmg_final=res[1],
                                p_liq_mean=res[2], pga_mean_g=res[3],
                                epistemic_std_initial=res[4],
                                dmg_by_class=res[5],
                            ))
        log.info(f"  main grid: seed {seed} done")
    return results


def run_mwr_sweep(cohort: List[L.CityGraph], cfg: Config,
                  method_name: str = "B4_cgstg_full") -> List[MethodResult]:
    """Mw × R sweep on the headline method only."""
    method_cfg = METHODS[method_name]
    results: List[MethodResult] = []
    for seed in range(cfg.seeds):
        for city in cohort:
            for Mw in cfg.Mw_list:
                for R_km in cfg.R_list_km:
                    for ssp in cfg.ssps:
                        for epoch in cfg.epochs:
                            ssp_rng = np.random.default_rng(seed * 100_000 + city.city_id * 10_000 +
                                                            hash(ssp) % 9973 + int(Mw * 10) + int(R_km))
                            # Smaller MC for sweep to keep runtime bounded
                            for mc in range(max(5, cfg.mc // 3)):
                                dGW = sample_dGW(ssp_rng, ssp, epoch, city.archetype)
                                res = run_one(city, Mw, R_km, dGW, method_cfg, ssp_rng)
                                results.append(MethodResult(
                                    seed=seed, city_id=city.city_id, archetype=city.archetype,
                                    label=city.label, ssp=ssp, epoch=epoch,
                                    Mw=Mw, R_km=R_km, dGW=dGW,
                                    method=method_name,
                                    mean_dmg_initial=res[0], mean_dmg_final=res[1],
                                    p_liq_mean=res[2], pga_mean_g=res[3],
                                    epistemic_std_initial=res[4],
                                    dmg_by_class=res[5],
                                ))
        log.info(f"  Mw×R sweep: seed {seed} done")
    return results


# ---------------------------------------------------------------------------
# Persistence + Stats
# ---------------------------------------------------------------------------

def flatten_to_rows(results: List[MethodResult]) -> List[Dict]:
    rows = []
    for r in results:
        d = asdict(r)
        for c in range(4):
            d[f"dmg_class{c}"] = d["dmg_by_class"].get(c, float("nan"))
        del d["dmg_by_class"]
        for k, v in list(d.items()):
            if isinstance(v, float):
                d[k] = round(v, 6)
        rows.append(d)
    return rows


def write_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        log.warning(f"empty rows for {path}")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def compute_baseline_comparison(results: List[MethodResult], cfg: Config, out: Path) -> None:
    """For each method, at SSP5-8.5 2100, pooled across seeds × cities × MC:
    mean damage, std, BCa 95% CI, paired Wilcoxon vs B0_static_hazus."""
    rng = np.random.default_rng(7)
    # Index by (method, ssp, epoch) → list of (seed, city, mc) damage_final
    rows = []
    for method in METHODS.keys():
        for ssp in cfg.ssps:
            for epoch in cfg.epochs:
                vals = [r.mean_dmg_final for r in results
                        if r.method == method and r.ssp == ssp and r.epoch == epoch]
                if not vals:
                    continue
                arr = np.array(vals)
                lo, hi = L.bca_ci(arr, n_resamples=cfg.bootstrap_n, rng_seed=hash((method, ssp, epoch)) & 0xFFFF)
                # Paired Wilcoxon vs B0 at same (ssp, epoch, seed, city, mc)
                if method != "B0_static_hazus":
                    pairs_a, pairs_b = [], []
                    base_dict = {(r.seed, r.city_id, r.dGW): r.mean_dmg_final for r in results
                                 if r.method == "B0_static_hazus" and r.ssp == ssp and r.epoch == epoch}
                    for r in results:
                        if r.method == method and r.ssp == ssp and r.epoch == epoch:
                            key = (r.seed, r.city_id, r.dGW)
                            if key in base_dict:
                                pairs_a.append(r.mean_dmg_final)
                                pairs_b.append(base_dict[key])
                    if len(pairs_a) > 5:
                        z, p = L.paired_wilcoxon(np.array(pairs_a), np.array(pairs_b))
                    else:
                        z, p = float("nan"), float("nan")
                else:
                    z, p = float("nan"), float("nan")
                rows.append({
                    "method": method, "ssp": ssp, "epoch": epoch,
                    "n": len(vals),
                    "mean_dmg_final": round(float(arr.mean()), 6),
                    "std_dmg_final": round(float(arr.std(ddof=1)) if arr.size > 1 else 0.0, 6),
                    "bca_ci_lo": round(lo, 6),
                    "bca_ci_hi": round(hi, 6),
                    "wilcoxon_vs_B0_z": round(z, 4) if not np.isnan(z) else "",
                    "wilcoxon_vs_B0_p": round(p, 6) if not np.isnan(p) else "",
                })
    write_csv(rows, out)


def compute_ablation_table(results: List[MethodResult], cfg: Config, out: Path) -> None:
    """For each ablation, report the *gap* (CG-STG full − ablated) at SSP5-8.5 2100."""
    rows = []
    full_dict = {(r.seed, r.city_id, r.dGW): r.mean_dmg_final for r in results
                 if r.method == "B4_cgstg_full" and r.ssp == "SSP5-8.5" and r.epoch == 2100}
    for method in METHODS.keys():
        if method == "B4_cgstg_full":
            continue
        gaps = []
        for r in results:
            if r.method == method and r.ssp == "SSP5-8.5" and r.epoch == 2100:
                key = (r.seed, r.city_id, r.dGW)
                if key in full_dict:
                    gaps.append(full_dict[key] - r.mean_dmg_final)
        if not gaps:
            continue
        arr = np.array(gaps)
        lo, hi = L.bca_ci(arr, n_resamples=cfg.bootstrap_n, rng_seed=hash(method) & 0xFFFF)
        rows.append({
            "ablated_or_baseline": method,
            "n": len(gaps),
            "mean_gap_full_minus_method": round(float(arr.mean()), 6),
            "std_gap": round(float(arr.std(ddof=1)) if arr.size > 1 else 0.0, 6),
            "bca_ci_lo": round(lo, 6),
            "bca_ci_hi": round(hi, 6),
            "interpretation": (
                "B4_cgstg_full ABOVE method" if arr.mean() > 0
                else "B4_cgstg_full BELOW method"
            ),
        })
    write_csv(rows, out)


def compute_mwr_summary(results: List[MethodResult], cfg: Config, out: Path) -> None:
    """Pooled mean damage at (Mw, R, epoch=2100, SSP5-8.5)."""
    rows = []
    for Mw in cfg.Mw_list:
        for R in cfg.R_list_km:
            vals = [r.mean_dmg_final for r in results
                    if r.Mw == Mw and r.R_km == R and r.ssp == "SSP5-8.5" and r.epoch == 2100]
            if not vals:
                continue
            arr = np.array(vals)
            lo, hi = L.bca_ci(arr, n_resamples=2999, rng_seed=int(Mw * 100 + R))
            rows.append({
                "Mw": Mw, "R_km": R, "n": len(vals),
                "mean_dmg_final": round(float(arr.mean()), 6),
                "bca_ci_lo": round(lo, 6),
                "bca_ci_hi": round(hi, 6),
            })
    write_csv(rows, out)


def per_city_climate_gap(results: List[MethodResult], cfg: Config, out: Path,
                          method: str = "B4_cgstg_full", ssp: str = "SSP5-8.5") -> None:
    """Per-city *climate-isolated* gap: B4 at epoch 2100 vs B4 at epoch 2020 SSP5-8.5.

    Both runs use the same model, same fragility ensemble, same cascading; the only
    difference is the climate-driven dGW at the given epoch. This isolates the
    climate signal from the cascade saturation.
    """
    rows = []
    for city_id in range(len(ARCHETYPES)):
        a2100 = {}
        a2020 = {}
        for r in results:
            if r.method == method and r.city_id == city_id and r.ssp == ssp:
                if r.epoch == 2100:
                    a2100.setdefault(r.seed, []).append(r.mean_dmg_final)
                elif r.epoch == 2020:
                    a2020.setdefault(r.seed, []).append(r.mean_dmg_final)
        gaps = []
        for seed in a2100:
            if seed in a2020:
                gaps.append(np.mean(a2100[seed]) - np.mean(a2020[seed]))
        gaps = np.array(gaps)
        if gaps.size < 3:
            continue
        lo, hi = L.bca_ci(gaps, n_resamples=2999, rng_seed=city_id + 100)
        arch, label = ARCHETYPES[city_id]
        rows.append({
            "city_id": city_id, "archetype": arch, "label": label,
            "method": method, "ssp": ssp,
            "n_seeds": gaps.size,
            "mean_climate_gap_2100_minus_2020": round(float(gaps.mean()), 6),
            "std_gap": round(float(gaps.std(ddof=1)) if gaps.size > 1 else 0.0, 6),
            "bca_ci_lo": round(lo, 6),
            "bca_ci_hi": round(hi, 6),
        })
    write_csv(rows, out)


def per_city_static_gap(results: List[MethodResult], cfg: Config, out: Path) -> None:
    """Per-city CG-STG (B4) − Static (B0) gap at SSP5-8.5 2100.

    The headline H3 spatial-structure table.
    """
    rows = []
    for city_id in range(len(ARCHETYPES)):
        full = [r.mean_dmg_final for r in results
                if r.method == "B4_cgstg_full" and r.city_id == city_id
                and r.ssp == "SSP5-8.5" and r.epoch == 2100]
        static = [r.mean_dmg_final for r in results
                  if r.method == "B0_static_hazus" and r.city_id == city_id
                  and r.ssp == "SSP5-8.5" and r.epoch == 2100]
        if not full or not static:
            continue
        # Paired by (seed, dGW)? B0 doesn't use dGW; pair by seed only
        full_d = {}
        static_d = {}
        for r in results:
            if r.method == "B4_cgstg_full" and r.city_id == city_id and r.ssp == "SSP5-8.5" and r.epoch == 2100:
                full_d.setdefault(r.seed, []).append(r.mean_dmg_final)
            if r.method == "B0_static_hazus" and r.city_id == city_id and r.ssp == "SSP5-8.5" and r.epoch == 2100:
                static_d.setdefault(r.seed, []).append(r.mean_dmg_final)
        gaps = []
        for seed in full_d:
            if seed in static_d:
                gaps.append(np.mean(full_d[seed]) - np.mean(static_d[seed]))
        gaps = np.array(gaps)
        lo, hi = L.bca_ci(gaps, n_resamples=2999, rng_seed=city_id) if gaps.size >= 3 else (float("nan"), float("nan"))
        arch, label = ARCHETYPES[city_id]
        rows.append({
            "city_id": city_id, "archetype": arch, "label": label,
            "n_seeds": gaps.size,
            "mean_gap_cgstg_minus_static": round(float(gaps.mean()), 6),
            "std_gap": round(float(gaps.std(ddof=1)) if gaps.size > 1 else 0.0, 6),
            "bca_ci_lo": round(lo, 6),
            "bca_ci_hi": round(hi, 6),
        })
    write_csv(rows, out)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_baseline_comparison(out_png: Path, baseline_csv: Path, ssps_to_plot=("SSP5-8.5", "SSP2-4.5", "Control-NoCC")) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception as e:
        log.warning(f"plot skip: {e}")
        return
    df = pd.read_csv(baseline_csv)
    df = df[df["epoch"] == 2100]
    methods_order = list(METHODS.keys())
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.27
    x = np.arange(len(methods_order))
    for i, ssp in enumerate(ssps_to_plot):
        sub = df[df["ssp"] == ssp].set_index("method").reindex(methods_order)
        ax.bar(x + (i - 1) * width,
               sub["mean_dmg_final"].values,
               width=width,
               yerr=[sub["mean_dmg_final"] - sub["bca_ci_lo"],
                     sub["bca_ci_hi"] - sub["mean_dmg_final"]],
               capsize=3, label=ssp)
    ax.set_xticks(x)
    ax.set_xticklabels(methods_order, rotation=45, ha="right")
    ax.set_ylabel("Mean final lifeline damage @ 2100")
    ax.set_title("Round 2 — methods comparison at horizon 2100 (BCa 95% CI)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_mwr_heatmap(out_png: Path, mwr_csv: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception as e:
        log.warning(f"plot skip: {e}")
        return
    df = pd.read_csv(mwr_csv)
    piv = df.pivot(index="Mw", columns="R_km", values="mean_dmg_final")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    im = ax.imshow(piv.values, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(np.arange(piv.shape[1]))
    ax.set_xticklabels([f"{int(v)}" for v in piv.columns])
    ax.set_yticks(np.arange(piv.shape[0]))
    ax.set_yticklabels([f"{v:.1f}" for v in piv.index])
    ax.set_xlabel("R_JB (km)")
    ax.set_ylabel("Mw")
    ax.set_title("CG-STG mean final damage @ 2100 (SSP5-8.5)")
    fig.colorbar(im, ax=ax, label="Mean damage")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_per_city_gap(out_png: Path, per_city_csv: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception as e:
        log.warning(f"plot skip: {e}")
        return
    df = pd.read_csv(per_city_csv).sort_values("mean_gap_cgstg_minus_static", ascending=False)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    yerr = np.vstack([df["mean_gap_cgstg_minus_static"] - df["bca_ci_lo"],
                      df["bca_ci_hi"] - df["mean_gap_cgstg_minus_static"]])
    ax.bar(np.arange(len(df)), df["mean_gap_cgstg_minus_static"], yerr=yerr, capsize=4)
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels([f"{a}\n({l})" for a, l in zip(df["archetype"], df["label"])], rotation=25, ha="right")
    ax.set_ylabel("CG-STG − Static gap @ 2100 SSP5-8.5")
    ax.set_title("Per-city static-bias spatial structure (BCa 95% CI)")
    ax.axhline(0, color="grey", lw=0.5)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--mc", type=int, default=30)
    parser.add_argument("--nodes", type=int, default=200)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = Config(seeds=args.seeds, mc=args.mc, n_nodes=args.nodes)
    if args.smoke:
        cfg = Config(seeds=max(3, cfg.seeds // 3), mc=max(5, cfg.mc // 5), n_nodes=100)

    t0 = datetime.utcnow()
    log.info(f"R2 start with cfg={asdict(cfg)}")

    cohort_rng = np.random.default_rng(2026_05_14)
    cohort = build_cohort(cohort_rng, n_nodes=cfg.n_nodes)
    log.info(f"Built cohort of {len(cohort)} cities: {[c.archetype for c in cohort]}")

    # Main grid: all methods at main_Mw, main_R, full SSP × epoch × seed × city × MC
    log.info(f"Running main methods grid at Mw={cfg.main_Mw}, R={cfg.main_R} km ...")
    main_results = run_main_methods_grid(cohort, cfg)
    log.info(f"  → {len(main_results)} method-records")

    # Headline-only Mw × R sweep
    log.info("Running Mw × R sweep for B4_cgstg_full only ...")
    mwr_results = run_mwr_sweep(cohort, cfg, method_name="B4_cgstg_full")
    log.info(f"  → {len(mwr_results)} sweep records")

    # Persist raw
    write_csv(flatten_to_rows(main_results), OUT_DIR / "main_methods_raw.csv")
    write_csv(flatten_to_rows(mwr_results), OUT_DIR / "mwr_sweep_raw.csv")

    # Summary tables
    compute_baseline_comparison(main_results, cfg, OUT_DIR / "baseline_comparison.csv")
    compute_ablation_table(main_results, cfg, OUT_DIR / "ablation_results.csv")
    compute_mwr_summary(mwr_results, cfg, OUT_DIR / "mwr_summary.csv")
    per_city_static_gap(main_results, cfg, OUT_DIR / "per_city_gap.csv")
    per_city_climate_gap(main_results, cfg, OUT_DIR / "per_city_climate_gap.csv")

    # Plots
    plot_baseline_comparison(OUT_DIR / "baseline_comparison.png", OUT_DIR / "baseline_comparison.csv")
    plot_mwr_heatmap(OUT_DIR / "mwr_heatmap.png", OUT_DIR / "mwr_summary.csv")
    plot_per_city_gap(OUT_DIR / "per_city_gap.png", OUT_DIR / "per_city_gap.csv")
    # Climate-isolated plot reuses the per-city plotter on different column names
    try:
        import pandas as pd, matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        df = pd.read_csv(OUT_DIR / "per_city_climate_gap.csv").sort_values(
            "mean_climate_gap_2100_minus_2020", ascending=False)
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        yerr = np.vstack([df["mean_climate_gap_2100_minus_2020"] - df["bca_ci_lo"],
                          df["bca_ci_hi"] - df["mean_climate_gap_2100_minus_2020"]])
        ax.bar(np.arange(len(df)), df["mean_climate_gap_2100_minus_2020"], yerr=yerr, capsize=4)
        ax.set_xticks(np.arange(len(df)))
        ax.set_xticklabels([f"{a}\n({l})" for a, l in zip(df["archetype"], df["label"])],
                            rotation=25, ha="right")
        ax.set_ylabel("CG-STG: 2100 − 2020 (SSP5-8.5), same cascade")
        ax.set_title("Climate-isolated per-city damage gap (BCa 95% CI)")
        ax.axhline(0, color="grey", lw=0.5)
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "per_city_climate_gap.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        log.warning(f"climate-gap plot skip: {e}")

    elapsed = (datetime.utcnow() - t0).total_seconds()
    meta = {
        "started_utc": t0.isoformat() + "Z",
        "finished_utc": datetime.utcnow().isoformat() + "Z",
        "elapsed_seconds": round(elapsed, 2),
        "config": asdict(cfg),
        "n_main": len(main_results),
        "n_sweep": len(mwr_results),
        "methods": list(METHODS.keys()),
        "cohort": [(c.archetype, c.label) for c in cohort],
    }
    (OUT_DIR / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"R2 done in {elapsed:.1f}s. Outputs in {OUT_DIR}")
    print(f"\n=== R2 finished in {elapsed:.1f}s — {len(main_results)+len(mwr_results)} records ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
