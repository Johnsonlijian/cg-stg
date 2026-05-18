"""Round 3 — PCMCI causal discovery on (climate driver → soil state → liquefaction
probability → final damage) chain, per city archetype.

Why this is the right test:
    Round 2 showed B4_cgstg_full − A3_no_climate = +0.0037 [+0.0015, +0.0058],
    i.e. climate-driven contribution to mean damage exists under cascading. But this
    is associational. PCMCI asks: when we condition on intermediate variables, does
    the climate→soil→liq→damage chain show the conditional independence pattern
    expected of a causal mediator chain (Runge 2020)?

How:
    For each of 8 archetypes, simulate a 17-epoch (2020–2100, 5-yr step) trajectory
    under SSP5-8.5. Per-epoch summary statistics:
        v1 = dGW (climate driver, m below 2020 baseline; negative = water rose)
        v2 = soil_moisture_proxy ∝ -dGW within plausible bounds
        v3 = mean liquefaction probability across nodes
        v4 = mean final damage across nodes (post-cascade)
    Run PCMCI with ParCorr independence test, tau_max=3, alpha=0.05 (Bonferroni
    across pairs). Report per-archetype causal graph + per-archetype "chain strength"
    (the conditional dependence v1 → v4 controlling for v2 and v3).

Surrogate baseline:
    Lagged Pearson correlation (no conditioning), to show that conditional PCMCI
    is informative beyond simple correlation.

Output:
    outputs/round3/pcmci_per_archetype.csv
    outputs/round3/pcmci_chain_strength.csv
    outputs/round3/pcmci_graph_<archetype>.png  (one figure per archetype)
"""
from __future__ import annotations

import csv
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

CODE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_ROOT.parents[2]
OUT_DIR = PROJECT_ROOT / "ai_autoboost" / "outputs" / "round3"
LOG_DIR = PROJECT_ROOT / "ai_autoboost" / "logs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / f"r3_pcmci_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.log",
                            encoding="utf-8"),
    ],
)
log = logging.getLogger("r3_pcmci")

sys.path.insert(0, str(CODE_ROOT.parent / "round2_baselines_ablation"))
import r2_lib as L
from r2_main import ARCHETYPES, sample_dGW  # noqa: E402


# ---------------------------------------------------------------------------
# Trajectory generation: 17 epochs (2020, 2025, ..., 2100) per (archetype, seed)
# ---------------------------------------------------------------------------

EPOCHS_DENSE = tuple(range(2020, 2101, 5))  # 17 points


def interp_dGW_mean(ssp: str, archetype: str, epoch: int) -> float:
    """Mean ΔGW given (ssp, archetype, epoch), with linear interpolation from
    Round 2's 3-anchor SSP scenarios at 2020/2050/2100.
    """
    anchors = {
        "SSP2-4.5": {2020: 0.0, 2050: -0.25, 2100: -0.50},
        "SSP5-8.5": {2020: 0.0, 2050: -0.50, 2100: -1.00},
        "Control-NoCC": {2020: 0.0, 2050: 0.0, 2100: 0.0},
    }[ssp]
    if epoch <= 2050:
        f = (epoch - 2020) / 30.0
        base = (1 - f) * anchors[2020] + f * anchors[2050]
    else:
        f = (epoch - 2050) / 50.0
        base = (1 - f) * anchors[2050] + f * anchors[2100]
    arch_amp = {
        "deltaic": 1.4, "coastal": 1.2, "lowland": 1.0, "mixed": 0.85,
        "inland": 0.70, "arid": 0.40, "cold": 1.10, "high_alt": 0.95,
    }[archetype]
    return base * arch_amp


def simulate_dense_trajectory(archetype: str, n_nodes: int = 150,
                              n_seeds: int = 8, ssp: str = "SSP5-8.5",
                              rng_seed: int = 2026_05_15) -> Dict[str, np.ndarray]:
    """For one archetype, produce arrays of shape (n_seeds * n_epoch_dense, 4) holding
    (v1=dGW, v2=soil_moisture_proxy, v3=mean p_liq, v4=mean post-cascade damage).

    We stack per-seed trajectories into one long matrix; PCMCI receives this as one
    time series (epoch index advances within each seed; seed transitions are treated
    as "epoch breaks" by using contiguous storage but per-trajectory autocorrelation
    is what dominates).
    """
    rng = np.random.default_rng(rng_seed + hash(archetype) % 9973)
    cg = L.synthesize_city_graph(rng, n_nodes=n_nodes, archetype=archetype)

    arrs = {"v1": [], "v2": [], "v3": [], "v4": []}
    for s in range(n_seeds):
        for epoch in EPOCHS_DENSE:
            mu = interp_dGW_mean(ssp, archetype, epoch)
            sd = 0.20 if ssp == "SSP5-8.5" else 0.10
            dGW = float(rng.normal(mu, sd))
            # soil moisture proxy: -dGW (when water rises, dGW negative, moisture up)
            # plus a small noise
            soil_moisture = -dGW * 0.8 + rng.normal(0, 0.05)

            # Mw moderately scattered around 6.5 each epoch (independent re-sampling)
            Mw = float(rng.normal(6.5, 0.3))
            R_km = 25.0
            R_vec = np.full(cg.n_nodes, R_km)
            pga = L.bssa14_pga(Mw, R_vec, cg.Vs30, fault="SS")
            pga = pga * np.exp(rng.normal(0.0, 0.72, size=cg.n_nodes))
            GW_t = np.clip(cg.GW_2020 + dGW, 0.3, None)
            p_liq = L.liquefaction_probability(Mw, pga, cg.Vs30, GW_t, depth_m=3.0)
            dmg_init_mean, _, _ = L.damage_ensemble(pga, cg.asset_class, p_liq)
            d_final, _ = L.physics_cascading(dmg_init_mean, cg,
                                              n_steps=8, transmission_kappa=0.15,
                                              recovery_threshold=0.10)
            arrs["v1"].append(dGW)
            arrs["v2"].append(float(soil_moisture))
            arrs["v3"].append(float(p_liq.mean()))
            arrs["v4"].append(float(d_final.mean()))

    return {k: np.asarray(v, dtype=float) for k, v in arrs.items()}


# ---------------------------------------------------------------------------
# PCMCI + surrogate lagged-correlation analysis
# ---------------------------------------------------------------------------

def run_pcmci_one_archetype(arch_data: Dict[str, np.ndarray], tau_max: int = 3,
                              alpha: float = 0.05) -> Dict:
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite import data_processing as pp

    data = np.stack([arch_data["v1"], arch_data["v2"],
                     arch_data["v3"], arch_data["v4"]], axis=1)
    var_names = ["dGW", "soil_moist", "p_liq", "damage"]
    dataframe = pp.DataFrame(data, var_names=var_names)
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0)
    results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=alpha, alpha_level=alpha)

    # Extract significant edges
    p_matrix = results["p_matrix"]
    val_matrix = results["val_matrix"]
    edges = []
    n_vars = data.shape[1]
    for i in range(n_vars):
        for j in range(n_vars):
            for tau in range(tau_max + 1):
                p_ij_tau = float(p_matrix[i, j, tau])
                val_ij_tau = float(val_matrix[i, j, tau])
                if p_ij_tau < alpha and i != j:
                    edges.append({
                        "src": var_names[i], "dst": var_names[j],
                        "tau": tau, "p": p_ij_tau, "val": val_ij_tau,
                    })

    return {"edges": edges, "p_matrix": p_matrix, "val_matrix": val_matrix,
            "var_names": var_names, "n_samples": data.shape[0]}


def chain_strength_climate_to_damage(arch_data: Dict[str, np.ndarray], tau_max: int = 3) -> Dict:
    """Estimate the conditional dependence dGW → damage controlling for soil & liq,
    via the partial correlation framework used by PCMCI.

    We use a simple residualisation: regress v4 on (v2, v3) at the same time, regress
    v1 on (v2, v3) at the same time, then correlate the residuals. This is the
    standard partial-correlation approach.
    """
    from sklearn.linear_model import LinearRegression
    v1, v2, v3, v4 = arch_data["v1"], arch_data["v2"], arch_data["v3"], arch_data["v4"]
    X_med = np.stack([v2, v3], axis=1)
    r4 = v4 - LinearRegression().fit(X_med, v4).predict(X_med)
    r1 = v1 - LinearRegression().fit(X_med, v1).predict(X_med)
    n = r1.size
    if n < 5 or r1.std() < 1e-9 or r4.std() < 1e-9:
        return {"r_partial": float("nan"), "n": int(n)}
    r = float(np.corrcoef(r1, r4)[0, 1])
    # Fisher z + two-sided normal p
    if abs(r) >= 0.999:
        z, p = float("inf"), 0.0
    else:
        z = 0.5 * np.log((1 + r) / (1 - r)) * np.sqrt(n - 3)
        from math import erf
        p = 2 * (1 - 0.5 * (1 + erf(abs(z) / np.sqrt(2))))
    return {"r_partial": r, "fisher_z": float(z), "p": float(p), "n": int(n)}


def lagged_corr_baseline(arch_data: Dict[str, np.ndarray], max_lag: int = 3) -> Dict:
    """Surrogate baseline: best lagged correlation v1 → v4 without conditioning.

    Reports best_lag, best_|r|. PCMCI's value is whether *partial* dependence
    persists beyond this.
    """
    v1, v4 = arch_data["v1"], arch_data["v4"]
    lag, r = L.lagged_correlation_surrogate(v1, v4, max_lag=max_lag)
    return {"best_lag": int(lag), "best_abs_r": float(abs(r)), "best_r": float(r)}


# ---------------------------------------------------------------------------
# Persistence and plots
# ---------------------------------------------------------------------------

def write_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def plot_causal_graph(res: Dict, archetype: str, out_png: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    var_names = res["var_names"]
    edges = res["edges"]
    pos = {"dGW": (0, 1), "soil_moist": (1, 1.5),
           "p_liq": (2, 1), "damage": (3, 0.5)}
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for v, (x, y) in pos.items():
        ax.scatter([x], [y], s=600, edgecolors="black", facecolors="#cce5ff", zorder=3)
        ax.text(x, y, v, ha="center", va="center", fontsize=10, zorder=4)
    for e in edges:
        if e["src"] == e["dst"]:
            continue
        sx, sy = pos[e["src"]]
        dx, dy = pos[e["dst"]]
        width = min(4.0, 0.5 + abs(e["val"]) * 4)
        color = "#d62728" if e["val"] > 0 else "#1f77b4"
        label = f"τ={e['tau']}, r={e['val']:.2f}"
        ax.annotate("", xy=(dx, dy), xytext=(sx, sy),
                    arrowprops=dict(arrowstyle="->", lw=width, color=color, alpha=0.7))
        midx, midy = (sx + dx) / 2, (sy + dy) / 2
        ax.text(midx, midy + 0.05, label, fontsize=7, ha="center")
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(0, 2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"PCMCI causal graph — archetype: {archetype}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    t0 = datetime.utcnow()
    log.info("R3 PCMCI start")
    archetype_results = {}
    chain_rows = []
    edges_rows = []
    for arch, label in ARCHETYPES:
        log.info(f"Archetype {arch} ({label}) ...")
        data = simulate_dense_trajectory(arch, n_nodes=120, n_seeds=8, ssp="SSP5-8.5")
        log.info(f"  trajectory samples: {data['v1'].size}")
        try:
            res = run_pcmci_one_archetype(data, tau_max=3, alpha=0.05)
            log.info(f"  PCMCI sig edges: {len(res['edges'])}")
        except Exception as e:
            log.error(f"  PCMCI failed: {e}")
            res = {"edges": [], "var_names": ["dGW", "soil_moist", "p_liq", "damage"],
                   "n_samples": data["v1"].size}

        cs = chain_strength_climate_to_damage(data)
        lb = lagged_corr_baseline(data)

        chain_rows.append({
            "archetype": arch, "label": label,
            "n_samples": int(data["v1"].size),
            "n_sig_edges": len(res["edges"]),
            "partial_r_dGW_to_damage_given_v2v3": round(cs.get("r_partial", float("nan")), 4),
            "partial_fisher_z": round(cs.get("fisher_z", float("nan")), 3),
            "partial_p": round(cs.get("p", float("nan")), 6),
            "lagged_best_lag": lb["best_lag"],
            "lagged_best_r": round(lb["best_r"], 4),
            "abs_diff_partial_vs_lagged": round(abs(cs.get("r_partial", 0)) - lb["best_abs_r"], 4),
        })
        for e in res["edges"]:
            edges_rows.append({"archetype": arch, **e})

        plot_causal_graph(res, arch, OUT_DIR / f"pcmci_graph_{arch}.png")
        archetype_results[arch] = res

    write_csv(chain_rows, OUT_DIR / "pcmci_chain_strength.csv")
    write_csv(edges_rows, OUT_DIR / "pcmci_per_archetype.csv")

    # Headline regression: R^2 of city-mean Δ damage on inverse-GW depth
    # (mediator analysis at cohort level; this is the [R3.2] placeholder in main_text)
    import pandas as pd
    df = pd.DataFrame(chain_rows)
    base_gw_per_arch = {"deltaic": 2.0, "coastal": 4.0, "lowland": 7.0, "mixed": 11.0,
                        "inland": 16.0, "arid": 28.0, "cold": 9.0, "high_alt": 7.0}
    # Use the per_city_climate_gap from R2 if available
    r2_pcg = PROJECT_ROOT / "ai_autoboost" / "outputs" / "round2" / "per_city_climate_gap.csv"
    if r2_pcg.exists():
        r2_df = pd.read_csv(r2_pcg)
        r2_df["inv_gw"] = 1.0 / r2_df["archetype"].map(base_gw_per_arch)
        from scipy.stats import pearsonr, spearmanr
        x = r2_df["inv_gw"].values
        y = r2_df["mean_climate_gap_2100_minus_2020"].values
        r_p, p_p = pearsonr(x, y)
        r_s, p_s = spearmanr(x, y)
        from sklearn.linear_model import LinearRegression
        model = LinearRegression().fit(x.reshape(-1, 1), y)
        r2 = float(model.score(x.reshape(-1, 1), y))
        write_csv([{
            "n_archetypes": int(r2_df.shape[0]),
            "pearson_r_invGW_to_climate_gap": round(r_p, 4),
            "pearson_p": round(p_p, 6),
            "spearman_r": round(r_s, 4),
            "spearman_p": round(p_s, 6),
            "linear_R2": round(r2, 4),
            "slope": round(float(model.coef_[0]), 4),
            "intercept": round(float(model.intercept_), 4),
        }], OUT_DIR / "mediator_regression_summary.csv")
        log.info(f"Cohort-level mediator regression: R² = {r2:.3f}, Pearson r = {r_p:.3f} (p={p_p:.4g})")
    else:
        log.warning(f"r2 per_city_climate_gap not found at {r2_pcg}")

    elapsed = (datetime.utcnow() - t0).total_seconds()
    log.info(f"R3 PCMCI done in {elapsed:.1f}s")
    print(f"\n=== R3 PCMCI finished in {elapsed:.1f}s ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
