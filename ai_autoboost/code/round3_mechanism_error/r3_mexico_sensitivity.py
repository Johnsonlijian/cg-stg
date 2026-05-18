"""Round 3 — Mexico-City negative-shift sensitivity.

R2 found high-altitude archetype Δ damage 2100-2020 = -0.006 [-0.012, -0.001]
under SSP5-8.5 — a statistically reliable *negative* signal. Reviewer A and B
asked: is this a parameterisation artefact, or does it survive (a) 30-seed
repetition and (b) sweeps of the climate-driver formula?

We sweep dGW under SSP5-8.5 for the high_alt archetype across
    - arch_amp ∈ {0.5, 0.75, 0.95, 1.10, 1.50}      # default 0.95
    - dGW_mean_2100 ∈ {-1.0, -0.5, 0.0, +0.5, +1.0}  # default -1.0 (water rose)
                                                      # +ve means water sank
n_seeds = 30. If the negative-shift signal persists across a wide subset of
parameterisations, the finding is robust; if it flips signs trivially, we
downgrade.

Output:
    outputs/round3/mexico_sensitivity_grid.csv
    outputs/round3/mexico_sensitivity_heatmap.png
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
for d in (OUT_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / f"r3_mex_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.log",
                            encoding="utf-8"),
    ],
)
log = logging.getLogger("r3_mex")

sys.path.insert(0, str(CODE_ROOT.parent / "round2_baselines_ablation"))
import r2_lib as L
from r2_main import METHODS, run_one


def run_one_pair(rng_master: int, n_seeds: int, n_mc: int, n_nodes: int,
                 arch_amp: float, dGW_2100_mean: float) -> Dict:
    """For one (arch_amp, dGW_2100_mean) cell: run n_seeds × n_mc CG-STG and report
    climate-isolated gap mean + 95% CI."""
    method_cfg = METHODS["B4_cgstg_full"]
    diffs = []
    for seed in range(n_seeds):
        cg = L.synthesize_city_graph(np.random.default_rng(seed + rng_master),
                                      n_nodes=n_nodes, archetype="high_alt")
        ssp_rng = np.random.default_rng(abs(seed * 1_000_000 + int(arch_amp * 1000) + int((dGW_2100_mean + 10) * 1000)))
        per_mc_2020 = []
        per_mc_2100 = []
        for mc in range(n_mc):
            # 2020: dGW = 0
            dGW_2020 = float(ssp_rng.normal(0.0, 0.05))
            _, d_final_2020, _, _, _, _ = run_one(cg, 6.5, 25.0, dGW_2020, method_cfg, ssp_rng)
            per_mc_2020.append(d_final_2020)
            # 2100: dGW ~ N(dGW_2100_mean * arch_amp, 0.20)
            dGW_2100 = float(ssp_rng.normal(dGW_2100_mean * arch_amp, 0.20))
            _, d_final_2100, _, _, _, _ = run_one(cg, 6.5, 25.0, dGW_2100, method_cfg, ssp_rng)
            per_mc_2100.append(d_final_2100)
        per_seed_2020 = np.mean(per_mc_2020)
        per_seed_2100 = np.mean(per_mc_2100)
        diffs.append(per_seed_2100 - per_seed_2020)
    diffs = np.array(diffs)
    # BCa CI
    lo, hi = L.bca_ci(diffs, n_resamples=2999, rng_seed=abs(int(arch_amp * 1000) + int((dGW_2100_mean + 10) * 1000) + 7))
    return {
        "arch_amp": arch_amp,
        "dGW_2100_mean": dGW_2100_mean,
        "n_seeds": int(diffs.size),
        "mean_gap_2100_minus_2020": round(float(diffs.mean()), 5),
        "std_gap": round(float(diffs.std(ddof=1)) if diffs.size > 1 else 0.0, 5),
        "bca_ci_lo": round(lo, 5),
        "bca_ci_hi": round(hi, 5),
        "sign": "negative" if hi < 0 else ("positive" if lo > 0 else "zero-crossing"),
    }


def main() -> int:
    t0 = datetime.utcnow()
    arch_amps = [0.50, 0.75, 0.95, 1.10, 1.50]
    dGW_2100_means = [-1.0, -0.5, 0.0, +0.5, +1.0]
    n_seeds = 30
    n_mc = 10  # smaller MC to keep total in reach
    n_nodes = 120

    rows = []
    for amp in arch_amps:
        for dGW in dGW_2100_means:
            r = run_one_pair(rng_master=2026_05_15,
                              n_seeds=n_seeds, n_mc=n_mc, n_nodes=n_nodes,
                              arch_amp=amp, dGW_2100_mean=dGW)
            log.info(f"  arch_amp={amp:.2f}  dGW_2100_mean={dGW:+.2f}  gap={r['mean_gap_2100_minus_2020']:+.5f} CI=[{r['bca_ci_lo']:+.5f}, {r['bca_ci_hi']:+.5f}]  ({r['sign']})")
            rows.append(r)

    out_csv = OUT_DIR / "mexico_sensitivity_grid.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Plot heatmap
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n_a = len(arch_amps); n_d = len(dGW_2100_means)
        grid_gap = np.zeros((n_a, n_d))
        grid_sign = np.zeros((n_a, n_d))
        for r in rows:
            i = arch_amps.index(r["arch_amp"])
            j = dGW_2100_means.index(r["dGW_2100_mean"])
            grid_gap[i, j] = r["mean_gap_2100_minus_2020"]
            grid_sign[i, j] = 1 if r["sign"] == "positive" else (-1 if r["sign"] == "negative" else 0)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        vmax = max(abs(grid_gap.min()), abs(grid_gap.max()), 1e-6)
        im0 = axes[0].imshow(grid_gap, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
        axes[0].set_xticks(range(n_d))
        axes[0].set_xticklabels([f"{d:+.1f}" for d in dGW_2100_means])
        axes[0].set_yticks(range(n_a))
        axes[0].set_yticklabels([f"{a:.2f}" for a in arch_amps])
        axes[0].set_xlabel("dGW_2100_mean (m)")
        axes[0].set_ylabel("arch_amp")
        axes[0].set_title("(a) Mean Δ damage 2100−2020\nhigh-altitude archetype, n=30 seeds")
        for i in range(n_a):
            for j in range(n_d):
                axes[0].text(j, i, f"{grid_gap[i,j]:+.3f}", ha="center", va="center",
                             color="black" if abs(grid_gap[i,j]) < vmax * 0.5 else "white", fontsize=8)
        fig.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(grid_sign, cmap="RdBu_r", vmin=-1, vmax=1, origin="lower")
        axes[1].set_xticks(range(n_d)); axes[1].set_xticklabels([f"{d:+.1f}" for d in dGW_2100_means])
        axes[1].set_yticks(range(n_a)); axes[1].set_yticklabels([f"{a:.2f}" for a in arch_amps])
        axes[1].set_xlabel("dGW_2100_mean (m)"); axes[1].set_ylabel("arch_amp")
        axes[1].set_title("(b) Sign of 95% BCa CI\nblue=negative, red=positive, white=zero-crossing")
        for i in range(n_a):
            for j in range(n_d):
                sign = rows[i * n_d + j]["sign"]
                axes[1].text(j, i, sign, ha="center", va="center", fontsize=7)

        fig.suptitle("Mexico-City (high-altitude archetype) — climate-driver parameterisation sensitivity")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "mexico_sensitivity_heatmap.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        log.warning(f"plot fail: {e}")

    # Verdict
    n_neg = sum(1 for r in rows if r["sign"] == "negative")
    n_pos = sum(1 for r in rows if r["sign"] == "positive")
    n_zero = sum(1 for r in rows if r["sign"] == "zero-crossing")
    log.info(f"Sensitivity verdict: {n_neg} negative-CI / {n_pos} positive-CI / {n_zero} zero-crossing out of {len(rows)} cells")
    log.info(f"Specifically: R2 default (arch_amp=0.95, dGW_2100=-1.0) → corresponds to row index (i={arch_amps.index(0.95)}, j={dGW_2100_means.index(-1.0)})")

    elapsed = (datetime.utcnow() - t0).total_seconds()
    log.info(f"R3 Mexico-City sensitivity done in {elapsed:.1f}s")
    print(f"\n=== Mexico-City sensitivity finished in {elapsed:.1f}s ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
