"""Round 13 -- RESS-oriented incremental analyses.

This script converts existing CG-STG outputs into reliability-engineering
evidence tables and adds a lightweight dependency-edge sensitivity experiment.
It does not overwrite earlier rounds.
"""
from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

CODE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_ROOT.parents[2]
OUT_DIR = PROJECT_ROOT / "ai_autoboost" / "outputs" / "round13_ress"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ROUND2 = PROJECT_ROOT / "ai_autoboost" / "outputs" / "round2"
ROUND9 = PROJECT_ROOT / "ai_autoboost" / "outputs" / "round9"

sys.path.insert(0, str(CODE_ROOT.parent / "round2_baselines_ablation"))
import r2_lib as L
from r2_main import ARCHETYPES, METHODS, build_cohort, run_one, sample_dGW


CLASS_LABELS = {
    0: "building",
    1: "water",
    2: "power",
    3: "transport",
}

ARCHETYPE_SEED_OFFSET = {name: 37 * idx for idx, (name, _label) in enumerate(ARCHETYPES)}


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def ci_mean(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if values.size < 5:
        return float(values.mean()), float("nan"), float("nan")
    lo, hi = L.bca_ci(values, n_resamples=2999, rng_seed=1301)
    return float(values.mean()), lo, hi


def method_series(raw: pd.DataFrame, method: str, ssp: str, epoch: int) -> pd.Series:
    sub = raw[(raw["method"] == method) & (raw["ssp"] == ssp) & (raw["epoch"] == epoch)].copy()
    sub["rep"] = sub.groupby(["seed", "city_id"]).cumcount()
    sub = sub.sort_values(["seed", "city_id", "rep"])
    sub = sub.set_index(["seed", "city_id", "rep"])
    return sub["mean_dmg_final"]


def decomposition_tables() -> None:
    raw = pd.read_csv(ROUND2 / "main_methods_raw.csv")
    d00 = method_series(raw, "B0_static_hazus", "Control-NoCC", 2100)
    d10 = method_series(raw, "B1_climate_hazus_no_cascade", "SSP5-8.5", 2100)
    d01 = method_series(raw, "B2_static_cascade", "Control-NoCC", 2100)
    d11 = method_series(raw, "B3_cgstg_hazus_only", "SSP5-8.5", 2100)
    dff = method_series(raw, "B4_cgstg_full", "SSP5-8.5", 2100)
    aligned = pd.concat({"D00_static_nocc": d00, "D10_static_climate": d10,
                         "D01_cascade_nocc": d01, "D11_cascade_climate": d11,
                         "Dfull_ensemble": dff}, axis=1).dropna()

    state_rows = []
    for col, desc in [
        ("D00_static_nocc", "Static-NoCC; no climate, no cascade, HAZUS"),
        ("D10_static_climate", "Static-Climate; climate, no cascade, HAZUS"),
        ("D01_cascade_nocc", "Cascade-NoCC; no climate, cascade, HAZUS"),
        ("D11_cascade_climate", "Cascade-Climate; climate and cascade, HAZUS"),
        ("Dfull_ensemble", "Full framework; climate, cascade, fragility ensemble"),
    ]:
        mean, lo, hi = ci_mean(aligned[col].to_numpy())
        state_rows.append({
            "state": col,
            "definition": desc,
            "n_paired_replicates": len(aligned),
            "mean_final_damage_2100": round(mean, 6),
            "bca_ci_lo": round(lo, 6),
            "bca_ci_hi": round(hi, 6),
        })
    write_rows(OUT_DIR / "static_cascade_climate_states.csv", state_rows)

    components = {
        "local_climate_effect": aligned["D10_static_climate"] - aligned["D00_static_nocc"],
        "network_cascade_effect": aligned["D01_cascade_nocc"] - aligned["D00_static_nocc"],
        "climate_cascade_interaction": aligned["D11_cascade_climate"] - aligned["D10_static_climate"] - aligned["D01_cascade_nocc"] + aligned["D00_static_nocc"],
        "fragility_ensemble_adjustment": aligned["Dfull_ensemble"] - aligned["D11_cascade_climate"],
        "full_minus_static_nocc": aligned["Dfull_ensemble"] - aligned["D00_static_nocc"],
    }
    comp_rows = []
    formulas = {
        "local_climate_effect": "D10 - D00",
        "network_cascade_effect": "D01 - D00",
        "climate_cascade_interaction": "D11 - D10 - D01 + D00",
        "fragility_ensemble_adjustment": "Dfull - D11",
        "full_minus_static_nocc": "Dfull - D00",
    }
    for name, values in components.items():
        mean, lo, hi = ci_mean(values.to_numpy())
        comp_rows.append({
            "component": name,
            "formula": formulas[name],
            "n_paired_replicates": len(values),
            "mean_delta_damage": round(mean, 6),
            "bca_ci_lo": round(lo, 6),
            "bca_ci_hi": round(hi, 6),
        })
    write_rows(OUT_DIR / "static_cascade_climate_decomposition.csv", comp_rows)


def surrogate_validation_table() -> None:
    metric_path = ROUND2 / "gnn_metric_summary.csv"
    if metric_path.exists():
        rows = pd.read_csv(metric_path).fillna("").to_dict(orient="records")
    else:
        loco = pd.read_csv(ROUND2 / "gnn_loco_results.csv")
        pivot = loco.pivot_table(index=["held_out_city_id", "held_out_archetype", "seed"],
                                 columns="model", values="test_rmse").reset_index()
        pivot["gnn_win"] = pivot["gnn"] < pivot["mlp"]
        rows = []
        for model in ["mlp", "gnn"]:
            vals = pivot[model].to_numpy()
            rows.append({
                "model": "Node-local MLP" if model == "mlp" else "GraphSAGE",
                "validation_split": "city leave-one-out",
                "n_pairs": len(vals),
                "rmse_mean": round(float(vals.mean()), 5),
                "rmse_std": round(float(vals.std(ddof=1)), 5),
                "rmse_wins_vs_other": int((pivot["gnn_win"].sum() if model == "gnn" else (~pivot["gnn_win"]).sum())),
            })
        rel = (pivot["mlp"].mean() - pivot["gnn"].mean()) / pivot["mlp"].mean() * 100.0
        rows.append({
            "model": "GraphSAGE advantage",
            "validation_split": "paired city leave-one-out",
            "n_pairs": len(pivot),
            "rmse_wins_vs_other": f"{int(pivot['gnn_win'].sum())}/{len(pivot)}",
            "relative_rmse_reduction_pct": round(float(rel), 2),
        })
    write_rows(OUT_DIR / "surrogate_validation_table.csv", rows)


def positive_city_table() -> None:
    top = pd.read_csv(ROUND9 / "cohort100_top_positive.csv")
    raw = pd.read_csv(ROUND2 / "main_methods_raw.csv")
    sub = raw[(raw["method"] == "B4_cgstg_full") & (raw["ssp"] == "SSP5-8.5") & (raw["epoch"] == 2100)]
    class_cols = [f"dmg_class{i}" for i in range(4)]
    prof = sub.groupby("archetype")[class_cols].mean().reset_index()
    profile = {}
    for _, row in prof.iterrows():
        vals = {CLASS_LABELS[i]: float(row[f"dmg_class{i}"]) for i in range(4)}
        profile[row["archetype"]] = vals
    rows = []
    for _, row in top.iterrows():
        vals = profile.get(row["archetype"], {})
        dominant = max(vals, key=vals.get) if vals else ""
        rows.append({
            "city": row["city"],
            "country": row["country"],
            "archetype": row["archetype"],
            "baseline_groundwater_m": row["gw_base_m"],
            "ssp585_damage_gap": row["mean_gap"],
            "ci_lo": row["ci_lo"],
            "ci_hi": row["ci_hi"],
            "dominant_asset_class_archetype_profile": dominant,
            "class_profile_source": "R2 archetype B4 SSP5-8.5 2100; not city-specific OSM attribution",
        })
    write_rows(OUT_DIR / "positive_ci_city_table_ress.csv", rows)


def module_and_uncertainty_tables() -> None:
    write_rows(OUT_DIR / "module_evidence_status.csv", [
        {"module": "climate perturbation", "method": "SSP/archetype groundwater perturbation", "output": "delta groundwater state", "evidence_status": "scenario screen", "main_limitation": "reduced-form perturbation, not hydrology simulation"},
        {"module": "ground motion", "method": "BSSA14 PGA", "output": "node-level PGA", "evidence_status": "standard GMPE", "main_limitation": "near-field and basin resonance limits"},
        {"module": "liquefaction", "method": "Boulanger-Idriss / Cetin-style probit", "output": "liquefaction probability", "evidence_status": "engineering triggering surrogate", "main_limitation": "Vs30-to-CPT proxy; no site CPT logs"},
        {"module": "fragility", "method": "HAZUS/GEM-like/Jaiswal-like ensemble", "output": "initial damage and epistemic spread", "evidence_status": "screening ensemble", "main_limitation": "not locally calibrated vulnerability"},
        {"module": "cascade", "method": "multiplicative-saturation graph propagation", "output": "system final damage", "evidence_status": "plausibility model with robustness checks", "main_limitation": "dependency graph is not utility-truth"},
        {"module": "GraphSAGE surrogate", "method": "topology-aware learned surrogate", "output": "fast post-cascade damage prediction", "evidence_status": "city-LOCO against simulator", "main_limitation": "trained on simulator, not real damage labels"},
        {"module": "PCMCI diagnostic", "method": "conditional-dependence mediator graph", "output": "mediator-dominance check", "evidence_status": "supplementary diagnostic", "main_limitation": "not strict counterfactual causality"},
        {"module": "DDPM scenario check", "method": "conditional diffusion", "output": "distributional fidelity check", "evidence_status": "supplementary only", "main_limitation": "not required for main reliability conclusions"},
    ])
    write_rows(OUT_DIR / "uncertainty_sources_table.csv", [
        {"source": "GMPE aleatory variability", "type": "aleatory", "treatment": "BSSA14 sigma and Monte Carlo sampling"},
        {"source": "groundwater scenario", "type": "scenario/epistemic", "treatment": "SSP and archetype perturbation"},
        {"source": "soil-state proxy", "type": "epistemic", "treatment": "soil perturbation sweep"},
        {"source": "dependency graph", "type": "epistemic/model", "treatment": "random rewire and class-edge ablation"},
        {"source": "fragility family", "type": "epistemic", "treatment": "weighted HAZUS/GEM-like/Jaiswal-like ensemble"},
        {"source": "surrogate model", "type": "model error", "treatment": "city leave-one-out RMSE"},
        {"source": "historical observations", "type": "validation envelope", "treatment": "sanity-check only, not calibration"},
    ])


@dataclass
class Variant:
    name: str
    description: str


VARIANTS = [
    Variant("baseline", "as-built synthetic dependency graph"),
    Variant("no_dependency_all", "remove all dependency edges"),
    Variant("no_lifeline_backbone", "remove non-building to non-building same-class backbone edges"),
    Variant("no_feeder_to_building", "remove all lifeline feeder-to-building edges"),
    Variant("remove_water_feeders", "remove water-to-building feeder edges"),
    Variant("remove_power_feeders", "remove power-to-building feeder edges"),
    Variant("remove_transport_feeders", "remove transport-to-building feeder edges"),
    Variant("halve_all_edges", "halve all dependency weights"),
    Variant("double_all_edges", "double all dependency weights, clipped at 1"),
]


def clone_graph(cg: L.CityGraph, adjacency: np.ndarray) -> L.CityGraph:
    return L.CityGraph(
        n_nodes=cg.n_nodes,
        Vs30=cg.Vs30.copy(),
        GW_2020=cg.GW_2020.copy(),
        asset_class=cg.asset_class.copy(),
        x_km=cg.x_km.copy(),
        y_km=cg.y_km.copy(),
        adjacency=adjacency.astype(np.float32),
        archetype=cg.archetype,
    )


def variant_graph(cg: L.CityGraph, variant: str) -> L.CityGraph:
    A = cg.adjacency.copy()
    src_class = cg.asset_class[:, None]
    dst_class = cg.asset_class[None, :]
    if variant == "baseline":
        pass
    elif variant == "no_dependency_all":
        A[:, :] = 0.0
    elif variant == "no_lifeline_backbone":
        A[(src_class != 0) & (dst_class != 0)] = 0.0
    elif variant == "no_feeder_to_building":
        A[(src_class != 0) & (dst_class == 0)] = 0.0
    elif variant == "remove_water_feeders":
        A[(src_class == 1) & (dst_class == 0)] = 0.0
    elif variant == "remove_power_feeders":
        A[(src_class == 2) & (dst_class == 0)] = 0.0
    elif variant == "remove_transport_feeders":
        A[(src_class == 3) & (dst_class == 0)] = 0.0
    elif variant == "halve_all_edges":
        A *= 0.5
    elif variant == "double_all_edges":
        A = np.clip(A * 2.0, 0.0, 1.0)
    else:
        raise ValueError(variant)
    return clone_graph(cg, A)


def edge_inventory(cohort: list[L.CityGraph]) -> None:
    rows = []
    for cg in cohort:
        A = cg.adjacency
        for s in range(4):
            for d in range(4):
                mask = (cg.asset_class[:, None] == s) & (cg.asset_class[None, :] == d) & (A > 0)
                rows.append({
                    "archetype": cg.archetype,
                    "src_class": CLASS_LABELS[s],
                    "dst_class": CLASS_LABELS[d],
                    "edge_count": int(mask.sum()),
                    "weight_sum": round(float(A[mask].sum()), 6) if mask.any() else 0.0,
                })
    write_rows(OUT_DIR / "dependency_edge_inventory.csv", rows)


def run_variant_gap(cg: L.CityGraph, variant: str, n_seeds: int = 4, n_mc: int = 6) -> float:
    cg_v = variant_graph(cg, variant)
    cfg = METHODS["B4_cgstg_full"]
    gaps = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(13000 + seed * 101 + ARCHETYPE_SEED_OFFSET[cg.archetype])
        d2020, d2100 = [], []
        for _ in range(n_mc):
            dgw_2020 = sample_dGW(rng, "SSP5-8.5", 2020, cg.archetype)
            dgw_2100 = sample_dGW(rng, "SSP5-8.5", 2100, cg.archetype)
            d2020.append(run_one(cg_v, 6.5, 25.0, dgw_2020, cfg, rng)[1])
            d2100.append(run_one(cg_v, 6.5, 25.0, dgw_2100, cfg, rng)[1])
        gaps.append(float(np.mean(d2100) - np.mean(d2020)))
    return float(np.mean(gaps))


def dependency_sensitivity() -> None:
    rng = np.random.default_rng(20260518)
    cohort = build_cohort(rng, n_nodes=120)
    edge_inventory(cohort)

    rows = []
    for variant in VARIANTS:
        gaps = []
        inv_gw = []
        for cg in cohort:
            gap = run_variant_gap(cg, variant.name)
            gaps.append(gap)
            inv_gw.append(1.0 / float(np.mean(cg.GW_2020)))
        x = np.asarray(inv_gw)
        y = np.asarray(gaps)
        model = LinearRegression().fit(x.reshape(-1, 1), y)
        r, p = pearsonr(x, y)
        rows.append({
            "variant": variant.name,
            "description": variant.description,
            "n_archetypes": len(cohort),
            "R2_gap_vs_inverse_groundwater": round(float(model.score(x.reshape(-1, 1), y)), 5),
            "pearson_r": round(float(r), 5),
            "pearson_p": round(float(p), 6),
            "mean_gap": round(float(np.mean(y)), 6),
            **{f"gap_{cg.archetype}": round(float(g), 6) for cg, g in zip(cohort, gaps)},
        })
    write_rows(OUT_DIR / "graph_dependency_class_sensitivity.csv", rows)


def plots() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    comp = pd.read_csv(OUT_DIR / "static_cascade_climate_decomposition.csv")
    fig, ax = plt.subplots(figsize=(7.0, 3.7))
    x = np.arange(len(comp))
    y = comp["mean_delta_damage"].to_numpy()
    err_lo = y - comp["bca_ci_lo"].to_numpy()
    err_hi = comp["bca_ci_hi"].to_numpy() - y
    ax.bar(x, y, yerr=[err_lo, err_hi], capsize=3, color=["#8da0cb", "#66c2a5", "#fc8d62", "#a6d854", "#e78ac3"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, comp["component"], rotation=25, ha="right")
    ax.set_ylabel("Delta mean final damage")
    ax.set_title("Static / climate / cascade decomposition (R2 paired archetype cohort)")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "Fig_R13_static_cascade_climate_decomposition.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    dep = pd.read_csv(OUT_DIR / "graph_dependency_class_sensitivity.csv")
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    x = np.arange(len(dep))
    ax.bar(x, dep["R2_gap_vs_inverse_groundwater"], color="#4c78a8")
    ax.set_xticks(x, dep["variant"], rotation=30, ha="right")
    ax.set_ylabel("R2: gap vs inverse groundwater")
    ax.set_title("Class-edge dependency sensitivity")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "Fig_R13_dependency_sensitivity.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    val = pd.read_csv(OUT_DIR / "surrogate_validation_table.csv")
    val = val[val["model"].isin(["Node-local MLP", "GraphSAGE"])]
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    ax.bar(val["model"], val["rmse_mean"].astype(float), color=["#f58518", "#4c78a8"])
    ax.set_ylabel("LOCO RMSE")
    ax.set_title("Topology-aware surrogate fidelity")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "Fig_R13_surrogate_validation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    decomposition_tables()
    surrogate_validation_table()
    positive_city_table()
    module_and_uncertainty_tables()
    dependency_sensitivity()
    plots()
    manifest = {
        "round": "R13_RESS",
        "outputs": sorted(p.name for p in OUT_DIR.iterdir() if p.is_file()),
        "notes": [
            "Static/cascade/climate decomposition uses paired R2 8-archetype physics outputs.",
            "Dependency sensitivity uses the actual synthetic dependency graph structure: same-class lifeline backbones plus feeder-to-building edges.",
            "Positive-CI city asset-class attribution is archetype-level, not city-specific OSM attribution.",
        ],
    }
    (OUT_DIR / "round13_ress_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote R13 RESS outputs to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
