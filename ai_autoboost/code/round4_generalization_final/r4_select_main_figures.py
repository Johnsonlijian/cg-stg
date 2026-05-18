"""R4.5 — Pick 6 main figures + copy to final_figures/ with Nature-Cities-friendly names."""
from __future__ import annotations
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIG_DST = PROJECT_ROOT / "ai_autoboost" / "outputs" / "round4" / "final_figures"
FIG_DST.mkdir(parents=True, exist_ok=True)

# Source figures to canonicalize as Fig 1-6 for Nature Cities main text
SELECTION = [
    ("figures/fig1_concept.png",
     "Fig01_concept.png",
     "CG-STG framework — climate → soil-state → susceptibility → cascading damage"),
    ("ai_autoboost/outputs/round4/expanded_cohort_climate_gap.png",
     "Fig02_cohort_climate_gap.png",
     "Climate-isolated Δ damage per real OSM city (2100 vs 2020 SSP5-8.5), n=18"),
    ("ai_autoboost/outputs/round4/expanded_cohort_regression.png",
     "Fig03_cohort_regression.png",
     "Cohort regression on inverse-baseline-GW depth (R² = 0.50, p = 0.001, n = 18)"),
    ("ai_autoboost/outputs/round4/loao_R2_stability.png",
     "Fig04_LOAO_R2_stability.png",
     "Regression-R² stability under leave-one-city-out and leave-one-archetype-out"),
    ("ai_autoboost/outputs/round4/robustness_heatmap.png",
     "Fig05_robustness.png",
     "R² stability under ±20% PGA, ±15% soil-moisture, ≤±20% dependency-graph rewire"),
    ("ai_autoboost/outputs/round3/error_residual_dist.png",
     "Fig06_per_class_per_archetype.png",
     "Per-archetype × per-asset-class damage regime under cascading (8 archetypes)"),
]

SI_FIGURES = [
    ("ai_autoboost/outputs/round2/gnn_loco.png",
     "SI_FigA_GNN_vs_MLP_LOCO.png",
     "GraphSAGE vs MLP per-node baseline LOCO test RMSE (16/16 GNN wins, p = 1.5e-5)"),
    ("ai_autoboost/outputs/round3/pcmci_graph_deltaic.png",
     "SI_FigB_PCMCI_deltaic.png",
     "PCMCI causal graph — deltaic archetype"),
    ("ai_autoboost/outputs/round3/osm_vs_archetype_comparison.png",
     "SI_FigC_OSM_vs_archetype.png",
     "OSM-real vs archetype-predicted climate gap (3 cities, R3)"),
    ("ai_autoboost/outputs/round3/mexico_sensitivity_heatmap.png",
     "SI_FigD_mexico_sensitivity.png",
     "Mexico-City 25-cell parameter sensitivity (R3 correction)"),
    ("ai_autoboost/outputs/round2/baseline_comparison.png",
     "SI_FigE_baseline_comparison.png",
     "Round-2 method-by-method comparison (B0-B4 + A1-A5)"),
    ("ai_autoboost/outputs/round2/mwr_heatmap.png",
     "SI_FigF_Mw_R_heatmap.png",
     "Mw × R hazard-space damage map (B4 full framework)"),
]


def main():
    print("Main figures:")
    for src_rel, dst_name, caption in SELECTION:
        src = PROJECT_ROOT / src_rel
        if src.exists():
            dst = FIG_DST / dst_name
            shutil.copy2(src, dst)
            print(f"  ✓ {dst_name}  ←  {src_rel}")
        else:
            print(f"  ✗ MISSING source: {src_rel}")
    print("\nSI figures:")
    for src_rel, dst_name, caption in SI_FIGURES:
        src = PROJECT_ROOT / src_rel
        if src.exists():
            dst = FIG_DST / dst_name
            shutil.copy2(src, dst)
            print(f"  ✓ {dst_name}")
        else:
            print(f"  ✗ MISSING: {src_rel}")

    # Write caption manifest
    manifest = ["# Final figure manifest — for Nature Cities submission package\n",
                "## Main text figures\n"]
    for _, dst_name, caption in SELECTION:
        manifest.append(f"### {dst_name}\n{caption}\n")
    manifest.append("\n## Supplementary Information figures\n")
    for _, dst_name, caption in SI_FIGURES:
        manifest.append(f"### {dst_name}\n{caption}\n")
    (FIG_DST / "caption_manifest.md").write_text("\n".join(manifest), encoding="utf-8")
    print(f"\nCaption manifest: {FIG_DST / 'caption_manifest.md'}")


if __name__ == "__main__":
    main()
