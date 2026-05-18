# FINAL_CLAIM_MATRIX — Round 7 close (project at near-submission state, R7 enhanced)

Final state after R7 substantive extensions: hi-precision rerun of 50-city × 4-SSP,
decadal trajectories for top-10 cities, forward-prediction SHA-256 registration.

| ID | 主张 | R0 | R1 | R2 | R3 | R4 | R5 | R6 | **R7 final** | Evidence file |
|---|---|---|---|---|---|---|---|---|---|---|
| C01 | Time-varying susceptibility from public formulas | D | C | B | A | A | A | A | **A** | r2_lib + 50-city cohort |
| C02 | CG-STG end-to-end framework | D | C | B | A | A | A | A | **A** | full pipeline R0-R7 |
| C03 | Climate shift significant vs sham | D | C | A | A | A | A | A++ | **A++** | R6/R7 SSP5-8.5 R²=0.56, p<1e-10 |
| C04 | CG-STG vs static gap +0.085 [+0.083, +0.087] | D | C | A | A | A | A | A | **A** | round2/ablation_results.csv |
| C05 | Bias has spatial structure | D | C | B | B | A | A | A | **A** | R7 7 strictly-positive-CI cities all deltaic/coastal |
| C05a | R2 Mexico-City bidirectional (CORRECTION) | — | — | A | C | C | C | C | **C (resolved)** | round3/mexico_sensitivity_grid |
| C06 | Climate→soil→liq→damage mediator chain (PCMCI) | D | D | D | B | B | B | B | **B** | round3/pcmci_per_archetype.csv |
| C07 | Cohort R² on inverse-GW depth | D | D | D | A | A | A++ | A++ | **A++** | R3 (R²=0.79 n=8) + R5 (0.745 n=30) + R6 (0.541 n=50@32) + R7 (0.561 n=50@72) |
| C08 | Cascading regimes differ by class | D | D | B | B | B | B | B | **B** | round3/error_per_class.csv |
| C09 | GNN LOCO 11.4% vs MLP, p=1.5e-5 | D | D | A | A | A | A | A | **A** | round2/gnn_vs_mlp_comparison.csv |
| C10 | Robust under perturbation | D | D | C | C | A | A | A | **A** | round4/robustness_summary R² ∈ [0.47, 0.75] |
| C11 | Reproducible | D | B | A | A | A | A | A | **A** | REPRODUCIBILITY.md + run_meta.json |
| C12 | Planning-layer, not code substitute | A | A | A | A | A | A | A | A | manuscript §4.3 |
| C13 | Cold-region permafrost mediator | D | D | D- | F | F | F | F | **F (deprecated)** | LIMITATIONS §6 |
| C14 | Deltaic > coastal > others | D | C | B | A | A | A++ | A++ | **A++** | R7 top-7 SSP5-8.5: all deltaic/coastal |
| C15 | Conditional diffusion preserves causal structure | D | D | D | D | D | B | B | **B** | round5/diffusion_causal_fidelity.csv |
| C16 | Wang 2025 differentiation | — | A | A | A | A | A | A | **A** | manuscript §1.2 + LIT_REVIEW |
| C17 | GraphSAGE > MLP in LOCO | — | — | A | A | A | A | A | **A** | round2/gnn_vs_mlp_comparison.csv |
| C18 | OSM end-to-end ready | — | — | B | A | A | A++ | A++ | **A++** | R5 30 + R6 50 + R7 50 hi-res |
| C19 | Fragility ensemble propagated | — | — | B | B | B | B | B | **B** | main_methods_raw.csv |
| C20 | Direct dGW→damage weak controlling for mediators | — | — | — | A | A | A++ | A++ | **A++** | R3 PCMCI + R5 diffusion |
| C21 | R2 Mexico-City Type-I error confirmed | — | — | — | A | A | A | A | **A** | round3/mexico_sensitivity_grid.csv |
| C22 | OSM-real ≈ archetype prediction | — | — | — | B | A | A | A | **A** | R4-R7 50-city |
| C23 | LOAO/LOCO R² stable | — | — | — | — | A | A | A | **A** | round4/loao_R2_stability.csv |
| C24 | Christchurch retrospective PGA in-range | — | — | — | — | B | B | B | **B** | round4/christchurch_retrospective.csv |
| C25 | n=30 cohort regression (R²=0.745) | — | — | — | — | — | A | A | **A** | round5/cohort30_regression.csv |
| C26 | 8/30 cities positive CI at R5; 7/50 positive CI at R7 hi-res | — | — | — | — | — | A | (compute trade-off) | **A++** | round5 + round7 cohort50_hires_top_positive_cities.csv |
| C27 | n=50 cohort cross-SSP regression with dose-response | — | — | — | — | — | — | A | **A++** | round6+round7 |
| C28 | Cross-SSP scaling: high-emission scenarios strongest signal | — | — | — | — | — | — | A | **A** | round6/cohort50_scaling.csv |
| C29 | Policy ranking output | — | — | — | — | — | — | B | **B** | round6/cohort50_ranking.csv |
| **C30 (R7 new)** | Hi-precision rerun recovers R5-level per-city CI at 50-city scale | — | — | — | — | — | — | — | **A** | round7/cohort50_hires_top_positive_cities.csv |
| **C31 (R7 new)** | Decadal trajectories show monotonic accumulation under SSP5-8.5 | — | — | — | — | — | — | — | **B** | round7/decadal_summary.csv |
| **C32 (R7 new)** | Forward-prediction registration with SHA-256 audit hash | — | — | — | — | — | — | — | **A** | round7/FORWARD_REGISTRATION.md + .sha256 |

## Final grade distribution (R7 close)

| Grade | Count | Notes |
|---|---|---|
| **A** | **24** | C01, C02, C03++, C04, C05, C07++, C09, C10, C11, C12, C14++, C16, C17, C18++, C20++, C21, C22, C23, C25, C26++, C27++, C28, C30, C32 |
| **B** | **7** | C06, C08, C15, C19, C24, C29, C31 |
| **C** | **1** | C05a (resolved) |
| **D** | **0** | (cleared at R5) |
| **F (deprecated)** | **1** | C13 |

**R7 key upgrades vs R6**:
- **C26 (positive-CI cities)**: A (R5) → R6 compute trade-off → **A++** (R7 recovered at 50-city scale with 7/50)
- **C27 (cross-SSP regression)**: A → **A++** (R7 hi-res: SSP5-8.5 R²=0.561, p<1e-10; SSP2-4.5 R²=0.377)
- New **C30** (hi-res restoration) A
- New **C31** (decadal trajectory monotone) B
- New **C32** (forward-prediction SHA-256 registration) A — methodologically novel for urban-resilience papers

## Total active claims: **32**

A = 24, B = 7, C = 1 (resolved), D = 0, F = 1 (deprecated)

## Distance from submission

- ✓ All claims grade-resolved (D = 0 since R5)
- ✓ Title 11 words (R5)
- ✓ Abstract ≈ 235 words with R7 update header
- ✓ 35 BibTeX entries
- ✓ 14+ figures incl. R7 decadal trajectories
- ✓ All FINAL_* docs current
- ✓ Forward-prediction SHA-256 hash locked
- 🔲 Author list / affiliations (USER must fill)
- 🔲 GitHub repo push (USER must do; recommend tagging the hash commit)
- 🔲 Zenodo DOI mint (USER must do; recommend depositing the locked hash CSV)
- 🔲 LaTeX / .docx conversion (USER preference per venue)
