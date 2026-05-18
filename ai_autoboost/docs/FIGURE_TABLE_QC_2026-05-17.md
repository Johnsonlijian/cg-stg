# CompoundSeismic Figure/Table QC Note

Date checked: 2026-05-17 (Asia/Shanghai)

Scope: local Markdown/Word manuscript completion package, 6 main figures, 6 supplementary figures, and 2 manuscript tables. The active manuscript files themselves are not redistributed in this public repository.

## QC Outcome

Current status: pass for the Markdown/Word completion package.

- Final figure directory: `ai_autoboost/outputs/final_qc_figures/`
- Contact sheets: `ai_autoboost/docs/figure_qc_contact_sheet_main_final.png`; `ai_autoboost/docs/figure_qc_contact_sheet_si_final.png`
- Metrics file: `ai_autoboost/docs/figure_qc_metrics_final.json`
- Local manuscript files were rebuilt after QC in the private project workspace; active submission manuscripts are intentionally excluded from this public reproducibility package.

## Figure Quality Checks

All final QC PNG files report 300 dpi metadata and no detected edge-clipping flag.

| Item | Final file | Pixel size | QC note |
| --- | --- | ---: | --- |
| Fig. 1 | `Fig01_concept_qc.png` | 1996 x 1420 | Title corrected to "Climate-Coupled"; panels readable; no clipping. |
| Fig. 2 | `Fig02_cohort100_regression_qc.png` | 1889 x 1588 | 2 x 2 regression layout; titles and axes separated. |
| Fig. 3 | `Fig03_global_map_ssp585_2100_qc.png` | 1955 x 937 | Natural Earth basemap added; top-city labels manually offset to avoid border clipping and major overlap. |
| Fig. 4 | `Fig04_LOAO_R2_stability_qc.png` | 3354 x 1158 | Holdout bars readable; no edge clipping. |
| Fig. 5 | `Fig05_robustness_qc.png` | 2572 x 1140 | Perturbation labels angled and readable; no edge clipping. |
| Fig. 6 | `Fig06_per_class_per_archetype_qc.png` | 2378 x 1054 | Grouped bars and error bars readable; no edge clipping. |
| SI Fig. A | `SI_FigA_GNN_vs_MLP_LOCO_qc.png` | 2138 x 1054 | GNN/MLP comparison readable; no clipping. |
| SI Fig. B | `SI_FigB_PCMCI_deltaic_qc.png` | 1536 x 747 | Weak direct edge changed to a light dashed curve; node labels and edge labels separated. |
| SI Fig. C | `SI_FigC_OSM_vs_archetype_qc.png` | 1898 x 1054 | Bars and legend readable; no clipping. |
| SI Fig. D | `SI_FigD_mexico_sensitivity_qc.png` | 2315 x 985 | Redrawn from CSV; long "zero-crossing" cell text replaced by N/Z/P categorical labels plus legend. |
| SI Fig. E | `SI_FigE_baseline_comparison_qc.png` | 2618 x 1172 | Method labels angled and readable; no clipping. |
| SI Fig. F | `SI_FigF_Mw_R_heatmap_qc.png` | 1500 x 1056 | Heatmap, colorbar and axes readable; no clipping. |

## Figure/Text/Table Consistency

The regenerated Markdown and Word files now contain explicit text callouts for:

- Main figures: Fig. 1, Fig. 2, Fig. 3, Fig. 4, Fig. 5, Fig. 6.
- Supplementary figures: SI Fig. A, SI Fig. B, SI Fig. C, SI Fig. D, SI Fig. E, SI Fig. F.
- Tables: Table 1 and Table 2.

A stale local "Fig. 8 / R10 global map" reference was removed and replaced by Fig. 3. The stale Round-4 caption manifest now points readers to `ai_autoboost/outputs/final_qc_figures/`.

## Residual Notes

For the current Markdown/Word package, the figure set is internally consistent and visually acceptable. For a later strict journal upload package, prefer exporting vector/PDF versions for all plots where source data are available; Fig. 1, Fig. 2, Fig. 3, SI Fig. B, and SI Fig. D already have regenerated PDF counterparts, while the remaining copied round figures are 300 dpi raster QC copies.
