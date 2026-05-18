# Climate-Modulated Seismic Reliability of Urban Lifeline Networks

This repository contains the public reproducibility package for the manuscript project:

**Climate-modulated seismic reliability of urban lifeline networks: a 100-city spatiotemporal graph screening framework**

This package contains code and derived non-sensitive outputs for a planning-layer reliability-screening workflow. The workflow combines climate-conditioned groundwater/soil-state perturbations, ground-motion and liquefaction modules, fragility ensembles, lifeline-network cascading, robustness checks, and topology-aware GraphSAGE surrogate evaluation. PCMCI and diffusion outputs are treated as supplementary diagnostics, not as the central reliability claim.

## Repository Contents

- `ai_autoboost/code/`: Python source for the experiment rounds.
- `ai_autoboost/outputs/`: derived CSV/JSON tables and generated figures.
- `ai_autoboost/outputs/final_qc_figures/`: publication/QC figure set used by the Markdown/Word completion package.
- `ai_autoboost/outputs/round13_ress/`: RESS-oriented static/cascade/climate decomposition, dependency sensitivity, and evidence-status tables.
- `ai_autoboost/docs/`: reproducibility, limitations, traceability, final gate, and figure/table QC notes.
- `DATASETS_AND_LINKS.csv`: source registry and redistribution notes.
- `REPRODUCIBLE_RUNBOOK.md`: install and rerun instructions.

This public package intentionally excludes raw third-party downloads, cache files, internal logs, active submission manuscripts, cover letters, reviewer-response drafts, credentials, and private author/funding material.

## Headline Derived Results

| Quantity | Value | Derived source |
| --- | --- | --- |
| 100-city cohort R2 under SSP5-8.5 | 0.475 | `ai_autoboost/outputs/round9/cohort100_regression.csv` |
| Pearson r under SSP5-8.5 | 0.690 | `ai_autoboost/outputs/round9/cohort100_regression.csv` |
| Strictly positive-CI cities under SSP5-8.5 | 14/100 | `ai_autoboost/outputs/round9/cohort100_top_positive.csv` |
| Local climate effect in RESS decomposition | +0.002446 mean damage; BCa CI [0.000996, 0.003849] | `ai_autoboost/outputs/round13_ress/static_cascade_climate_decomposition.csv` |
| Network cascade effect in RESS decomposition | +0.091074 mean damage; BCa CI [0.089184, 0.092902] | `ai_autoboost/outputs/round13_ress/static_cascade_climate_decomposition.csv` |
| GraphSAGE LOCO advantage | 11.4% versus MLP; Wilcoxon p = 1.5e-5 | `ai_autoboost/outputs/round2/gnn_vs_mlp_comparison.csv` |
| Robustness R2 range | 0.47 to 0.75 across perturbations | `ai_autoboost/outputs/round4/robustness_summary.csv` |
| Forward prediction hash | `43e37430...442100` | `ai_autoboost/outputs/round7/FORWARD_REGISTRATION.md` |

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r ai_autoboost/code/requirements.txt
```

Optional geospatial and graph-learning extensions:

```bash
python -m pip install osmnx geopandas
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install tigramite scikit-learn joblib
```

See `REPRODUCIBLE_RUNBOOK.md` for the full command sequence.

## Figure QC

The final QC figure set is in `ai_autoboost/outputs/final_qc_figures/`. The checked figure/table report is:

`ai_autoboost/docs/FIGURE_TABLE_QC_2026-05-17.md`

All final QC PNG files report approximately 300 dpi metadata, and no edge-clipping flags were detected in the final check.

## Data and Code Availability Text

Suggested manuscript wording:

> The code, reproducible runbook, derived non-sensitive tables, generated figures, source registry, and citation metadata are available at https://github.com/Johnsonlijian/cg-stg. Raw third-party data and downloaded archives are not redistributed; source links and access notes are listed in `DATASETS_AND_LINKS.csv`.

## Citation

Use `CITATION.cff` for machine-readable citation metadata. A placeholder BibTeX record is also included below:

```bibtex
@misc{cgstg2026,
  title = {Climate-modulated seismic reliability of urban lifeline networks: a 100-city spatiotemporal graph screening framework},
  author = {Li, Jian and collaborators},
  year = {2026},
  howpublished = {GitHub repository},
  url = {https://github.com/Johnsonlijian/cg-stg},
  note = {Public reproducibility package; manuscript in preparation}
}
```

## License

Code is released under the MIT License. Derived tables and figures are provided for research reproducibility under the terms described in `DATASETS_AND_LINKS.csv`; third-party source data remain governed by their original licenses.
