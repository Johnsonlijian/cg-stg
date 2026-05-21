# Reproducible Runbook

This runbook describes how to rerun the public CG-STG reproducibility package.

## 1. Environment

Recommended baseline:

- Python 3.11 or later
- Windows, Linux, or macOS
- CPU is sufficient for most scripts
- Optional GPU for heavier graph-learning experiments

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r ai_autoboost/code/requirements.txt
```

Optional packages:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install osmnx geopandas tigramite scikit-learn joblib
```

## 2. Smoke Test

```bash
python ai_autoboost/code/run_all_experiments.py --stage env
python ai_autoboost/code/run_all_experiments.py --stage r1 --seeds 10 --mc 50
```

Expected outputs include files in `ai_autoboost/outputs/round1/`, including `main_results.csv`, `seed_stability.csv`, and `run_meta.json`.

## 3. Main Experiment Sequence

Run from the repository root:

```bash
python ai_autoboost/code/round2_baselines_ablation/r2_main.py --seeds 10 --mc 20
python ai_autoboost/code/round2_baselines_ablation/r2_gnn.py
python ai_autoboost/code/round3_mechanism_error/r3_pcmci.py
python ai_autoboost/code/round3_mechanism_error/r3_osm_pipeline.py
python ai_autoboost/code/round3_mechanism_error/r3_mexico_sensitivity.py
python ai_autoboost/code/round3_mechanism_error/r3_error_analysis.py
python ai_autoboost/code/round4_generalization_final/r4_expanded_cohort.py
python ai_autoboost/code/round4_generalization_final/r4_loao.py
python ai_autoboost/code/round4_generalization_final/r4_robustness.py
python ai_autoboost/code/round4_generalization_final/r4_christchurch.py
python ai_autoboost/code/round5_extension/r5_cohort30.py
python ai_autoboost/code/round5_extension/r5_diffusion.py
python ai_autoboost/code/round6_extension/r6_cohort50.py
python ai_autoboost/code/round7_extension/r7_cohort50_hires.py
python ai_autoboost/code/round7_extension/r7_decadal.py
python ai_autoboost/code/round7_extension/r7_registration.py
python ai_autoboost/code/round8_extension/r8_multi_event.py
python ai_autoboost/code/round8_extension/r8_decadal_acceleration.py
python ai_autoboost/code/round9_extension/r9_cohort100.py
python ai_autoboost/code/round10_extension/r10_global_map.py
python ai_autoboost/code/round12_qc/build_publication_figures.py
python ai_autoboost/code/round13_ress/ress_incremental_analysis.py
python ai_autoboost/code/round14_submission_cleanup/build_ress_framework_figure.py
```

## 4. Key Checks

After rerun, compare the derived outputs against:

- `ai_autoboost/outputs/round9/cohort100_regression.csv`
- `ai_autoboost/outputs/round9/cohort100_top_positive.csv`
- `ai_autoboost/outputs/round2/gnn_vs_mlp_comparison.csv`
- `ai_autoboost/outputs/round4/robustness_summary.csv`
- `ai_autoboost/outputs/final_qc_figures/`
- `ai_autoboost/outputs/round13_ress/static_cascade_climate_decomposition.csv`
- `ai_autoboost/outputs/round13_ress/graph_dependency_class_sensitivity.csv`
- `ai_autoboost/outputs/round13_ress/surrogate_validation_table.csv`
- `ai_autoboost/outputs/round2/gnn_metric_summary.csv`
- `ai_autoboost/outputs/round2/gnn_parity.png`
- `ai_autoboost/outputs/round14_submission_cleanup/Fig_R14_RESS_framework.png`

The forward-registration file is documented in:

`ai_autoboost/outputs/round7/FORWARD_REGISTRATION.md`

## 5. Public Data Boundary

This repository includes code, source registry metadata, derived non-sensitive tables, generated figures, and reproducibility notes. It does not redistribute raw third-party data, downloaded archives, private credentials, active submission manuscripts, cover letters, reviewer-response drafts, or internal logs.
