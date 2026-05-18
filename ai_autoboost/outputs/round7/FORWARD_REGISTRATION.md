# Forward Prediction Registration — CG-STG framework, 2026-05-15
## Purpose
This document locks CG-STG predictions for 50 real OSM-anchored cities × 4 SSP
scenarios × the 2100 horizon. The locked file `forward_predictions_locked.csv`
and its SHA-256 hash `forward_predictions_locked.sha256` are timestamped 2026-05-15.
Future earthquakes in any of the 50 cities can be checked against these locked
predictions as anecdotal validation; the hash prevents post-hoc revision.

## Locked file
- Path: `ai_autoboost/outputs/round7/forward_predictions_locked.csv`
- Records: 200
- SHA-256: `43e3743049381ca094d364857a0893fc7be171c3933e69d462d4a90661442100`
- Timestamp UTC: 2026-05-15T08:16:10.728360Z

## What the predictions are
For each (city, SSP) pair:
- `mean_gap` = expected mean of (mean lifeline damage at 2100) − (at 2020) under CG-STG
- `ci_lo`, `ci_hi` = BCa 95% bootstrap CI on that gap, n_seeds=6 × n_mc=12 per cell
- `sign` = `positive` if CI strictly above 0, `negative` if strictly below, `zero-crossing` else

## Per-SSP top-3 positive predictions
### Control-NoCC
- n cities: 50; strictly-positive CI: 0/50
- Top 3 mean gaps:
   - **Tianjin** : Δ damage = +0.00309 (95% CI [-0.01476, +0.02321])
   - **HoChiMinh** : Δ damage = +0.00274 (95% CI [-0.01687, +0.02249])
   - **Mumbai** : Δ damage = +0.00227 (95% CI [-0.01797, +0.02261])

### SSP1-2.6
- n cities: 50; strictly-positive CI: 0/50
- Top 3 mean gaps:
   - **Santiago** : Δ damage = +0.01603 (95% CI [-0.00770, +0.03937])
   - **Dhaka** : Δ damage = +0.01468 (95% CI [-0.00206, +0.03084])
   - **Seattle** : Δ damage = +0.01463 (95% CI [-0.00723, +0.03578])

### SSP2-4.5
- n cities: 50; strictly-positive CI: 1/50
- Top 3 mean gaps:
   - **Dhaka** : Δ damage = +0.03111 (95% CI [+0.01427, +0.04799])
   - **NewOrleans** : Δ damage = +0.01652 (95% CI [-0.00065, +0.03341])
   - **Tianjin** : Δ damage = +0.01650 (95% CI [-0.00550, +0.04031])

### SSP5-8.5
- n cities: 50; strictly-positive CI: 7/50
- Top 3 mean gaps:
   - **Shanghai** : Δ damage = +0.02986 (95% CI [+0.01133, +0.04883])
   - **Christchurch** : Δ damage = +0.02704 (95% CI [+0.00801, +0.04583])
   - **Dhaka** : Δ damage = +0.02572 (95% CI [+0.00814, +0.04313])

## How to audit
1. Compute SHA-256 of `forward_predictions_locked.csv`:
   ```
   sha256sum ai_autoboost/outputs/round7/forward_predictions_locked.csv
   ```
   Expected: `43e3743049381ca094d364857a0893fc7be171c3933e69d462d4a90661442100`
2. Compare locked CG-STG climate-isolated 2100 gap CI with any observed post-event damage.
3. Locked file is **immutable**; the CG-STG predictions presented in the published
   manuscript and at this registration are identical.

## Caveats
- These are **planning-layer** predictions (vulnerability-upper-bound), not actuarial
  forecasts. CG-STG damage rate is consistently 1.3–4× higher than observed at
  Christchurch 2010/2011 (LIMITATIONS §8 + §9).
- The framework is conditioned on the BSSA14 GMPE + Boulanger-Idriss 2014
  liquefaction triggering + climate-driven dGW (SSP-anchored) — any of these
  upstream choices may change in future GMPE/CRR/CMIP6 revisions; the registration
  is tied to this specific framework state, hashed at this date.

## Related artefacts
- 50-city anchor catalog: `ai_autoboost/code/round4_generalization_final/r4_cohort_anchors.py` (R4 18) + `r5_cohort30.py` (R5 12) + `r6_cohort50.py` (R6 20)
- Pipeline source: `r7_cohort50_hires.py`
- Manuscript reference: revised_main_text.md §3.5
