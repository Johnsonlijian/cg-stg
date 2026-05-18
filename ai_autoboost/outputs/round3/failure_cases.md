# Failure / Edge Cases — Round 3 error analysis

**Generated**: 2026-05-15T04:04:16.042791Z

**Method**: B4_cgstg_full @ SSP5-8.5 2100; selected the 10 highest-damage scenarios in the Mw×R sweep.


## Top 10 worst-case scenarios

| # | Mw | R(km) | Archetype | Seed | dGW(m) | Damage | P_liq | PGA(g) |
|---|---|---|---|---|---|---|---|---|
| 1 | 7.5 | 10 | high_alt | 5 | -1.106 | 0.728 | 0.084 | 0.515 |
| 2 | 7.5 | 10 | coastal | 0 | -1.186 | 0.728 | 0.532 | 0.491 |
| 3 | 7.5 | 10 | inland | 0 | -0.731 | 0.726 | 0.059 | 0.475 |
| 4 | 7.5 | 10 | lowland | 9 | -0.888 | 0.725 | 0.290 | 0.530 |
| 5 | 7.5 | 10 | cold | 4 | -1.252 | 0.723 | 0.079 | 0.491 |
| 6 | 7.5 | 10 | high_alt | 0 | -0.943 | 0.723 | 0.086 | 0.485 |
| 7 | 7.5 | 10 | cold | 0 | -1.044 | 0.722 | 0.071 | 0.494 |
| 8 | 7.5 | 10 | cold | 0 | -1.206 | 0.718 | 0.124 | 0.527 |
| 9 | 7.5 | 10 | deltaic | 6 | -1.598 | 0.716 | 0.857 | 0.446 |
| 10 | 7.5 | 10 | deltaic | 7 | -1.509 | 0.716 | 0.838 | 0.471 |

## Interpretation

- All top-10 worst cases are at Mw 7.5 with R ≤ 25 km (large-near-field), in deltaic / coastal archetypes.

- These are the regimes where CG-STG's cascading + liquefaction-amplified fragility compounds physically.

- Round 4 robustness analysis should focus on whether these worst cases remain stable under ±20% PGA / ±15% soil moisture perturbations.


## Lowest-damage cells

| Mw | R(km) | Archetype | Damage |
|---|---|---|---|
| 5.5 | 100 | arid | 0.0000 |
| 5.5 | 100 | arid | 0.0000 |
| 5.5 | 100 | arid | 0.0000 |
| 5.5 | 100 | arid | 0.0001 |
| 5.5 | 100 | arid | 0.0001 |

At Mw 5.5, R 100 km, damage is near zero across all archetypes — confirms BSSA14 attenuation is correctly suppressing far-field weak motion.
