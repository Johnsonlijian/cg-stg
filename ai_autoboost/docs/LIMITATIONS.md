# LIMITATIONS.md — Round 3 update

Per master-prompt §九, this paper's published limitations should be specific, technical,
and quantitative — not bromides. Each limitation below is paired with what it means
for the conclusions and what would be required to remove it.

## 1. Synthetic ground-motion

We use BSSA14 NGA-West2 (Boore et al. 2014) closed-form formulas with stochastic
inter-event sigma. We do not run 3D physics-based ground-motion simulation (SCEC CyberShake,
SW4) because of cost.

- **Effect on conclusions**: high-frequency motion is underestimated; site-amplification
  uncertainty propagates only through Vs30 + nonlinear site term, not through 3D basin
  effects.
- **Removal requirement**: institutional access to SCEC CyberShake catalogs or 3D
  ground-motion simulation tools at city scale.

## 2. Reduced-form soil-state model

Soil-moisture proxy is a constant-coefficient inverse of dGW, anchored to GLDAS-2
calibrated parameters. We do **not** run a full Biot poroelastic simulation.

- **Effect on conclusions**: the magnitude of dGW → soil_moisture coupling is by
  construction tight (|r| ≥ 0.96 across archetypes); this overestimates how strongly
  identifiable the mediator chain would be from observational hydrologic data alone.
  PCMCI's mediator-chain detection is therefore primarily evidence about the
  *framework* internal consistency, not a claim about real-world causal identifiability.
- **Removal requirement**: integration with calibrated regional hydrology models
  (e.g. ParFlow-CLM, MIKE SHE) per study area.

## 3. Dependency graph is plausibility-based, not utility-truth

We construct lifeline dependency edges by topology heuristics (k-nearest same-class
edges, building → nearest feeder). Real water / power / transport interdependencies are
operator-confidential and asset-specific.

- **Effect on conclusions**: the cascading regime taxonomy (Building < Water,
  Power, Transport in mean damage; Water always highest) reflects the heuristic-graph
  structure as much as any real utility interdependency. The 11.4% GraphSAGE
  improvement over MLP demonstrates that *some* topological structure carries
  predictive signal, but the magnitude is not generalizable beyond cities with
  similar heuristic-graph structure.
- **Removal requirement**: utility-operator partnership with anonymized dependency data
  (e.g. via NIST Community Resilience Planning Guide collaborators).

## 4. Eight archetype cohort, not a city-by-city global sample

We evaluate on 8 archetypes anchored to one representative city each. The OSM probe
demonstrates the framework consumes real topology, but full per-city CG-STG with
city-specific Vs30 / GW / SoilGrids inputs is reserved for Round 4 / a follow-up.

- **Effect on conclusions**: cohort regression R² = 0.79 over 8 points is a 6-degree-of-
  freedom fit; replication on 30+ cities is required before this is reported as a
  global-scale phenomenon.
- **Removal requirement**: Round 4 expanded city sample (target N ≥ 20) using OSM +
  SoilGrids + GLDAS-region-clip + WorldPop, all of which the framework already supports.

## 5. Climate driver is a per-epoch scalar, not annual continuous

We sample ΔGW at 3 (R2) or 17 (R3 PCMCI) epoch slices, not annual continuous trajectories.
This means our claims are about epoch-shift magnitudes, not seasonal extremes.

- **Effect on conclusions**: under-resolves event-clustering scenarios (e.g. multi-year
  drought followed by extreme recharge). Climate-induced compound seismic risk under
  such clustering may be larger than our 17-epoch resolution shows.
- **Removal requirement**: daily-to-monthly hydroclimatic forcing with full Biot or
  surrogate; large compute.

## 6. R2 Mexico-City negative-shift was a small-sample artefact (R3 correction)

R2 reported, with 10 seeds, that the high-altitude archetype showed a "statistically
significant negative climate-induced shift" of −0.006 [−0.012, −0.001]. R3 30-seed
sensitivity at the *exact same parameterisation* yields zero-crossing CI [−0.007, +0.003].
Across 25 (arch_amp, dGW_2100) parameterisations, only 6/25 cells return strictly
negative CI; **0/25 return strictly positive**. We therefore downgrade the bidirectional
H3 framing from R2 to: "deltaic and coastal show positive climate-induced gain; most
archetypes null; high-altitude shows no detectable positive gain under any plausible
parameterisation". This is more honest and consistent with physical intuition.

## 7. No instrumented validation

No earthquake has occurred in any of our 8 archetype anchor cities during the model's
operational window that could be used for forward validation. The framework's claims are
about *static-bias quantification*, not earthquake forecasting.

- **Effect on conclusions**: we have no head-to-head numerical comparison to a real
  instrumented event. All claims are framework-internal-consistency claims.
- **Removal requirement**: not within our reach; the natural validation event would
  have to be a future moderate earthquake in a cohort city with instrumented response.
  We propose this as future work in §4.6.

## 8. GraphSAGE surrogate trained on physics-anchored cascading, not real damage

The GNN surrogate learns to reproduce the multiplicative-saturation cascading simulator
from Round 2, not real post-event damage observations.

- **Effect on conclusions**: 11.4% GraphSAGE > MLP advantage means the GNN *exploits
  graph structure better than a per-node MLP for this physics model*. It does **not**
  mean the GNN would beat MLP on real instrumental data.
- **Removal requirement**: post-event sat-derived damage proxies + utility-reported
  outage maps.

## 9. We do not claim policy-grade calibration

The framework is offered as a long-horizon **planning-layer bias-quantification tool**,
not as an engineering-code substitute or an actuarial risk product.

## Future-work bridges (mapped from limitations)

| Limitation | Bridge in Round 4 | Bridge in follow-up paper |
|---|---|---|
| 1 (Ground motion) | Cross-validate against ShakeMap re-analysis on 1 historic event | SCEC CyberShake |
| 2 (Soil state) | Add single-city full-Biot reference comparison | Regional hydrology coupling |
| 3 (Dependency graph) | Randomize 50 dependency topologies, report stability | Utility-truth collaboration |
| 4 (Cohort size) | Expand to 20+ cities via OSM pipeline | 100+ global cohort |
| 5 (Epoch granularity) | Add 6 intermediate epochs (10-yr step) | Annual-continuous coupled run |
| 6 (Mexico-City artefact) | Verified in R3, documented | — |
| 7 (No instrumented validation) | Christchurch 2010-2011 retrospective comparison | Forward instrumented validation |
| 8 (GNN surrogate target) | Add satellite-damage-proxy training subset | Damage observatory partnership |
