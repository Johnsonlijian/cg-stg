# Data Leakage Audit — Round 1 Smoke Test

**Generated**: 2026-05-14T15:00:32.122766Z
**Pipeline mode**: Round 1 smoke, all-synthetic (no real data download yet).

## Splits enforced

- **Seed split**: 10 independent seeds; per-seed RNG state never shared across seeds.
- **City split**: 5 cities; per-city RNG state is `seed * 1000 + city_id` — independent.
- **Epoch isolation**: each epoch's climate driver is sampled with an independent RNG state seeded by `(seed, city_id, ssp, epoch)`. No epoch ever uses information from another epoch.
- **SSP isolation**: SSP draws are independent; the Control-NoCC sham is generated separately.
- **Static baseline**: built from epoch 2020 of the same (seed, city_id). It is treated as a per-pair *reference*, never as a training target.

## What is NOT a leakage risk in Round 1

This smoke pipeline does not yet train any ML model. No information leakage between train and test
is possible because there is no train/test partition. Round 2 introduces the GNN and at that point
this audit will be extended to a city-LOCO + fault-source isolation check.

## What IS still a risk

- The Round 1 synthetic pipeline shares the **same** hazard-attenuation closed-form across all cities;
  this is not a leakage per se but **does** mean H1 detectability is artificially uniform across cities.
  Round 2 must replace with a per-city GMPE selection.
- The static baseline uses the same per-node Vs30/soil_class as CG-STG — by design, since we are
  comparing the *climate-mediated susceptibility shift*, not site characterisation. This is documented
  in §3.3 of `revised_main_text.md`.

## Verdict (Round 1)

- ❌ No leakage detected at the smoke-test stage.
- Round 2 audit must extend to: (a) city-LOCO; (b) fault-source split within city; (c) train/val split
  for any learned susceptibility surrogate.
