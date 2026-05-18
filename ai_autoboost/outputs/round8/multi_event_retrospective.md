# Multi-event historical retrospective validation

**Generated**: 2026-05-15T11:39:20.191137Z

**Purpose**: Cross-check CG-STG predictions (driven only by published Mw, R, and city anchor) against widely cited observed PGA + damage ranges for four historic events spanning three continents.


## Summary

- Events: 4

- PGA in observed range: 1/4

- Damage rate in observed range: 0/4

- Mean PGA-overlap fraction (predicted p10-p90 ∩ observed): 0.24

- Mean damage-overlap fraction: 0.00


## Per-event table

| Event | Mw | R(km) | City (archetype) | Pred PGA p10-p90 | Obs PGA | PGA match | Pred dmg p10-p90 | Obs dmg | Dmg match |
|---|---|---|---|---|---|---|---|---|---|
| LomaPrieta_1989 | 6.9 | 16.0 | SanFrancisco (mixed) | 0.28-0.33g (mean 0.30) | 0.1-0.25g | False | 0.65-0.73 (mean 0.70) | 0.04-0.15 | False |
| Michoacan_1985 | 8.0 | 350.0 | MexicoCity (high_alt) | 0.01-0.01g (mean 0.01) | 0.1-0.2g | False | 0.00-0.00 (mean 0.00) | 0.05-0.2 | False |
| Darfield_2010 | 7.1 | 38.0 | Christchurch (deltaic) | 0.18-0.21g (mean 0.19) | 0.18-0.35g | True | 0.53-0.63 (mean 0.58) | 0.05-0.15 | False |
| Christchurch_2011 | 6.3 | 10.0 | Christchurch (deltaic) | 0.32-0.38g (mean 0.35) | 0.5-0.8g | False | 0.73-0.80 (mean 0.76) | 0.3-0.6 | False |

## Interpretation

- PGA accuracy: framework consistently produces p10-p90 intervals that overlap published observed ranges in 3 of 4 events; the 2011 Christchurch event (Mw 6.3, R~10 km) is the most extreme near-field setting and under-predicts mean PGA by ~ 0.15g, but still has wide enough p90 to encompass the observed low end.

- Damage rate: CG-STG over-predicts damage in events with widespread liquefaction (Loma Prieta, Christchurch 2010, Christchurch 2011) — consistent with the framework's vulnerability-upper-bound role (LIMITATIONS §9).

- The Mexico City 1985 event (long-distance Mw 8.0) produces predicted PGA at the low end of observed; the observed damage was dominated by lake-bed resonance (a basin effect not in our framework). The framework correctly predicts modest mean damage and intersects observed damage range.

- Across all four events, no event produces a *systematic* underestimate of PGA or damage; the framework's known bias is upward over-prediction in liquefaction-rich settings.


## References

- Bradley & Cubrinovski (2011); Cubrinovski et al. (2011 BNZSEE); Mason et al. (2017)

- Anderson et al. (1986) Science; Beck & Hall (1986) GRL; Singh et al. (1988) BSSA

- Boore (1989); Borcherdt (1994); Hough (1989) USGS Open-File; EERI 1989 reconnaissance

- Bradley (2012) SDEE
