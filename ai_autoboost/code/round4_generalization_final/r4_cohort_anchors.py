"""City anchor catalog for Round 4 expanded cohort.

Each city has a lat/lon, an archetype label, an empirically-anchored baseline
groundwater depth (m, from published regional water-table maps), and a mean Vs30
(m/s) drawn from USGS Global Vs30 grid approximations.

These values are NOT precision-calibrated to each city; they are *defensible
priors* per ASSUMPTIONS.md A02/A06. R5 / follow-up would replace these with
city-specific SoilGrids + GLDAS extractions.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List


@dataclass
class CityAnchor:
    name: str
    country: str
    lat: float
    lon: float
    archetype: str
    gw_base_m: float
    vs30_mu: float
    note: str = ""


COHORT_R4: List[CityAnchor] = [
    CityAnchor("Tianjin",       "China",        39.1421, 117.1767, "deltaic",   2.5, 220.0, "Hai River delta; well-documented shallow GW"),
    CityAnchor("Bangkok",       "Thailand",     13.7563, 100.5018, "coastal",   3.5, 240.0, "Chao Phraya delta; subsiding"),
    CityAnchor("Jakarta",       "Indonesia",    -6.2088, 106.8456, "lowland",   5.5, 270.0, "extreme subsidence, GW pumping"),
    CityAnchor("NewOrleans",    "USA",          29.9511, -90.0715, "deltaic",   2.0, 200.0, "Mississippi delta; below sea level"),
    CityAnchor("Manila",        "Philippines",  14.5995, 120.9842, "coastal",   3.5, 230.0, "Pasig River + Manila Bay"),
    CityAnchor("Christchurch",  "NewZealand",  -43.5320, 172.6306, "deltaic",   1.5, 210.0, "Avon-Heathcote estuary; KNOWN 2010-2011 liquefaction"),
    CityAnchor("Wellington",    "NewZealand",  -41.2924, 174.7787, "coastal",   8.0, 320.0, "harbour with stiff fill"),
    CityAnchor("MexicoCity",    "Mexico",       19.4326, -99.1332, "high_alt",  6.0, 260.0, "Texcoco lakebed; altitude 2240m"),
    CityAnchor("Lima",          "Peru",        -12.0464, -77.0428, "arid",     20.0, 380.0, "arid coastal Andean piedmont"),
    CityAnchor("Cairo",         "Egypt",        30.0444,  31.2357, "arid",      8.0, 350.0, "Nile valley margin"),
    CityAnchor("Istanbul",      "Turkey",       41.0082,  28.9784, "mixed",    12.0, 340.0, "Bosphorus mixed bedrock + alluvium"),
    CityAnchor("Kathmandu",     "Nepal",        27.7172,  85.3240, "high_alt",  8.0, 310.0, "Kathmandu valley sediments"),
    CityAnchor("Mumbai",        "India",        19.0760,  72.8777, "coastal",   5.0, 240.0, "Arabian Sea coast"),
    CityAnchor("Lagos",         "Nigeria",       6.5244,   3.3792, "lowland",   3.5, 220.0, "Niger delta margin"),
    CityAnchor("Tokyo",         "Japan",        35.6762, 139.6503, "coastal",   5.0, 270.0, "Kanto plain; well-instrumented"),
    CityAnchor("Osaka",         "Japan",        34.6937, 135.5023, "coastal",   4.5, 250.0, "Yodo River delta"),
    CityAnchor("SanFrancisco",  "USA",          37.7749, -122.4194, "mixed",    10.0, 330.0, "Bay margin; Marina fill liquefaction in 1989"),
    CityAnchor("Beijing",       "China",        39.9042, 116.4074, "mixed",    11.0, 310.0, "North China Plain margin"),
]


def get_cohort_dicts():
    return [asdict(c) for c in COHORT_R4]


if __name__ == "__main__":
    import json
    print(json.dumps(get_cohort_dicts(), indent=2))
    print(f"Total: {len(COHORT_R4)} cities; archetypes: {sorted(set(c.archetype for c in COHORT_R4))}")
