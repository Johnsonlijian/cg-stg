"""Round 3 — End-to-end OSM → CityGraph → CG-STG pipeline on real cities.

Demonstrates the framework consumes real city topology, not just synthesized archetypes.

For each of Tianjin / Bangkok / Jakarta (1 km radius around city centre):
    1. Fetch road network + buildings via osmnx
    2. Map OSM nodes to lifeline classes (road intersections → transport;
       building polygons → building loss receptors; randomly designate a small
       fraction of high-degree road nodes as water/power proxies)
    3. Assign per-node Vs30 + groundwater depth based on city's deltaic baseline
    4. Run B0_static_hazus, B4_cgstg_full at Mw=6.5 R=25 km
    5. Compare CG-STG damage gap to archetype-based predictions from R2

Output:
    outputs/round3/osm_city_graphs/<city>.json
    outputs/round3/osm_end_to_end_results.csv
    outputs/round3/osm_vs_archetype_comparison.png
"""
from __future__ import annotations

import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

CODE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_ROOT.parents[2]
OUT_DIR = PROJECT_ROOT / "ai_autoboost" / "outputs" / "round3"
OSM_DIR = OUT_DIR / "osm_city_graphs"
LOG_DIR = PROJECT_ROOT / "ai_autoboost" / "logs"
for d in (OUT_DIR, OSM_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / f"r3_osm_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.log",
                            encoding="utf-8"),
    ],
)
log = logging.getLogger("r3_osm")

sys.path.insert(0, str(CODE_ROOT.parent / "round2_baselines_ablation"))
import r2_lib as L


# ---------------------------------------------------------------------------
# Cities to load (lat, lon, baseline GW depth, mean Vs30, archetype)
# ---------------------------------------------------------------------------

CITIES = [
    {"name": "Tianjin",  "lat": 39.1421, "lon": 117.1767, "gw_base": 2.5, "vs30_mu": 220.0, "archetype_match": "deltaic"},
    {"name": "Bangkok",  "lat": 13.7563, "lon": 100.5018, "gw_base": 3.5, "vs30_mu": 240.0, "archetype_match": "coastal"},
    {"name": "Jakarta",  "lat": -6.2088, "lon": 106.8456, "gw_base": 5.5, "vs30_mu": 270.0, "archetype_match": "lowland"},
]


def fetch_osm_graph(city: dict, dist_m: int = 1000, sample_n: int = 150) -> L.CityGraph | None:
    """Pull OSM road network + a sample of buildings; convert into a CityGraph for CG-STG."""
    try:
        import osmnx as ox  # type: ignore
        ox.settings.use_cache = True
        ox.settings.timeout = 60
        G = ox.graph_from_point((city["lat"], city["lon"]), dist=dist_m,
                                 network_type="drive", simplify=True)
        log.info(f"  {city['name']}: road graph n_nodes={G.number_of_nodes()} n_edges={G.number_of_edges()}")
        try:
            bldgs = ox.features.features_from_point((city["lat"], city["lon"]),
                                                     tags={"building": True}, dist=dist_m)
            n_bldg = len(bldgs)
        except Exception as e:
            log.warning(f"  building fetch fail: {e}")
            bldgs = None
            n_bldg = 0
    except Exception as e:
        log.error(f"{city['name']} OSM fetch failed: {e}")
        return None

    # Build node lists
    import networkx as nx
    pos = {}
    for n, data in G.nodes(data=True):
        pos[n] = (data.get("x", 0.0), data.get("y", 0.0))
    coords_road = np.array([(pos[n][0], pos[n][1]) for n in G.nodes()])
    if coords_road.size == 0:
        return None

    # Sample transport nodes (road intersections with high degree)
    degs = dict(G.degree())
    road_sorted = sorted(G.nodes(), key=lambda n: degs.get(n, 0), reverse=True)
    n_road_target = max(3, int(round(sample_n * 0.10)))
    transport_keys = road_sorted[:n_road_target]

    # Building nodes (sample subset)
    if bldgs is not None and n_bldg > 0:
        try:
            bldg_centroids = bldgs.geometry.centroid
            xs = np.array(bldg_centroids.x)
            ys = np.array(bldg_centroids.y)
            n_bldg_target = max(20, int(round(sample_n * 0.70)))
            if xs.size > n_bldg_target:
                idx = np.random.default_rng(42).choice(xs.size, size=n_bldg_target, replace=False)
                xs = xs[idx]; ys = ys[idx]
        except Exception as e:
            log.warning(f"building centroid extraction fail: {e}; falling back")
            xs = coords_road[:int(round(sample_n * 0.70)), 0]
            ys = coords_road[:int(round(sample_n * 0.70)), 1]
    else:
        # No buildings — synthesize building locations as offsets from road graph
        n_bldg_target = max(20, int(round(sample_n * 0.70)))
        rng = np.random.default_rng(42)
        idx = rng.choice(coords_road.shape[0], size=n_bldg_target, replace=True)
        xs = coords_road[idx, 0] + rng.normal(0, 5e-5, n_bldg_target)
        ys = coords_road[idx, 1] + rng.normal(0, 5e-5, n_bldg_target)

    # Synthesize water + power node locations as random subset of road intersections
    n_water_target = max(3, int(round(sample_n * 0.10)))
    n_power_target = max(3, int(round(sample_n * 0.10)))
    rng = np.random.default_rng(43)
    remaining_road = [n for n in road_sorted if n not in transport_keys]
    water_keys = rng.choice(remaining_road, size=n_water_target, replace=False) if len(remaining_road) >= n_water_target else remaining_road[:n_water_target]
    remaining_road2 = [n for n in remaining_road if n not in list(water_keys)]
    power_keys = rng.choice(remaining_road2, size=n_power_target, replace=False) if len(remaining_road2) >= n_power_target else remaining_road2[:n_power_target]

    # Combine all into a single node list
    node_records = []
    for n in transport_keys:
        node_records.append({"x": pos[n][0], "y": pos[n][1], "asset_class": 3, "type": "transport"})
    for n in water_keys:
        node_records.append({"x": pos[n][0], "y": pos[n][1], "asset_class": 1, "type": "water"})
    for n in power_keys:
        node_records.append({"x": pos[n][0], "y": pos[n][1], "asset_class": 2, "type": "power"})
    for x_, y_ in zip(xs, ys):
        node_records.append({"x": float(x_), "y": float(y_), "asset_class": 0, "type": "building"})

    n_nodes = len(node_records)
    # Convert lat/lon to local km
    lat0 = city["lat"]
    lon0 = city["lon"]
    R_earth = 6371.0
    x_km = np.array([(r["x"] - lon0) * np.cos(np.deg2rad(lat0)) * np.pi / 180.0 * R_earth for r in node_records])
    y_km = np.array([(r["y"] - lat0) * np.pi / 180.0 * R_earth for r in node_records])
    asset_class = np.array([r["asset_class"] for r in node_records], dtype=np.int64)

    # Vs30 + GW (vary across nodes with city-baseline anchor)
    rng_attr = np.random.default_rng(hash(city["name"]) & 0xFFFFF)
    Vs30 = rng_attr.normal(city["vs30_mu"], 50.0, size=n_nodes).clip(120.0, 800.0)
    GW = rng_attr.normal(city["gw_base"], 1.5, size=n_nodes).clip(0.5, 30.0)

    # Adjacency: each non-building node connects to 3 nearest peers of same class;
    # each building connects to its nearest water/power/transport neighbour
    A = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    coords = np.stack([x_km, y_km], axis=1)
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    np.fill_diagonal(dist, np.inf)
    for c in (1, 2, 3):
        idx = np.where(asset_class == c)[0]
        if idx.size < 2:
            continue
        sub = dist[np.ix_(idx, idx)]
        for il, i in enumerate(idx):
            nbrs = idx[np.argsort(sub[il])[:3]]
            for j in nbrs:
                if j != i:
                    A[i, j] = 1.0 / (1.0 + dist[i, j])
    buildings = np.where(asset_class == 0)[0]
    for i in buildings:
        for c in (1, 2, 3):
            idx_c = np.where(asset_class == c)[0]
            if idx_c.size == 0:
                continue
            j = idx_c[np.argmin(dist[i, idx_c])]
            A[j, i] = 1.0 / (1.0 + dist[i, j])

    cg = L.CityGraph(
        n_nodes=n_nodes,
        Vs30=Vs30, GW_2020=GW, asset_class=asset_class,
        x_km=x_km, y_km=y_km, adjacency=A, archetype=city.get("archetype_match", "unknown"),
    )
    object.__setattr__(cg, "city_name", city["name"])
    object.__setattr__(cg, "n_road_nodes_total", G.number_of_nodes())
    object.__setattr__(cg, "n_buildings_total", n_bldg)

    return cg


def run_cgstg_on_real_city(cg: L.CityGraph, n_seeds: int = 6, n_mc: int = 20,
                            Mw: float = 6.5, R_km: float = 25.0,
                            ssps: Tuple[str, ...] = ("SSP5-8.5", "Control-NoCC"),
                            epochs: Tuple[int, ...] = (2020, 2100)) -> List[Dict]:
    """Run B4_cgstg_full + B0_static_hazus on the loaded city graph."""
    sys.path.insert(0, str(CODE_ROOT.parent / "round2_baselines_ablation"))
    from r2_main import sample_dGW, METHODS, run_one  # noqa: E402

    results = []
    for seed in range(n_seeds):
        rng_seed = np.random.default_rng(seed)
        for ssp in ssps:
            for epoch in epochs:
                ssp_rng = np.random.default_rng(seed * 100_000 + hash(ssp) % 9973 + epoch)
                for mc in range(n_mc):
                    dGW = sample_dGW(ssp_rng, ssp, epoch, cg.archetype) if cg.archetype != "unknown" else 0.0
                    for method_name in ("B0_static_hazus", "B4_cgstg_full"):
                        cfg = METHODS[method_name]
                        d_init, d_final, p_liq, pga, eps, dbc = run_one(cg, Mw, R_km, dGW, cfg, ssp_rng)
                        results.append({
                            "city": getattr(cg, "city_name", "?"),
                            "archetype": cg.archetype, "seed": seed, "ssp": ssp, "epoch": epoch,
                            "mc": mc, "dGW": dGW,
                            "method": method_name,
                            "mean_dmg_final": float(d_final),
                            "mean_p_liq": float(p_liq),
                        })
    return results


def main() -> int:
    t0 = datetime.utcnow()
    log.info("R3 OSM end-to-end start")
    all_rows = []
    city_meta = []
    for city in CITIES:
        log.info(f"Loading OSM for {city['name']} @ ({city['lat']:.4f}, {city['lon']:.4f}) ...")
        cg = fetch_osm_graph(city, dist_m=1000, sample_n=150)
        if cg is None:
            log.warning(f"  skip {city['name']}")
            continue
        # Save graph metadata
        meta = {
            "city": city["name"],
            "n_nodes_in_graph": int(cg.n_nodes),
            "n_road_nodes_total": int(getattr(cg, "n_road_nodes_total", -1)),
            "n_buildings_total": int(getattr(cg, "n_buildings_total", -1)),
            "vs30_mean": float(cg.Vs30.mean()),
            "vs30_std": float(cg.Vs30.std()),
            "gw_mean": float(cg.GW_2020.mean()),
            "asset_class_counts": {int(k): int(v) for k, v in zip(*np.unique(cg.asset_class, return_counts=True))},
            "archetype_match": cg.archetype,
        }
        (OSM_DIR / f"{city['name']}_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                                            encoding="utf-8")
        city_meta.append(meta)

        log.info(f"  Running CG-STG on {city['name']} (n_nodes={cg.n_nodes}, archetype={cg.archetype}) ...")
        rows = run_cgstg_on_real_city(cg, n_seeds=6, n_mc=15, Mw=6.5, R_km=25.0)
        all_rows.extend(rows)
        log.info(f"  → {len(rows)} method-records")

    # Persist raw
    if all_rows:
        with (OUT_DIR / "osm_end_to_end_raw.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)

    # Summarise per city: B4 − B0 gap at SSP5-8.5 2100 vs at Control-NoCC 2100
    import pandas as pd
    df = pd.DataFrame(all_rows)
    summary = []
    for city in df["city"].unique():
        sub = df[df["city"] == city]
        for ssp in ("SSP5-8.5", "Control-NoCC"):
            for epoch in (2020, 2100):
                b0 = sub[(sub["method"] == "B0_static_hazus") & (sub["ssp"] == ssp) & (sub["epoch"] == epoch)]["mean_dmg_final"]
                b4 = sub[(sub["method"] == "B4_cgstg_full") & (sub["ssp"] == ssp) & (sub["epoch"] == epoch)]["mean_dmg_final"]
                if b0.size > 0 and b4.size > 0:
                    gap = float(b4.mean() - b0.mean())
                    summary.append({
                        "city": city, "ssp": ssp, "epoch": epoch,
                        "n_b4": int(b4.size),
                        "b0_mean": round(float(b0.mean()), 5),
                        "b4_mean": round(float(b4.mean()), 5),
                        "gap_b4_minus_b0": round(gap, 5),
                    })
        # Climate gap: B4 2100 - B4 2020 at SSP5-8.5
        for ssp in ("SSP5-8.5",):
            b4_2100 = sub[(sub["method"] == "B4_cgstg_full") & (sub["ssp"] == ssp) & (sub["epoch"] == 2100)]["mean_dmg_final"]
            b4_2020 = sub[(sub["method"] == "B4_cgstg_full") & (sub["ssp"] == ssp) & (sub["epoch"] == 2020)]["mean_dmg_final"]
            if b4_2100.size > 0 and b4_2020.size > 0:
                cg_diff = float(b4_2100.mean() - b4_2020.mean())
                # Bootstrap CI
                rng = np.random.default_rng(13)
                boots = []
                a = b4_2100.values
                b = b4_2020.values
                for _ in range(2000):
                    boots.append(rng.choice(a, a.size, replace=True).mean() -
                                  rng.choice(b, b.size, replace=True).mean())
                lo = float(np.percentile(boots, 2.5))
                hi = float(np.percentile(boots, 97.5))
                summary.append({
                    "city": city, "ssp": "SSP5-8.5_CLIMATE_ISOLATED",
                    "epoch": "2100-2020",
                    "n_b4": int(b4_2100.size),
                    "b0_mean": "",
                    "b4_mean": "",
                    "gap_b4_minus_b0": round(cg_diff, 5),
                    "ci95_lo": round(lo, 5),
                    "ci95_hi": round(hi, 5),
                })

    if summary:
        all_keys = set()
        for r in summary:
            all_keys.update(r.keys())
        fn = sorted(all_keys, key=lambda k: (k != "city", k != "ssp", k != "epoch", k))
        for r in summary:
            for k in fn:
                r.setdefault(k, "")
        with (OUT_DIR / "osm_end_to_end_summary.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fn)
            w.writeheader()
            w.writerows(summary)

    # Plot OSM vs archetype comparison
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        archetype_gap = {
            "deltaic": 0.0239, "coastal": 0.0072, "lowland": 0.0016,
            "mixed": -0.0010, "inland": 0.0029, "arid": 0.0012,
            "cold": -0.0002, "high_alt": -0.0057,
        }
        osm_climate = [r for r in summary if r["ssp"] == "SSP5-8.5_CLIMATE_ISOLATED"]
        cities = [r["city"] for r in osm_climate]
        gaps_osm = [r["gap_b4_minus_b0"] for r in osm_climate]
        gaps_archetype = [archetype_gap.get(next((c["archetype_match"] for c in CITIES if c["name"] == ci), "unknown"), float("nan")) for ci in cities]
        x = np.arange(len(cities))
        fig, ax = plt.subplots(figsize=(8, 4.5))
        width = 0.35
        ax.bar(x - width / 2, gaps_osm, width, label="OSM-real-graph", color="#d62728")
        ax.bar(x + width / 2, gaps_archetype, width, label="Archetype prediction (R2)", color="#1f77b4")
        ax.set_xticks(x)
        ax.set_xticklabels(cities, rotation=10, ha="right")
        ax.set_ylabel("CG-STG climate-isolated gap (2100−2020, SSP5-8.5)")
        ax.set_title("OSM-real vs archetype-predicted climate-induced damage gap")
        ax.axhline(0, color="grey", lw=0.5)
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "osm_vs_archetype_comparison.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        log.warning(f"OSM comparison plot fail: {e}")

    elapsed = (datetime.utcnow() - t0).total_seconds()
    log.info(f"R3 OSM end-to-end done in {elapsed:.1f}s; cities loaded: {len(city_meta)}")
    print(f"\n=== R3 OSM finished in {elapsed:.1f}s ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
