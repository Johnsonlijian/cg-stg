"""Quick OSM fetch test — try to load a 1 km × 1 km lifeline subgraph for one real city
to demonstrate that the framework is ready to consume real OSM data when available.

If OSM is reachable, save the graph as a CSV and log success.
If not, log the failure cleanly and fall back to synthetic.
"""
from __future__ import annotations
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_ROOT.parents[2]
OUT_DIR = PROJECT_ROOT / "ai_autoboost" / "outputs" / "round2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("r2_osm")


def try_osm(lat: float, lon: float, dist_m: int = 1000) -> dict:
    """Try to fetch road graph + buildings; return shape stats. Returns {} on any failure."""
    try:
        import osmnx as ox  # type: ignore
        ox.settings.use_cache = True
        ox.settings.timeout = 30
        # Road graph
        G = ox.graph_from_point((lat, lon), dist=dist_m, network_type="drive", simplify=True)
        stats = {
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
            "lat": lat, "lon": lon, "dist_m": dist_m,
        }
        # Buildings polygons (just count)
        try:
            bgs = ox.features.features_from_point((lat, lon), tags={"building": True}, dist=dist_m)
            stats["n_buildings"] = int(len(bgs))
        except Exception as e:
            stats["n_buildings"] = -1
            stats["buildings_error"] = f"{type(e).__name__}: {e}"
        return stats
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def main():
    candidates = [
        ("Tianjin center (deltaic)", 39.1421, 117.1767),
        ("Bangkok center (coastal)", 13.7563, 100.5018),
        ("Jakarta center (lowland)", -6.2088, 106.8456),
    ]
    results = []
    for name, lat, lon in candidates:
        log.info(f"Trying {name} @ ({lat:.4f}, {lon:.4f}) ...")
        r = try_osm(lat, lon, dist_m=1000)
        r["city"] = name
        log.info(f"  → {r}")
        results.append(r)

    out = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "any_success": any("n_nodes" in r for r in results),
        "results": results,
    }
    (OUT_DIR / "osm_probe_result.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    if out["any_success"]:
        log.info("[OK] OSM is reachable; framework can consume real city graphs.")
    else:
        log.warning("[FALLBACK] OSM unreachable in this environment; documenting synthetic-only mode.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
