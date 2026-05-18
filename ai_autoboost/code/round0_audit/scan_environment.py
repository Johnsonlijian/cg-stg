"""Scan local Python / GPU / package environment and write a JSON snapshot.

Usage:
    python scan_environment.py [--out outputs/round0/env_scan.json]

Exit code 0 always (informational only).
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime
from pathlib import Path


def safe_import(name: str) -> dict:
    try:
        mod = __import__(name)
        version = getattr(mod, "__version__", "unknown")
        return {"available": True, "version": str(version)}
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def gpu_probe() -> dict:
    info = {"backend": None, "cuda_available": False, "devices": []}
    try:
        import torch  # type: ignore
        info["backend"] = "torch"
        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            for i in range(torch.cuda.device_count()):
                info["devices"].append({
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "total_memory_mib": torch.cuda.get_device_properties(i).total_memory // (1024 * 1024),
                })
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="outputs/round0/env_scan.json")
    args = parser.parse_args()

    rec = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "packages": {p: safe_import(p) for p in [
            "numpy", "scipy", "pandas", "matplotlib", "networkx",
            "statsmodels", "yaml", "tqdm",
            "torch", "torch_geometric", "tigramite",
            "geopandas", "rasterio", "xarray", "osmnx",
        ]},
        "gpu": gpu_probe(),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] env scan written to {out_path}")
    print(f"  python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print(f"  gpu_cuda_available: {rec['gpu'].get('cuda_available')}")
    avail_pkgs = [k for k, v in rec["packages"].items() if v.get("available")]
    missing_pkgs = [k for k, v in rec["packages"].items() if not v.get("available")]
    print(f"  available packages ({len(avail_pkgs)}): {', '.join(avail_pkgs)}")
    print(f"  missing packages ({len(missing_pkgs)}): {', '.join(missing_pkgs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
