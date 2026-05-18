"""Round 7.3 — Forward-prediction registration file.

Locks CG-STG predictions for the 50-city × 4-SSP × 2100 horizon at hi-res precision,
adds a SHA-256 hash of the prediction file, and writes a registration document
that the community can audit when future events occur.

This is the "pre-registration" pattern from clinical trials applied to a
network-scale framework: the prediction file is hashed *now*, and any future
real event in any of the 50 cities can be checked against it.

Output:
    outputs/round7/forward_predictions_locked.csv  (the prediction file)
    outputs/round7/forward_predictions_locked.sha256  (the hash)
    outputs/round7/FORWARD_REGISTRATION.md  (the registration document)
"""
from __future__ import annotations
import csv
import hashlib
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

CODE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_ROOT.parents[2]
R7_DIR = PROJECT_ROOT / "ai_autoboost" / "outputs" / "round7"
R6_DIR = PROJECT_ROOT / "ai_autoboost" / "outputs" / "round6"
LOG_DIR = PROJECT_ROOT / "ai_autoboost" / "logs"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("r7_reg")


def main() -> int:
    t0 = datetime.utcnow()
    # The locked prediction file is the R7 hi-res cohort50 summary,
    # which contains city × SSP × 2100 climate gap mean and CI.
    src = R7_DIR / "cohort50_hires_summary.csv"
    if not src.exists():
        log.error(f"Required input missing: {src}")
        return 1
    dst = R7_DIR / "forward_predictions_locked.csv"
    shutil.copy2(src, dst)

    # SHA-256 of locked file
    h = hashlib.sha256()
    with dst.open("rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    hash_hex = h.hexdigest()
    (R7_DIR / "forward_predictions_locked.sha256").write_text(
        f"{hash_hex}  forward_predictions_locked.csv\n", encoding="utf-8")
    log.info(f"SHA-256: {hash_hex}")

    # Read the predictions file to summarize for the registration document
    rows = []
    with dst.open(encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append(r)
    # Count + top cities
    n_total = len(rows)
    ssp_summary = {}
    for ssp in ["Control-NoCC", "SSP1-2.6", "SSP2-4.5", "SSP5-8.5"]:
        sub = [r for r in rows if r["ssp"] == ssp]
        ssp_summary[ssp] = {
            "n": len(sub),
            "n_positive_CI": sum(1 for r in sub if r["sign"] == "positive"),
            "top3_cities": [],
        }
        sorted_sub = sorted(sub, key=lambda r: float(r["mean_gap"]) if r["mean_gap"] else -999, reverse=True)
        for r in sorted_sub[:3]:
            ssp_summary[ssp]["top3_cities"].append({
                "city": r["city"], "mean_gap": float(r["mean_gap"]),
                "ci_lo": float(r["ci_lo"]), "ci_hi": float(r["ci_hi"]),
            })

    # Registration markdown
    md = ["# Forward Prediction Registration — CG-STG framework, 2026-05-15\n",
          "## Purpose\n",
          "This document locks CG-STG predictions for 50 real OSM-anchored cities × 4 SSP\n",
          "scenarios × the 2100 horizon. The locked file `forward_predictions_locked.csv`\n",
          "and its SHA-256 hash `forward_predictions_locked.sha256` are timestamped 2026-05-15.\n",
          "Future earthquakes in any of the 50 cities can be checked against these locked\n",
          "predictions as anecdotal validation; the hash prevents post-hoc revision.\n",
          "\n## Locked file\n",
          f"- Path: `ai_autoboost/outputs/round7/forward_predictions_locked.csv`\n",
          f"- Records: {n_total}\n",
          f"- SHA-256: `{hash_hex}`\n",
          f"- Timestamp UTC: {t0.isoformat()}Z\n",
          "\n## What the predictions are\n",
          "For each (city, SSP) pair:\n",
          "- `mean_gap` = expected mean of (mean lifeline damage at 2100) − (at 2020) under CG-STG\n",
          "- `ci_lo`, `ci_hi` = BCa 95% bootstrap CI on that gap, n_seeds=6 × n_mc=12 per cell\n",
          "- `sign` = `positive` if CI strictly above 0, `negative` if strictly below, `zero-crossing` else\n",
          "\n## Per-SSP top-3 positive predictions\n"]
    for ssp, s in ssp_summary.items():
        md.append(f"### {ssp}\n")
        md.append(f"- n cities: {s['n']}; strictly-positive CI: {s['n_positive_CI']}/{s['n']}\n")
        md.append("- Top 3 mean gaps:\n")
        for t in s["top3_cities"]:
            md.append(f"   - **{t['city']}** : Δ damage = {t['mean_gap']:+.5f} (95% CI [{t['ci_lo']:+.5f}, {t['ci_hi']:+.5f}])\n")
        md.append("\n")

    md += [
        "## How to audit\n",
        "1. Compute SHA-256 of `forward_predictions_locked.csv`:\n",
        "   ```\n   sha256sum ai_autoboost/outputs/round7/forward_predictions_locked.csv\n   ```\n",
        f"   Expected: `{hash_hex}`\n",
        "2. Compare locked CG-STG climate-isolated 2100 gap CI with any observed post-event damage.\n",
        "3. Locked file is **immutable**; the CG-STG predictions presented in the published\n",
        "   manuscript and at this registration are identical.\n",
        "\n## Caveats\n",
        "- These are **planning-layer** predictions (vulnerability-upper-bound), not actuarial\n",
        "  forecasts. CG-STG damage rate is consistently 1.3–4× higher than observed at\n",
        "  Christchurch 2010/2011 (LIMITATIONS §8 + §9).\n",
        "- The framework is conditioned on the BSSA14 GMPE + Boulanger-Idriss 2014\n",
        "  liquefaction triggering + climate-driven dGW (SSP-anchored) — any of these\n",
        "  upstream choices may change in future GMPE/CRR/CMIP6 revisions; the registration\n",
        "  is tied to this specific framework state, hashed at this date.\n",
        "\n## Related artefacts\n",
        "- 50-city anchor catalog: `ai_autoboost/code/round4_generalization_final/r4_cohort_anchors.py` (R4 18) + `r5_cohort30.py` (R5 12) + `r6_cohort50.py` (R6 20)\n",
        "- Pipeline source: `r7_cohort50_hires.py`\n",
        "- Manuscript reference: revised_main_text.md §3.5\n",
    ]

    (R7_DIR / "FORWARD_REGISTRATION.md").write_text("".join(md), encoding="utf-8")
    log.info(f"FORWARD_REGISTRATION written; {n_total} predictions locked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
