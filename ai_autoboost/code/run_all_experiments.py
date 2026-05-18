"""Top-level orchestrator for CG-STG experiments.

Stages (Rounds):
    --stage env       Scan environment (Round 0)
    --stage r1        Round 1 minimum viable pipeline (CPU OK)
    --stage r2        Round 2 baselines + ablations (GPU recommended; not yet implemented)
    --stage r3        Round 3 mechanism + error analysis (not yet implemented)
    --stage r4        Round 4 generalization + final figures (not yet implemented)

Examples:
    python run_all_experiments.py --stage env
    python run_all_experiments.py --stage r1 --seeds 10 --cities 3 --mc 50
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parent


def run(cmd: list[str]) -> int:
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    return subprocess.call([sys.executable, *cmd])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["env", "r1", "r2", "r3", "r4"], required=True)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--cities", type=int, default=3)
    parser.add_argument("--mc", type=int, default=50, help="Monte Carlo samples per scenario")
    parser.add_argument("--smoke", action="store_true", help="Tiny smoke-test sizes")
    args = parser.parse_args()

    if args.smoke:
        args.seeds = max(3, args.seeds // 3)
        args.mc = max(10, args.mc // 5)

    if args.stage == "env":
        return run([str(CODE_ROOT / "round0_audit" / "scan_environment.py"),
                    "--out", "outputs/round0/env_scan.json"])

    if args.stage == "r1":
        rc = 0
        rc |= run([str(CODE_ROOT / "round1_reproducibility" / "r1_main.py"),
                   "--seeds", str(args.seeds),
                   "--cities", str(args.cities),
                   "--mc", str(args.mc)])
        return rc

    if args.stage in ("r2", "r3", "r4"):
        print(f"[stage {args.stage}] not yet implemented in this revision; will be added in subsequent rounds.")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
