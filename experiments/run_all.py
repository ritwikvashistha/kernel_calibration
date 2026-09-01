"""Run the full experiment suite in sequence.

Each experiment runs as its own subprocess (so one failure doesn't abort the
rest, and ``--x64`` is applied from a clean interpreter). Flags are passed through.

    python experiments/run_all.py --scale smoke                 # fast local check
    python experiments/run_all.py --scale full --x64 --out runs/2026-08   # server
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

EXPERIMENTS = [
    "exp_type_i.py",
    "exp_power.py",
    "exp_bandwidth.py",
    "exp_recalibration.py",
    "exp_baselines.py",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=["smoke", "small", "full"], default="smoke")
    parser.add_argument("--out", default="experiments/results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--x64", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--only", nargs="*", help="Run only these scripts (e.g. --only exp_type_i.py)."
    )
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    scripts = args.only or EXPERIMENTS
    passthrough = ["--scale", args.scale, "--out", args.out, "--seed", str(args.seed)]
    if args.x64:
        passthrough.append("--x64")
    if args.no_plots:
        passthrough.append("--no-plots")

    failures = []
    for script in scripts:
        print(f"\n{'=' * 70}\n>>> {script}\n{'=' * 70}")
        rc = subprocess.run([sys.executable, str(here / script), *passthrough]).returncode
        if rc != 0:
            failures.append((script, rc))
            print(f"[run_all] {script} exited with code {rc}")

    print(f"\n{'=' * 70}")
    if failures:
        print("[run_all] FAILURES:", failures)
        sys.exit(1)
    print(f"[run_all] all {len(scripts)} experiments completed. Outputs in {args.out}/")


if __name__ == "__main__":
    main()
