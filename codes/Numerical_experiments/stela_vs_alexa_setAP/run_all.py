#!/usr/bin/env python3
"""
Role: Run the full STELA vs ALEXA analysis chain.


Usage:
    python run_all.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE.parent.parent / "Algorithm_codes" / "results"


def run(script: str, *arguments: str) -> None:
    path = HERE / script
    if not path.exists():
        raise SystemExit(f"missing script: {path}")
    command = [sys.executable, str(path), *arguments]
    print(f"\n{'=' * 70}\n{script}\n{'=' * 70}")
    completed = subprocess.run(command)
    if completed.returncode != 0:
        raise SystemExit(f"{script} failed with code {completed.returncode}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--outdir", type=Path, default=HERE)
    parser.add_argument("--alexa", default="expe_alexa_setAP")
    parser.add_argument("--stela", default="expe_stela_setAP")
    parser.add_argument("--ranks", type=int, default=30)
    parser.add_argument("--tol", type=float, default=2e-6)
    args = parser.parse_args(argv)

    results = args.results.resolve()
    outdir = args.outdir.resolve()
    if not results.is_dir():
        raise SystemExit(f"results directory not found: {results}\n"
                         f"pass it explicitly with --results")
    outdir.mkdir(parents=True, exist_ok=True)

    run("make_summary.py", str(results), "--alexa", args.alexa,
        "--stela", args.stela, "--outdir", str(outdir))
    run("make_boxplot.py", str(outdir / "summary.csv"), "--outdir", str(outdir))
    run("make_vectors.py", str(outdir / "vectors.csv"), "--outdir", str(outdir),
        "--ranks", str(args.ranks), "--tol", str(args.tol))

    print(f"\n{'=' * 70}\ndone -- everything written to {outdir}")
    print("look at divergence.csv first: it lists every instance where the two "
          "value vectors\ncontradict each other, sorted by worst relative gap.")
    return 0


if __name__ == "__main__":
    sys.exit(main())