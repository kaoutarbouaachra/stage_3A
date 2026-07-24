"""
Function: run_alexa.py
Role: CLI driver for ALEXA (Assignment-based LEXicographic Algorithm).

Usage (from Algorithm_codes/):
    python -m alexa.run_alexa toy
    python -m alexa.run_alexa setA-01 --max-rank 5 --time-limit 120
    python -m alexa.run_alexa setAP-001 --data-dir=../setAP/ --time-limit=600
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

from core.paths import RESULTS_DIR, LOGS_DIR, CODES_DIR, CHECKER_DIR
from core.dataset import add_dataset_args, resolve_paths, instance_name

# Uppercase, to match STELA and CARLA. Output namespacing is
# results/<instance>/<ALGO>/ and logs/<ALGO>/ across every package.
ALGO_NAME = "ALEXA"


### Gurobi setup -- same two-path pattern as the other packages.
def _setup_gurobi():
    for p in [Path.home() / "gurobi.lic", CODES_DIR / "gurobi.lic"]:
        if p.exists():
            os.environ["GRB_LICENSE_FILE"] = str(p)
            return str(p)
    return None


_lic = _setup_gurobi()


#################################
### Logging
#################################

class _Tee:
    """Writes simultaneously to stdout and a log file."""

    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

    def fileno(self):
        return sys.__stdout__.fileno()


def _setup_logging(name: str) -> tuple:
    log_dir = LOGS_DIR / ALGO_NAME
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{name}_{timestamp}.log"
    log_fh = open(log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_fh)
    return log_path, log_fh


def _teardown_logging(log_fh) -> None:
    sys.stdout = sys.__stdout__
    log_fh.close()


def _results_dir_for(name: str) -> Path:
    d = RESULTS_DIR / name / ALGO_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


#################################
### CLI
#################################

def parse_args():
    p = argparse.ArgumentParser(
        description="T-ASR -- ALEXA (assignment-based lexicographic MILP)"
    )
    add_dataset_args(p)

    p.add_argument(
        "--checker-dir", "--checker_dir",
        dest="checker_dir", default=None,
        help="Checker binary directory",
    )

    # --max-iter is the spelling used by STELA and CARLA; both are accepted
    # so that a single batch script can drive the three packages.
    p.add_argument(
        "--max-rank", "--max-iter",
        type=int, default=None, dest="max_rank",
        help="stop after this many ranks (default: all N=|A|x|T|)",
    )
    p.add_argument(
        "--time-limit", type=float, default=300.0, dest="time_limit",
        help="Gurobi time limit PER RANK/ITERATION, in seconds",
    )
    p.add_argument(
        "--mip-gap", type=float, default=1e-8, dest="mip_gap",
        help="Relative MIP optimality gap, applied per rank. Must stay "
             "strictly below --eps.",
    )
    p.add_argument(
        "--eps", "--epsilon", dest="eps", type=float, default=1e-6,
        help="One-sided freeze tolerance: L[k] <= l_k_opt + eps. Must dominate "
             "the solver tolerances (MIPGap, FeasibilityTol = 1e-6).",
    )
    return p.parse_args()


#################################
### Main
#################################

def main() -> int:
    args = parse_args()
    name = instance_name(args)

    log_path, log_fh = _setup_logging(name)
    res_dir = _results_dir_for(name)
    try:
        net_f, tm_f, scen_f = resolve_paths(args)

        print("=" * 68)
        print("  T-ASR -- ALEXA (Assignment-based LEXicographic Algorithm)")
        print(f"  Algorithm    : {ALGO_NAME}")
        print(f"  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 68)
        print(f"  Instance     : {name}")
        print(f"  Dataset      : {net_f.parent}")
        print(f"  Log          : {log_path}")
        print(f"  Results dir  : {res_dir}")
        print("=" * 68)

        from core.instance import load_instance, summary
        from core.ecmp import compute_all_splits, compute_loads, sorted_load_vector
        from stela.solver import write_solution
        from .solver import AssignmentLexSolver

        inst = load_instance(net_f, tm_f, scen_f)
        print(f"\n  {summary(inst)}")
        r = compute_all_splits(inst)

        trivial = {(d_id, t): [] for (d_id, _, _, _) in inst.demands for t in inst.T}
        base_vec = sorted_load_vector(compute_loads(trivial, inst, r), inst)
        print(f"  MLU baseline  : {base_vec[0]:.6f}")

        solver = AssignmentLexSolver(
            inst, r,
            time_limit=args.time_limit,
            mip_gap=args.mip_gap,
            eps=args.eps,
            max_rank=args.max_rank,
            verbose=False,
        )
        waypoints, z_stars = solver.solve()

        final_vec = (sorted_load_vector(compute_loads(waypoints, inst, r), inst)
                     if waypoints else base_vec)
        mlu_sol = final_vec[0]
        pct = (base_vec[0] - mlu_sol) / max(base_vec[0], 1e-12) * 100
        print(f"\n  {'='*64}")
        print(f"  Algorithm       : {ALGO_NAME}")
        print(f"  MLU baseline    : {base_vec[0]:.6f}")
        print(f"  MLU solution    : {mlu_sol:.6f}  ({pct:+.2f}%)")
        print(f"  Certified ranks : {len(z_stars)} / {solver.built.N}")

        sol_path = res_dir / f"{name}-srpaths.json"
        checker_valid = None
        checker_mlu = None
        if waypoints:
            write_solution(waypoints, inst, sol_path)
            print(f"  Saved -> {sol_path}")

            try:
                from core.checker import find_checker, run_checker
                checker_dir = Path(args.checker_dir) if args.checker_dir else CHECKER_DIR
                checker_bin = find_checker(checker_dir)
                if checker_bin:
                    print("\n  Running checker...")
                    result = run_checker(
                        net_file=net_f, tm_file=tm_f, scenario_file=scen_f,
                        srpaths_file=sol_path, checker_bin=checker_bin,
                    )
                    if result:
                        checker_valid = result.valid
                        checker_mlu = result.max_load
                        print(f"  Feasible    : {result.valid}")
                        print(f"  MLU checker : {result.max_load:.6f}")
            except Exception as e:
                print(f"  Checker: {e}")

        summary_path = res_dir / f"{name}-summary.json"
        with open(summary_path, "w") as f:
            json.dump({
                "algorithm":     ALGO_NAME,
                "instance":      name,
                "N":             solver.built.N,
                "ranks_solved":  len(z_stars),
                "mlu_baseline":  base_vec[0],
                "mlu_solution":  mlu_sol,
                "reduction_pct": pct,
                "z_stars":       z_stars,
                "time_limit":    args.time_limit,
                "mip_gap":       args.mip_gap,
                "eps":           args.eps,
                "checker_valid": checker_valid,
                "checker_mlu":   checker_mlu,
            }, f, indent=2)
        print(f"  Summary saved -> {summary_path}")
        print("=" * 68)
    finally:
        _teardown_logging(log_fh)

    return 0


if __name__ == "__main__":
    sys.exit(main())