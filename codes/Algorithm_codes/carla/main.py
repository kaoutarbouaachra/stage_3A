"""
Function: main.py
Role: Entry point for the CARLA solver.

Usage (from Algorithm_codes/) :
    # positional name, explicit dataset folder (batch style)
    python -m carla.main setAP-001 --data-dir=../setAP/ --time-limit=600
    # named form, auto-resolved location
    python -m carla.main --instance setA-01
    # full manual mode
    python -m carla.main --net path/net.json --tm path/tm.json --scenario path/scenario.json
    # free-threshold variant for the frozen levels
    python -m carla.main --instance setA-01 --free-threshold
"""

import argparse
import json
import os
import time
from pathlib import Path

from core.instance import load_instance, summary
from core.ecmp_from_scratch import compute_all_splits, compute_loads, sorted_load_vector
from core.dataset import add_dataset_args, resolve_paths, instance_name
from .solver import CumulativeLexSolver, write_solution
from core.checker import find_checker, run_checker
from core.paths import CODES_DIR, RESULTS_DIR, CHECKER_DIR

ALGO = "CARLA"


#################################
### Gurobi license setup
#################################

def setup_gurobi_license():
    """
    Automatically locate a Gurobi license file in common locations.
    """
    candidates = [
        Path.home() / "gurobi.lic",
        CODES_DIR / "gurobi.lic",
    ]

    for p in candidates:
        if p.exists():
            os.environ["GRB_LICENSE_FILE"] = str(p)
            return str(p)

    return None


GRB_LICENSE = setup_gurobi_license()


#################################
### CLI
#################################

def parse_args():
    parser = argparse.ArgumentParser(
        description="T-ASR Cumulative Lex-Minimax Solver (CARLA)"
    )
    add_dataset_args(parser)

    parser.add_argument(
        "--out-dir", "-o",
        default=None,
        help="Output folder"
    )

    parser.add_argument(
        "--checker-dir",
        default=None,
        help="Checker binary directory"
    )

    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--time-limit", type=float, default=300.0)

    parser.add_argument(
        "--mip-gap", type=float, default=0.0,
        help="Relative MIP gap. Keep at 0: the objective Theta_k scales "
             "like k*lambda_1, so a relative gap is not the right tolerance."
    )
    parser.add_argument(
        "--mip-gap-abs", type=float, default=1e-8,
        help="Absolute MIP gap. This is the meaningful tolerance for CARLA. "
             "Must stay strictly below --epsilon."
    )
    parser.add_argument(
        "--epsilon", type=float, default=1e-6,
        help="One-sided slack used when freezing a level. Must dominate both "
             "--mip-gap-abs and the solver FeasibilityTol (1e-6)."
    )
    parser.add_argument(
        "--free-threshold", action="store_true",
        help="Use the aggregated form j*z_j + sum u^(j) <= Theta_j for the "
             "frozen levels instead of fixing z_j = lambda_j."
    )
    parser.add_argument("--verbose-gurobi", action="store_true")

    return parser.parse_args()


#################################
### Helpers
#################################

def section(title):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_lex_vector(lam_bars, theta_bars, N):
    print("\nLex vector (first values):")
    for i, (lb, tb) in enumerate(zip(lam_bars[:20], theta_bars[:20]), 1):
        bar = "#" * int(lb * 30)
        print(f"  lambda_{i:<3d} = {lb:.6f}   Theta_{i:<3d} = {tb:.6f}  {bar}")
    if len(lam_bars) > 20:
        print(f"  ... ({len(lam_bars)} ranks certified, N={N})")


def print_checker(checker_result, mlu):
    diff = abs(checker_result.max_load - mlu)
    print("\nChecker comparison:")
    print(f"  Python MLU  : {mlu:.6f}")
    print(f"  Checker MLU : {checker_result.max_load:.6f}")
    print(f"  Gap         : {diff:.2e}")
    print("  OK" if diff < 1e-4 else "  WARNING")


def write_summary(path, inst, lam_bars, theta_bars, N, elapsed, args):
    """Dump the certified prefix and the run parameters next to the solution."""
    payload = {
        "algorithm":      ALGO,
        "instance":       inst.name,
        "N":              N,
        "ranks_solved":   len(lam_bars),
        "lambda_bars":    lam_bars,
        "theta_bars":     theta_bars,
        "time_limit":     args.time_limit,
        "mip_gap":        args.mip_gap,
        "mip_gap_abs":    args.mip_gap_abs,
        "epsilon":        args.epsilon,
        "free_threshold": bool(args.free_threshold),
        "total_time_s":   elapsed,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Summary written -> {path}")


#################################
### Main
#################################

def main():
    args = parse_args()
    net_f, tm_f, scen_f = resolve_paths(args)
    out_dir = Path(args.out_dir) if args.out_dir else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    t_global = time.perf_counter()

    ### Load instance
    section("1. Loading instance")
    inst = load_instance(net_f, tm_f, scen_f)
    print(summary(inst))

    ### ECMP
    section("2. ECMP computation")
    r = compute_all_splits(inst)

    ### Baseline
    section("3. Baseline")
    trivial = {(d, t): [] for (d, _, _, _) in inst.demands for t in inst.T}
    base_loads = compute_loads(trivial, inst, r)
    base_vec = sorted_load_vector(base_loads, inst)
    print(f"MLU baseline: {base_vec[0]:.6f}")

    ### Solver
    section("4. Solver")
    print(f"License: {GRB_LICENSE or 'NOT FOUND'}")
    solver = CumulativeLexSolver(
        inst, r,
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        mip_gap_abs=args.mip_gap_abs,
        epsilon=args.epsilon,
        max_iter=args.max_iter,
        free_threshold=args.free_threshold,
        verbose=args.verbose_gurobi,
    )

    waypoints, lam_bars, theta_bars = solver.solve()

    ### Results
    section("5. Results")
    print_lex_vector(lam_bars, theta_bars, solver.N)
    sol_loads = compute_loads(waypoints, inst, r)
    sol_vec = sorted_load_vector(sol_loads, inst)
    mlu_sol = sol_vec[0]
    print(f"\nMLU solution: {mlu_sol:.6f}")

    ### Save solution
    instance_out_dir = out_dir / instance_name(args) / ALGO
    instance_out_dir.mkdir(parents=True, exist_ok=True)
    sol_path = instance_out_dir / f"{inst.name}-srpaths.json"
    write_solution(waypoints, inst, sol_path)

    elapsed = time.perf_counter() - t_global
    write_summary(
        instance_out_dir / f"{inst.name}-summary.json",
        inst, lam_bars, theta_bars, solver.N, elapsed, args,
    )

    ### Checker
    section("6. Checker")
    checker_dir = Path(args.checker_dir) if args.checker_dir else CHECKER_DIR
    checker_bin = find_checker(checker_dir)
    if checker_bin:
        checker_result = run_checker(
            net_file=net_f,
            tm_file=tm_f,
            scenario_file=scen_f,
            srpaths_file=sol_path,
            checker_bin=checker_bin,
        )
        if checker_result:
            print_checker(checker_result, mlu_sol)

    ### Final
    print("\nDone.")
    print(f"Total time: {time.perf_counter() - t_global:.2f}s")


if __name__ == "__main__":
    main()