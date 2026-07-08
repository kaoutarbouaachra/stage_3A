"""
Function: main.py
Role: Entry point for the STELA solver.

Usage (depuis Algorithm_codes/) :
    # simple
    python -m stela.main --instance setA-01
    # explicit dataset path
    python -m stela.main --instance setA-01 --data-dir path/to/setA-01
    # full manual mode
    python -m stela.main --net path/net.json --tm path/tm.json --scenario path/scenario.json
"""

import argparse
import json
import sys
import os
import time
from pathlib import Path
from core.instance import load_instance, summary
from core.ecmp_from_scratch     import compute_all_splits, compute_loads, sorted_load_vector
from .solver        import SequentialLexSolver, write_solution
from core.checker   import find_checker, run_checker
from core.paths     import CODES_DIR, RESULTS_DIR, CHECKER_DIR


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
        description="T-ASR Sequential Lex-Minimax Solver"
    )

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--instance", "-i",
        help="Instance name (e.g. setA-01, toy, etc.)"
    )

    group.add_argument("--net", help="Path to net.json")

    parser.add_argument("--tm", help="Path to tm.json")
    parser.add_argument("--scenario", help="Path to scenario.json")

    parser.add_argument(
        "--data-dir", "-d",
        default=None,
        help="Base folder for instance files"
    )

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
    parser.add_argument("--mip-gap", type=float, default=1e-4)
    parser.add_argument("--verbose-gurobi", action="store_true")

    return parser.parse_args()


#################################
# Path resolution
#################################
def resolve_paths(args):
    """
    Role: Resolve dataset paths. Instances live in CODES_DIR
          (toy_instance/, setA/<name>/, ar3_instance/, ...).
    """
    if args.net:
        return Path(args.net), Path(args.tm), Path(args.scenario)
    name = args.instance
    if args.data_dir:
        base = Path(args.data_dir)
    elif name.startswith("setA"):
        base = CODES_DIR / "setA" / name
    elif name.startswith("ar"):
        base = CODES_DIR / "ar3_instance"
    elif name == "toy":
        base = CODES_DIR / "toy_instance"
    else:
        base = CODES_DIR / f"{name}_instance"
    net_f  = base / f"{name}-net.json"
    tm_f   = base / f"{name}-tm.json"
    scen_f = base / f"{name}-scenario.json"

    for p in (net_f, tm_f, scen_f):
        if not p.exists():
            print(f"[ERROR] Missing file: {p}")
            sys.exit(1)

    return net_f, tm_f, scen_f

#################################
### Helpers
#################################

def section(title):
    print("\n" + "═" * 60)
    print(f" {title}")
    print("═" * 60)

def print_z_stars(z_stars, N):
    print("\nLex vector (first values):")
    for i, z in enumerate(z_stars[:20], 1):
        bar = "█" * int(z * 30)
        print(f"  z*_{i:3d} = {z:.6f} {bar}")
    if len(z_stars) > 20:
        print(f"  ... ({len(z_stars)} values total, N={N})")

def print_checker(checker_result, mlu):
    diff = abs(checker_result.max_load - mlu)
    print("\nChecker comparison:")
    print(f"  Python MLU  : {mlu:.6f}")
    print(f"  Checker MLU : {checker_result.max_load:.6f}")
    print(f"  Gap         : {diff:.2e}")
    print("  OK" if diff < 1e-4 else "  WARNING")


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
    solver = SequentialLexSolver(
        inst, r,
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        max_iter=args.max_iter,
        verbose=args.verbose_gurobi,
    )

    waypoints, z_stars = solver.solve()

    ### Results
    section("5. Results")
    print_z_stars(z_stars, solver.N)
    sol_loads = compute_loads(waypoints, inst, r)
    sol_vec = sorted_load_vector(sol_loads, inst)
    mlu_sol = sol_vec[0]
    print(f"\nMLU solution: {mlu_sol:.6f}")

    ### Save solution
    instance_out_dir = out_dir / (args.instance or "output")
    instance_out_dir.mkdir(parents=True, exist_ok=True)
    sol_path = instance_out_dir / f"{inst.name}-srpaths.json"
    write_solution(waypoints, inst, sol_path)

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