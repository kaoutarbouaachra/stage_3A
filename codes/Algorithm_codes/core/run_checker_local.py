"""
Function: run_checker_local.py
Role:
    Run the official Orange C++ checker on a T-ASR solution file.
    Shows detailed logs when the solution is infeasible.

Usage (depuis Algorithm_codes/) :
    python -m core.run_checker_local --instance setA-01
    python -m core.run_checker_local --instance setA-02 --srpaths path/to/file.txt
"""

import argparse
import json
import sys
from pathlib import Path

from .checker import find_checker, run_checker, build_instructions
from .paths import CODES_DIR, RESULTS_DIR, CHECKER_DIR


def parse_args():
    p = argparse.ArgumentParser(
        description="Run the Orange checker on a T-ASR solution file."
    )
    p.add_argument("--instance",    required=True)
    p.add_argument("--srpaths",     default=None)
    p.add_argument("--data_dir",    default=None)
    p.add_argument("--checker_dir", default=None)
    return p.parse_args()


def detect_base(name: str, override: str = None) -> Path:
    if override:
        return Path(override)
    if name.startswith("setA"):
        return CODES_DIR / "setA" / name
    if name.startswith("ar"):
        return CODES_DIR / "ar3_instance"
    if name == "toy":
        return CODES_DIR / "toy_instance"
    return CODES_DIR / f"{name}_instance"


###################################
###  Main
###################################
def main():
    args = parse_args()
    name = args.instance
    base   = detect_base(name, args.data_dir)
    net_f  = base / f"{name}-net.json"
    tm_f   = base / f"{name}-tm.json"
    scen_f = base / f"{name}-scenario.json"
    sol_f  = Path(args.srpaths) if args.srpaths \
             else RESULTS_DIR / name / f"{name}-srpaths.json"
    print("=" * 62)
    print("  Orange Checker — T-ASR Solution Validator")
    print("=" * 62)
    print(f"  Instance : {name}")
    print(f"  Net      : {net_f}")
    print(f"  TM       : {tm_f}")
    print(f"  Scenario : {scen_f}")
    print(f"  SR paths : {sol_f}")
    print("=" * 62)

    ### Check all files exist
    missing = [f for f in [net_f, tm_f, scen_f, sol_f] if not f.exists()]
    if missing:
        print("\n  ERROR — missing files:")
        for f in missing:
            print(f"    {f}")
        sys.exit(1)

    ### Find checker binary
    checker_dir = Path(args.checker_dir) if args.checker_dir else CHECKER_DIR
    checker_bin = find_checker(checker_dir)
    if checker_bin is None:
        print("\n  ERROR — checker not found.")
        print(build_instructions(checker_dir))
        sys.exit(1)

    print(f"\n  Checker  : {checker_bin}")
    print("  Running ...")

    ### Run checker
    result = run_checker(
        net_file      = net_f,
        tm_file       = tm_f,
        scenario_file = scen_f,
        srpaths_file  = sol_f,
        checker_bin   = checker_bin,
        verbose       = True,
    )
    if result is None:
        print("  ERROR — checker returned no result.")
        sys.exit(1)

    ### Display results
    print("\n" + "=" * 62)
    print("  RESULTS")
    print("=" * 62)
    print(f"  Valid         : {result.valid}")
    print(f"  MLU           : {result.max_load:.6f}")
    if result.total_cost is not None:
        print(f"  Total cost    : {result.total_cost}")
    if result.total_srpaths is not None:
        print(f"  Total srpaths : {result.total_srpaths}")
    if result.total_segments is not None:
        print(f"  Total segments: {result.total_segments}")
    # MLU per time slot
    if result.objectives:
        print(f"\n  MLU per time slot:")
        for obj in result.objectives:
            print(f"    t={obj['t']} : MLU={obj.get('mlu', '?'):.6f}"
                  f"  jain={obj.get('jain_index', '?'):.4f}"
                  f"  inv_cost={obj.get('inv_load_cost', '?'):.6f}")
    # Top saturated arcs
    if result.saturations:
        print(f"\n  Top-20 most loaded arcs:")
        for s in result.saturations[:20]:
            print(f"    t={s['t']}  {s['from']:>20} → {s['to']:<20}  "
                  f"sat={s['sat']:.6f}")
        print(f"  Total arcs    : {len(result.saturations)}")
    # If infeasible; show raw output for debugging
    if not result.valid:
        print(f"\n  INFEASIBLE — raw checker output:")
        print(f"  {result.raw_output}")

    # Compare with Python solution if available
    py_f = RESULTS_DIR / name / f"{name}-summary.json"
    if py_f.exists() and result.valid:
        with open(py_f) as f:
            py_data = json.load(f)
        py_mlu = py_data.get("mlu_solution")
        if py_mlu:
            gap = abs(result.max_load - py_mlu)
            print(f"\n  Python MLU    : {py_mlu:.6f}")
            print(f"  Checker MLU   : {result.max_load:.6f}")
            print(f"  Gap           : {gap:.2e}  "
                  f"{'OK' if gap < 1e-4 else 'CHECK REQUIRED'}")

    ### Save results to JSON
    out_dir = RESULTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_f = out_dir / f"{name}-checker-loads.json"

    with open(out_f, "w") as f:
        json.dump({
            "instance":      name,
            "valid":         result.valid,
            "max_load":      result.max_load,
            "loads_sorted":  result.loads_sorted,
            "objectives":    result.objectives,
            "total_cost":    result.total_cost,
            "total_srpaths": result.total_srpaths,
        }, f, indent=2)
    print(f"\n  Saved → {out_f}")
    print("=" * 62)


if __name__ == "__main__":
    main()