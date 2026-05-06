import os
import sys
import json
import argparse
import datetime
from pathlib import Path
import warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings("ignore")
BLUE   = '#1F77B4'
ORANGE = '#FF7F0E'
# =========================
# ARGUMENTS
# =========================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="../toy_instance/")
    parser.add_argument("--checker_dir", type=str, default="../checker")
    parser.add_argument("--time_limit", type=int, default=120)
    parser.add_argument("--max_iter", type=int, default=36)
    parser.add_argument("--mip_gap", type=float, default=1e-4)
    return parser.parse_args()


# =========================
# FOLDERS
# =========================
def setup_folders(instance):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    log_dir = Path("logs") / instance / timestamp
    res_dir = Path("results") / instance

    log_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    return log_dir, res_dir


# =========================
# LOGGER
# =========================
class Logger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "w", encoding="utf-8")  # ✅ FIX

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

# =========================
# MAIN
# =========================
def main():
    args = parse_args()

    INSTANCE_NAME = args.instance
    DATA_DIR = Path(args.data_dir)
    CHECKER_DIR = Path(args.checker_dir)

    MAX_ITER = args.max_iter
    TIME_LIMIT = args.time_limit
    MIP_GAP = args.mip_gap

    # Setup folders
    log_dir, res_dir = setup_folders(INSTANCE_NAME)

    # Logging
    sys.stdout = Logger(log_dir / "run.log")

    print("=" * 60)
    print(f"Instance : {INSTANCE_NAME}")
    print(f"Logs     : {log_dir}")
    print(f"Results  : {res_dir}")
    print("=" * 60)

    # Save config
    with open(log_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # =========================
    # IMPORTS PROJET
    # =========================
    sys.path.insert(0, '..')

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from instance import load_instance, summary
    from ecmp import compute_all_splits, compute_loads, sorted_load_vector
    from solver import SequentialLexSolver, write_solution
    from checker import find_checker, run_checker

    # =========================
    # LOAD INSTANCE
    # =========================
    base = DATA_DIR

    inst = load_instance(
        base / f'{INSTANCE_NAME}-net.json',
        base / f'{INSTANCE_NAME}-tm.json',
        base / f'{INSTANCE_NAME}-scenario.json',
    )

    print(summary(inst))

    # =========================
    # ECMP
    # =========================
    print("\nComputing ECMP...")
    r = compute_all_splits(inst)

    trivial_wp = {
        (d_id, t): []
        for (d_id, _, _, _) in inst.demands
        for t in inst.T
    }

    baseline_loads = compute_loads(trivial_wp, inst, r)
    baseline_vec = sorted_load_vector(baseline_loads, inst)

    print(f"Baseline MLU = {baseline_vec[0]:.6f}")

    # =========================
    # SOLVER
    # =========================
    print("\nRunning solver...")

    solver = SequentialLexSolver(
        inst, r,
        time_limit=TIME_LIMIT,
        mip_gap=MIP_GAP,
        max_iter=MAX_ITER,
        verbose=True,
    )

    waypoints, z_stars = solver.solve()

    sol_path = res_dir / f"{INSTANCE_NAME}-srpaths.json"
    write_solution(waypoints, inst, sol_path)

    print(f"Solution saved → {sol_path}")

    # =========================
    # CHECKER
    # =========================
    checker_bin = find_checker(CHECKER_DIR)

    if checker_bin:
        result = run_checker(
            net_file=base / f'{INSTANCE_NAME}-net.json',
            tm_file=base / f'{INSTANCE_NAME}-tm.json',
            scenario_file=base / f'{INSTANCE_NAME}-scenario.json',
            srpaths_file=sol_path,
            checker_bin=checker_bin,
        )

        print(f"Feasible : {result.feasible}")
        print(f"MLU (checker) : {result.max_load:.6f}")

    else:
        print("Checker not found")

    # =========================
    # PLOTS 
    # =========================

    # --- Compute solution loads ---
    sol_loads = compute_loads(waypoints, inst, r)

    vec_base = sorted_load_vector(baseline_loads, inst)
    vec_sol  = sorted_load_vector(sol_loads, inst)

    mlu_base = vec_base[0]
    mlu_sol  = vec_sol[0]
    pct = (mlu_base - mlu_sol) / max(mlu_base, 1e-12) * 100
    def plot_sorted_loads(loads_dict, inst, ax, title, color='blue', vline=None):
        vec = sorted_load_vector(loads_dict, inst)
        nonzero = [v for v in vec if v > 1e-9]

        if not nonzero:
            ax.text(0.5, 0.5, 'No load', ha='center', va='center', transform=ax.transAxes)
            return

        ax.bar(range(len(nonzero)), nonzero, color=color, alpha=0.75, width=0.8)

        if vline is not None:
            ax.axhline(vline, linestyle='--', linewidth=1.5,
                    label=f'MLU = {vline:.4f}')
            ax.legend()

        ax.axhline(1.0, linestyle=':', linewidth=1, alpha=0.3)

        ax.set_xlabel('Sorted arcs')
        ax.set_ylabel('Load')
        ax.set_title(title)

    # =========================
    # 1. OVERLAY PLOT 
    # =========================
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(vec_base,
            linestyle='--',
            linewidth=2,
            label=f'Baseline (MLU={mlu_base:.4f})')

    ax.plot(vec_sol,
            linewidth=2,
            label=f'Solution (MLU={mlu_sol:.4f}, ↓ {pct:.1f}%)')

    ax.fill_between(range(len(vec_base)), vec_base, alpha=0.1)
    ax.fill_between(range(len(vec_sol)), vec_sol, alpha=0.2)

    ax.set_xlabel('Rank (sorted arcs)')
    ax.set_ylabel('Load λ(a,t)')
    ax.set_title('Sorted load vectors comparison', fontweight='bold')

    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(res_dir / f'{INSTANCE_NAME}_loads_overlay.png', bbox_inches='tight')
    plt.close()


    # =========================
    # 2. SIDE-BY-SIDE 
    # =========================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    plot_sorted_loads(
        baseline_loads, inst, ax1,
        title=f'Baseline (MLU={mlu_base:.4f})',
        color=BLUE,
        vline=mlu_base
    )

    plot_sorted_loads(
        sol_loads, inst, ax2,
        title=f'Solution (MLU={mlu_sol:.4f}, ↓ {pct:.1f}%)',
        color=ORANGE,
        vline=mlu_sol
    )

    ax1.set_ylabel('Load λ(a,t)')

    plt.suptitle(
        f'{INSTANCE_NAME} — Load distribution comparison',
        fontsize=14, fontweight='bold'
    )

    plt.tight_layout()
    plt.savefig(res_dir / f'{INSTANCE_NAME}_loads_comparison.png', bbox_inches='tight')
    plt.close()


    # =========================
    # 3. FULL LEX VECTOR (optionnel mais utile)
    # =========================
    
    N_total = len(inst.arc_ids) * len(inst.T)

    def pad(v):
        return v + [0.0] * (N_total - len(v))

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(pad(vec_base), linestyle='--', label='Baseline')
    ax.plot(pad(vec_sol), label='Solution')

    if result and hasattr(result, "loads_sorted"):
        ax.plot(pad(result.loads_sorted), linestyle=':', label='Checker')

    ax.legend()
    ax.set_title('Lex vector')

    plt.tight_layout()
    plt.savefig(res_dir / f'{INSTANCE_NAME}_lex.png')
    plt.close()

# =========================
if __name__ == "__main__":
    main()