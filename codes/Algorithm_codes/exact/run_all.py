"""
Function: run_all.py
Role: Run the exact sequential lex-minimax MIP on a single T-ASR instance.
Produces detailed logs with timing, iteration info, and 4 figures.

Usage (depuis Algorithm_codes/) :
    python -m exact.run_all --instance toy
    python -m exact.run_all --instance setA-01 --time_limit 300
"""

import os
import sys
import json
import argparse
import datetime
import time
from pathlib import Path
from core.paths import CODES_DIR, RESULTS_DIR, LOGS_DIR
from core.paths import CHECKER_DIR as DEFAULT_CHECKER_DIR
import warnings

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings("ignore")

BLUE   = '#1F77B4'
ORANGE = '#FF7F0E'
RED    = '#D62728'
GREEN  = '#2CA02C'

###  Gurobi Setup

def _setup_gurobi():
    for p in [
        Path.home() / "gurobi.lic",
        CODES_DIR / "gurobi.lic",
    ]:
        if p.exists():
            os.environ["GRB_LICENSE_FILE"] = str(p)
            return str(p)
    return None

lic = _setup_gurobi()


def parse_args():
    parser = argparse.ArgumentParser(
        description="T-ASR — run exact MIP on a single instance"
    )
    parser.add_argument("--instance",    type=str, required=True)
    parser.add_argument("--data_dir",    type=str, default=None,
                        help="Override auto-detected data folder")
    parser.add_argument("--checker_dir", type=str, default=None)
    parser.add_argument("--time_limit",  type=int,   default=120)
    parser.add_argument("--max_iter",    type=int,   default=None)
    parser.add_argument("--mip_gap",     type=float, default=1e-4)
    return parser.parse_args()

#####################################
###  Folders
#####################################

def setup_folders(instance: str):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = LOGS_DIR / instance / timestamp
    res_dir = RESULTS_DIR / instance
    log_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    return log_dir, res_dir


#########################################################
###  Logger — duplicates stdout to file + adds timestamps
########################################################

class Logger:
    def __init__(self, log_file: Path):
        self.terminal = sys.stdout
        self.log      = open(log_file, "w", encoding="utf-8")
        self._t0      = time.perf_counter()

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def elapsed(self) -> str:
        s = time.perf_counter() - self._t0
        m = int(s // 60)
        return f"{m}m{s - m*60:.1f}s" if m > 0 else f"{s:.2f}s"

#####################################################
###  Logging helpers
#####################################################

def section(title: str, logger: Logger):
    w = 62
    print(f"\n{'='*w}")
    print(f"  {title}  [{logger.elapsed()}]")
    print(f"{'='*w}")

def subsection(title: str, logger: Logger):
    print(f"\n  -- {title}  [{logger.elapsed()}]")

def log_kv(key: str, value, width: int = 22):
    print(f"  {key:<{width}}: {value}")

def progress_bar(current: int, total: int, width: int = 30) -> str:
    filled = int(width * current / max(total, 1))
    bar    = '█' * filled + '░' * (width - filled)
    pct    = 100 * current / max(total, 1)
    return f"[{bar}] {current}/{total} ({pct:.0f}%)"

#####################################
###  Checker with JSON fix
#####################################

def run_checker_with_fix(net_f, tm_f, scen_f, sol_path, checker_bin):
    from core.checker import run_checker
    result = run_checker(
        net_file=net_f, tm_file=tm_f,
        scenario_file=scen_f, srpaths_file=sol_path,
        checker_bin=checker_bin,
    )
    if result is None:
        return None
    if result.max_load == float('inf'):
        print("  Checker returned inf — applying JSON fix ...")
        try:
            data   = json.loads(result.raw_output)
            loads  = sorted([obj["mlu"] for obj in data.get("objectives", [])],
                            reverse=True)
            if loads:
                result.loads_sorted = loads
                result.max_load     = loads[0]
                result.feasible     = True
                print(f"  JSON fix OK — corrected MLU = {result.max_load:.6f}")
        except Exception as e:
            print(f"  JSON fix failed: {e}")
    return result

#####################################
###  Plot helpers
#####################################

def plot_sorted_loads(loads_dict, inst, ax, title, color='blue', vline=None):
    from core.ecmp_from_scratch import sorted_load_vector
    vec     = sorted_load_vector(loads_dict, inst)
    nonzero = [v for v in vec if v > 1e-9]
    if not nonzero:
        ax.text(0.5, 0.5, 'No load', ha='center', va='center',
                transform=ax.transAxes)
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

######################################
###  Main
######################################

def main():
    args = parse_args()
    INSTANCE_NAME = args.instance
    CHECKER_DIR   = Path(args.checker_dir) if args.checker_dir else DEFAULT_CHECKER_DIR
    MAX_ITER      = args.max_iter
    TIME_LIMIT    = args.time_limit
    MIP_GAP       = args.mip_gap

    ## Detect data folder automatically based on instance name
    if args.data_dir:
        base = Path(args.data_dir)
    elif INSTANCE_NAME.startswith("setA"):
        base = CODES_DIR / "setA" / INSTANCE_NAME
    elif INSTANCE_NAME.startswith("ar"):
        base = CODES_DIR / "ar3_instance"
    elif INSTANCE_NAME == "toy":
        base = CODES_DIR / "toy_instance"
    else:
        base = CODES_DIR / f"{INSTANCE_NAME}_instance"
    net_f  = base / f"{INSTANCE_NAME}-net.json"
    tm_f   = base / f"{INSTANCE_NAME}-tm.json"
    scen_f = base / f"{INSTANCE_NAME}-scenario.json"

    ## Setup folders and logger
    log_dir, res_dir = setup_folders(INSTANCE_NAME)
    logger = Logger(log_dir / "run.log")
    sys.stdout = logger
    t_global = time.perf_counter()

    #####################################
    ### HEADER
    #####################################
    print("=" * 62)
    print("  T-ASR — Sequential Lex-Minimax MIP Solver")
    print(f"  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 62)
    log_kv("Instance",     INSTANCE_NAME)
    log_kv("Gurobi lic",   lic or "NOT FOUND")
    log_kv("time_limit",   f"{TIME_LIMIT}s per sub-problem")
    log_kv("max_iter",     MAX_ITER or "N = |A| x |T| (full)")
    log_kv("mip_gap",      MIP_GAP)
    log_kv("Data folder",  str(base))
    log_kv("Log folder",   str(log_dir))
    log_kv("Results",      str(res_dir))
    print("=" * 62)
    with open(log_dir / "config.json", "w") as f:
        json.dump({
            "instance": INSTANCE_NAME, "time_limit": TIME_LIMIT,
            "max_iter": MAX_ITER, "mip_gap": MIP_GAP,
            "data_dir": str(base), "gurobi_lic": lic,
        }, f, indent=2)

    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from core.instance import load_instance, summary
    from core.ecmp_from_scratch     import compute_all_splits, compute_loads, sorted_load_vector
    from .solver        import SequentialLexSolver, write_solution
    from core.checker   import find_checker

    #####################################
    ### SECTION 1 — LOAD INSTANCE
    #####################################
    section("1. Loading instance", logger)
    t0   = time.perf_counter()
    inst = load_instance(net_f, tm_f, scen_f)
    t_load = time.perf_counter() - t0
    N = len(inst.arc_ids) * len(inst.T)
    n_iter = MAX_ITER or N
    print(f"  {summary(inst)}")
    log_kv("N = |A| x |T|",  N)
    log_kv("Max lex iters",  n_iter)
    log_kv("Load time",      f"{t_load:.3f}s")
    log_kv("Interventions",  dict(inst.interv))
    log_kv("Budgets",        dict(inst.budget))

    #####################################
    ### SECTION 2 — ECMP & BASELINE
    #####################################
    section("2. ECMP coefficients & baseline", logger)
    t0 = time.perf_counter()
    r  = compute_all_splits(inst)
    t_ecmp = time.perf_counter() - t0
    trivial_wp     = {(d_id, t): [] for (d_id,_,_,_) in inst.demands
                      for t in inst.T}
    baseline_loads = compute_loads(trivial_wp, inst, r)
    baseline_vec   = sorted_load_vector(baseline_loads, inst)
    mlu_base       = baseline_vec[0]
    total_coeff = sum(
        len([v for v in r[t].values() if v > 0]) for t in inst.T
    )
    log_kv("ECMP time",       f"{t_ecmp:.3f}s")
    log_kv("Non-zero coeffs", total_coeff)
    log_kv("Baseline MLU",    f"{mlu_base:.6f}")

    ######################################
    ### SECTION 3 — SOLVER
    ######################################
    section("3. Sequential lex-minimax MIP solver", logger)
    log_kv("Strategy",     "Minimize λ'(1), then λ'(2), ..., up to λ'(N)")
    log_kv("Solver",       "Gurobi MIP")
    log_kv("Time limit",   f"{TIME_LIMIT}s per sub-problem")
    log_kv("MIP gap",      MIP_GAP)
    print()
    t_solver_start = time.perf_counter()
    solver = SequentialLexSolver(
        inst, r,
        time_limit = TIME_LIMIT,
        mip_gap    = MIP_GAP,
        max_iter   = MAX_ITER,
        verbose    = True,
    )
    waypoints, z_stars = solver.solve()
    t_solver = time.perf_counter() - t_solver_start

    ## Post-solve stats
    subsection("Solver statistics", logger)
    n_solved   = len(z_stars)
    converged  = n_solved < n_iter
    t_per_iter = t_solver / max(n_solved, 1)
    log_kv("Iterations done",   f"{n_solved} / {n_iter}")
    log_kv("Converged early",   "YES — all loads = 0" if converged else "NO — time limit")
    log_kv("Total solver time", f"{t_solver:.1f}s")
    log_kv("Avg time/iter",     f"{t_per_iter:.2f}s")
    log_kv("Progress",          progress_bar(n_solved, n_iter))
    if z_stars:
        log_kv("MLU (z*_1)",    f"{z_stars[0]:.6f}")
        log_kv("Final z*",      f"{z_stars[-1]:.6f}")
        total_improvement = z_stars[0] - z_stars[-1]
        log_kv("Total lex gain",f"{total_improvement:.6f}")

        print(f"\n  z* trajectory (first 10 iterations):")
        for k, z in enumerate(z_stars[:10], 1):
            delta = "" if k == 1 else f"  (Δ = {z_stars[k-2]-z:.6f})"
            bar   = '|' * max(0, int(z * 20))
            print(f"    k={k:>3}  z*={z:.6f}  {bar}{delta}")
        if len(z_stars) > 10:
            print(f"    ... ({len(z_stars) - 10} more iterations)")

    ## Save solution
    sol_path = res_dir / f"{INSTANCE_NAME}-srpaths.json"
    write_solution(waypoints, inst, sol_path)
    print(f"\n  Solution saved → {sol_path}")

    ################################################
    ### SECTION 4 — SOLUTION QUALITY
    ################################################
    section("4. Solution quality analysis", logger)
    sol_loads = compute_loads(waypoints, inst, r)
    vec_sol   = sorted_load_vector(sol_loads, inst)
    mlu_sol   = vec_sol[0]
    pct       = (mlu_base - mlu_sol) / max(mlu_base, 1e-12) * 100
    log_kv("MLU baseline",   f"{mlu_base:.6f}")
    log_kv("MLU solution",   f"{mlu_sol:.6f}")
    log_kv("MLU reduction",  f"{pct:+.2f}%")

    ## Lex vector stats
    improved  = sum(1 for b, s in zip(baseline_vec, vec_sol) if s < b - 1e-9)
    unchanged = sum(1 for b, s in zip(baseline_vec, vec_sol) if abs(b-s) <= 1e-9)
    degraded  = sum(1 for b, s in zip(baseline_vec, vec_sol) if s > b + 1e-9)
    first_div = next((i+1 for i, (b,s) in enumerate(zip(baseline_vec, vec_sol))
                      if abs(b-s) > 1e-9), None)
    subsection("Lexicographic vector analysis", logger)
    log_kv("Total components",  N)
    log_kv("Improved (s < b)",  f"{improved}  ({improved/N*100:.0f}%)")
    log_kv("Unchanged",         f"{unchanged}  ({unchanged/N*100:.0f}%)")
    log_kv("Degraded (s > b)",  f"{degraded}  ({degraded/N*100:.0f}%)")
    log_kv("First divergence",  f"rank {first_div}" if first_div else "none (identical)")

    ## Cumulative sum improvements
    subsection("Cumulative load sum reduction", logger)
    for k in [1, 5, 10, min(20, N), N]:
        bl_k  = sum(baseline_vec[:k])
        sol_k = sum(vec_sol[:k])
        r_k   = (bl_k - sol_k) / max(bl_k, 1e-9) * 100
        print(f"    Top-{k:<5}  Σbaseline={bl_k:.4f}  "
              f"Σsolution={sol_k:.4f}  reduction={r_k:+.2f}%")

    ## Waypoints summary
    subsection("Waypoints in solution", logger)
    wp_counts = {}
    for (d_id, src, tgt, _) in inst.demands:
        for t in inst.T:
            wps = waypoints.get((d_id, t), [])
            n   = len(wps)
            wp_counts[n] = wp_counts.get(n, 0) + 1
    total_dt = len(inst.demands) * len(inst.T)
    for n_wp, count in sorted(wp_counts.items()):
        bar = '█' * int(20 * count / total_dt)
        label = "IGP (no waypoint)" if n_wp == 0 else f"{n_wp} waypoint(s)"
        print(f"    {label:<22} : {count:>4} paths  {bar}")

    ###################################################
    ### SECTION 5 — CHECKER
    ###################################################
    section("5. Official Orange checker", logger)
    checker_result = None
    checker_bin    = find_checker(CHECKER_DIR)
    if checker_bin:
        log_kv("Checker binary", str(checker_bin))
        t0 = time.perf_counter()
        checker_result = run_checker_with_fix(
            net_f, tm_f, scen_f, sol_path, checker_bin
        )
        t_ck = time.perf_counter() - t0
        if checker_result:
            gap = abs(checker_result.max_load - mlu_sol)
            log_kv("Feasible",         checker_result.feasible)
            log_kv("MLU checker",      f"{checker_result.max_load:.6f}")
            log_kv("MLU Python",       f"{mlu_sol:.6f}")
            log_kv("Python/chkr gap",  f"{gap:.2e}  {'OK' if gap < 1e-4 else 'CHECK REQUIRED'}")
            log_kv("Checker time",     f"{t_ck:.2f}s")

            ck_path = res_dir / f"{INSTANCE_NAME}-checker-loads.json"
            with open(ck_path, "w") as f:
                json.dump({
                    "feasible":     checker_result.feasible,
                    "max_load":     checker_result.max_load,
                    "loads_sorted": checker_result.loads_sorted,
                }, f, indent=2)
    else:
        print(f"  Checker not found in {CHECKER_DIR}")

    ###################################################
    ### SECTION 6 — FIGURES
    ###################################################
    section("6. Saving figures", logger)
    t0 = time.perf_counter()

    # Figure 1 — Overlay lex vectors
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(baseline_vec, linestyle='--', linewidth=2,
            color=BLUE,   label=f'Baseline (MLU={mlu_base:.4f})')
    ax.plot(vec_sol,      linewidth=2,
            color=ORANGE, label=f'Solution (MLU={mlu_sol:.4f}, {pct:+.1f}%)')
    ax.fill_between(range(len(baseline_vec)), baseline_vec, alpha=0.10, color=BLUE)
    ax.fill_between(range(len(vec_sol)),      vec_sol,      alpha=0.20, color=ORANGE)
    ax.set_xlabel('Rank k (sorted arcs)')
    ax.set_ylabel('Load λ\'(k)')
    ax.set_title(f'{INSTANCE_NAME} — Sorted load vectors: baseline vs solution',
                 fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(res_dir / f'{INSTANCE_NAME}_loads_overlay.png', bbox_inches='tight')
    plt.close()

    # Figure 2 — Side-by-side bars
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    plot_sorted_loads(baseline_loads, inst, ax1,
                      title=f'Baseline (MLU={mlu_base:.4f})',
                      color=BLUE, vline=mlu_base)
    plot_sorted_loads(sol_loads, inst, ax2,
                      title=f'Solution (MLU={mlu_sol:.4f}, {pct:+.1f}%)',
                      color=ORANGE, vline=mlu_sol)
    ax1.set_ylabel('Load λ(a,t)')
    plt.suptitle(f'{INSTANCE_NAME} — Load distribution (style paper Fig.4)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(res_dir / f'{INSTANCE_NAME}_loads_comparison.png', bbox_inches='tight')
    plt.close()

    # Figure 3 — Full lex vector (+ checker if available)
    N_total = len(inst.arc_ids) * len(inst.T)
    pad     = lambda v: list(v) + [0.0] * (N_total - len(v))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(range(N_total), pad(baseline_vec), alpha=0.10, color=BLUE)
    ax.plot(range(N_total), pad(baseline_vec), linestyle='--', linewidth=1.5,
            color=BLUE,   label=f'Baseline (MLU={mlu_base:.4f})')
    ax.fill_between(range(N_total), pad(vec_sol), alpha=0.20, color=ORANGE)
    ax.plot(range(N_total), pad(vec_sol), linewidth=2,
            color=ORANGE, label=f'Solution (MLU={mlu_sol:.4f})')
    if (checker_result and checker_result.loads_sorted
            and checker_result.max_load != float('inf')):
        ax.plot(range(N_total), pad(checker_result.loads_sorted),
                linestyle=':', linewidth=1.5, color=RED,
                label=f'Checker (MLU={checker_result.max_load:.4f})')
    ax.set_xlabel('Rank k'); ax.set_ylabel('Load')
    ax.set_title(f'{INSTANCE_NAME} — Full lexicographic load vector',
                 fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set_xlim(0, N_total-1); ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(res_dir / f'{INSTANCE_NAME}_lex.png', bbox_inches='tight')
    plt.close()

    # Figure 4 — z* convergence
    if z_stars:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
        ks = range(1, len(z_stars) + 1)
        ax1.plot(ks, z_stars, color=ORANGE, marker='o', ms=4, linewidth=1.5)
        ax1.set_xlabel('Iteration k'); ax1.set_ylabel('z*_k (MLU at step k)')
        ax1.set_title(f'{INSTANCE_NAME} — z* convergence')
        ax1.grid(alpha=0.3)
        deltas = [0.0] + [z_stars[i-1]-z_stars[i] for i in range(1, len(z_stars))]
        ax2.bar(ks, deltas, color=GREEN, alpha=0.8)
        ax2.set_xlabel('k'); ax2.set_ylabel('Delta z*_k (marginal gain)')
        ax2.set_title(f'{INSTANCE_NAME} — Marginal improvement per iteration')
        ax2.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(res_dir / f'{INSTANCE_NAME}_convergence.png', bbox_inches='tight')
        plt.close()

    t_fig = time.perf_counter() - t0
    print(f"  Figures saved in {t_fig:.2f}s → {res_dir}/")
    for f in sorted(res_dir.glob("*.png")):
        print(f"    {f.name}")

    #########################################
    ### FINAL SUMMARY
    #########################################
    t_total = time.perf_counter() - t_global
    section("FINAL SUMMARY", logger)
    print(f"  {summary(inst)}")
    print()
    print(f"  Timing breakdown")
    print(f"    Load instance   : {t_load:.3f}s")
    print(f"    ECMP compute    : {t_ecmp:.3f}s")
    print(f"    MIP solver      : {t_solver:.1f}s")
    print(f"    Figures         : {t_fig:.2f}s")
    print(f"    TOTAL           : {t_total:.1f}s")
    print()
    print(f"  Objective")
    print(f"    MLU baseline    : {mlu_base:.6f}")
    print(f"    MLU solution    : {mlu_sol:.6f}  ({pct:+.2f}%)")
    if z_stars:
        print(f"    Lex iterations  : {n_solved} / {n_iter}"
              f"  ({'converged' if converged else 'time limit'})")
        print(f"    Avg time/iter   : {t_per_iter:.2f}s")
    print()
    print(f"  Components improved : {improved}/{N}  ({improved/N*100:.0f}%)")
    print(f"  First divergence    : rank {first_div}" if first_div else
          f"  First divergence    : none (identical vectors)")
    if checker_result:
        print(f"  Checker MLU         : {checker_result.max_load:.6f}  "
              f"(feasible={checker_result.feasible})")
    print()
    print(f"  Results  → {res_dir}/")
    print(f"  Log      → {log_dir}/run.log")
    print("=" * 62)

    ## Save summary JSON
    with open(res_dir / f"{INSTANCE_NAME}-summary.json", "w") as f:
        json.dump({
            "instance":          INSTANCE_NAME,
            "n_nodes":           len(inst.nodes),
            "n_arcs":            len(inst.arc_ids),
            "n_demands":         len(inst.demands),
            "n_times":           len(inst.T),
            "max_seg":           inst.max_seg,
            "N":                 N,
            "mlu_baseline":      mlu_base,
            "mlu_solution":      mlu_sol,
            "reduction_pct":     pct,
            "n_iterations":      n_solved,
            "converged":         converged,
            "t_solver_s":        round(t_solver, 2),
            "t_total_s":         round(t_total, 2),
            "lex_improved":      improved,
            "lex_unchanged":     unchanged,
            "lex_degraded":      degraded,
            "first_divergence":  first_div,
            "checker_mlu":       checker_result.max_load if checker_result else None,
            "feasible":          checker_result.feasible if checker_result else None,
        }, f, indent=2)


if __name__ == "__main__":
    main()