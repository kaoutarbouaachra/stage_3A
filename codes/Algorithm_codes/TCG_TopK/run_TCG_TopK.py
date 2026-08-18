"""
Function: run_TCG_TopK.py
Role: the lexicographic pair-path Column Generation algorithm for T-ASR.
NB: (this version implements the algorithm with heuristic pricing of the diagonal + off-diagonal phases)


SCOPE: assumes h = 2 

Usage (from Algorithm_codes/):
    python -m TCG_topk.run_TCG_TopK --instance setA-01 --K 5
    python -m TCG_topk.run_TCG_TopK setAP-001 --data-dir=../setAP/ --K 5
    python -m TCG_topk.run_TCG_TopK --instance setA-10 --K 3 --time-limit 900
"""

import argparse
import json
import os
import sys
import time
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from core.paths import CODES_DIR, RESULTS_DIR, LOGS_DIR, CHECKER_DIR
from core.dataset import add_dataset_args, resolve_paths, instance_name

# Official algorithm identifier -- used for both the results and log
# subfolders, and stamped into every summary JSON, so that outputs from
# different TCG variants never collide and are self-identifying.
# Uppercase, to match ALEXA / STELA / CARLA / TCG_DIAG / TCG_EXACT.
ALGO_NAME = "TCG_TOPK"


### Gurobi setup
def _setup_gurobi():
    for p in [Path.home() / "gurobi.lic",
              CODES_DIR / "gurobi.lic"]:
        if p.exists():
            os.environ["GRB_LICENSE_FILE"] = str(p)
            return str(p)
    return None


_lic = _setup_gurobi()


# ---------------------------------------------------------------------------
# Tee: duplicates sys.stdout to a log file
# ---------------------------------------------------------------------------
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
        # Required for subprocesses that call isatty / fileno
        return sys.__stdout__.fileno()


def _setup_logging(name: str) -> tuple:
    """
    Creates logs/<ALGO_NAME>/, opens the log file and redirects sys.stdout to a
    _Tee. Returns (log_path, log_fh) so the caller can close cleanly at the end.
    """
    log_dir = LOGS_DIR / ALGO_NAME
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{name}_{timestamp}.log"
    log_fh = open(log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_fh)
    return log_path, log_fh


def _teardown_logging(log_fh):
    """Restores sys.stdout and closes the log file."""
    sys.stdout = sys.__stdout__
    try:
        log_fh.close()
    except Exception:
        pass


def _release_gurobi():
    """
    Release the WLS license token. gurobipy keeps a hidden default environment
    alive for the whole process; disposing it explicitly returns the token as
    soon as the run is over.
    """
    try:
        import gurobipy as gp
        gp.disposeDefaultEnv()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="T-ASR -- Rank-by-Rank Pair-Path CG (TCG-TopK)"
    )
    add_dataset_args(p)

    p.add_argument("--checker-dir", "--checker_dir",
                   dest="checker_dir", default=None,
                   help="Checker binary directory")
    p.add_argument("--K", type=int, default=5,
                   help="Number of lex ranks to certify (default 5)")
    p.add_argument("--time-limit", "--time_limit",
                   dest="time_limit", type=float, default=900.0,
                   help="Global cap, in seconds")
    p.add_argument("--lp-time", "--lp_time",
                   dest="lp_time", type=float, default=90.0,
                   help="Max time per LP-RMP optimize() call")
    p.add_argument("--mip-time", "--mip_time",
                   dest="mip_time", type=float, default=180.0,
                   help="Max time per rank MIP")
    p.add_argument("--max-cg-iter", "--max_cg_iter",
                   dest="max_cg_iter", type=int, default=20,
                   help="Max CG iterations per rank")
    p.add_argument("--topk-cols", "--topk_cols",
                   dest="topk_cols", type=int, default=3,
                   help="Columns added per demand per iteration")
    p.add_argument("--rc-tol", "--rc_tol",
                   dest="rc_tol", type=float, default=-1e-6)
    return p.parse_args()


#############################################
### LEX RMP (master problem for a given rank)
#############################################

class LexRMP:
    """
    Function: LexRMP
    Role: Restricted Master Problem for rank k in the lexicographic algorithm.

    Description:
        The objective changes at each rank.
        The inherited constraints (Tj, Nj) for j < k are added cumulatively.
    """

    def __init__(self, inst, r):
        self.inst = inst
        self.r = r
        self.demands_info = {
            d_id: (src, tgt, vols)
            for (d_id, src, tgt, vols) in inst.demands
        }
        self.arc_time_pairs = [
            (l['id'], t)
            for l in inst.links.values()
            for t in inst.T
            if l['id'] not in inst.interv.get(t, set())
        ]

        # Persistent caches keyed by (d_id, ti). Gamma and beta are intrinsic to
        # the trajectory: independent of the duals and of the rank k. The pool is
        # append-only, so ti is a stable identifier.
        self._gamma_cache: Dict[Tuple[int, int], Dict[Tuple[int, int], float]] = {}
        self._beta_cache: Dict[Tuple[int, int], int] = {}

    def _precompute_gamma(self, pool):
        """
        Function: _precompute_gamma
        Role: Computes the contributions Gamma_{d,a,t} for the columns that are
              not cached yet, and returns the full caches.

        Only NEW columns are evaluated: the previous version recomputed the whole
        pool at every LP build, which made each CG iteration cost O(pool) and the
        rank cost O(pool^2).
        """
        from .cg_master0 import trajectory_load_contribution, trajectory_budget_cost

        for d_id, (src, tgt, vols) in self.demands_info.items():
            for ti, traj in enumerate(pool.get(d_id)):
                key = (d_id, ti)
                if key in self._gamma_cache:
                    continue
                self._gamma_cache[key] = trajectory_load_contribution(
                    traj, src, tgt, vols, self.inst, self.r)
                self._beta_cache[key] = trajectory_budget_cost(traj, src, tgt)

        return self._gamma_cache, self._beta_cache

    def _check_pool(self, pool):
        for d_id in self.demands_info:
            if not pool.get(d_id):
                raise ValueError(
                    f"demand {d_id} has no column; seed the trivial pool first")

    def _build_model(self, pool, rank_k, z_stars,
                     binary=False, time_limit=120.0, verbose=False):
        """
        Function: _build_model
        Role: Builds and solves the RMP at rank k.
        binary=False -> LP relaxation (for duals); requires proven optimality.
        binary=True  -> MIP (for the solution); any feasible incumbent accepted.
        Returns None on failure.

        The Gurobi model is disposed before returning: everything needed
        downstream (.X, .Pi) is copied into `result` first, so disposing here
        keeps the process footprint flat across the many builds of a run.
        """
        import gurobipy as gp
        from gurobipy import GRB

        self._check_pool(pool)
        gamma_cache, beta_cache = self._precompute_gamma(pool)
        m = gp.Model(f"RMP_rank{rank_k}_{'mip' if binary else 'lp'}")
        m.setParam("OutputFlag", 0 if not verbose else 1)
        m.setParam("TimeLimit", time_limit)
        vtype = GRB.BINARY if binary else GRB.CONTINUOUS

        try:
            ### Routing variables z_{d,ti}
            z = {}
            for d_id in self.demands_info:
                for ti in range(len(pool.get(d_id))):
                    z[(d_id, ti)] = m.addVar(
                        lb=0.0, ub=1.0, vtype=vtype,
                        name=f"z_{d_id}_{ti}")

            ### Current rank threshold variable
            z_thresh = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="z_thresh")

            ### Load variables lambda(a,t)
            lam = {}
            for (arc_id, t) in self.arc_time_pairs:
                lam[(arc_id, t)] = m.addVar(
                    lb=0.0, vtype=GRB.CONTINUOUS,
                    name=f"lam_{arc_id}_{t}")

            ### Surplus variables y^(j)_{a,t} for j = 1..k
            # y[(j, arc_id, t)] = 1 if lambda(a,t) > rank-j threshold
            y = {}
            for j in range(1, rank_k + 1):
                for (arc_id, t) in self.arc_time_pairs:
                    y[(j, arc_id, t)] = m.addVar(
                        lb=0.0, ub=1.0,
                        vtype=GRB.BINARY if binary else GRB.CONTINUOUS,
                        name=f"y_{j}_{arc_id}_{t}")

            ### Selection constraint (S)
            cons_S = {}
            for d_id in self.demands_info:
                trajs = pool.get(d_id)
                cons_S[d_id] = m.addConstr(
                    gp.quicksum(z[(d_id, ti)]
                                for ti in range(len(trajs))) == 1,
                    name=f"S_{d_id}")

            ### Constraint (L): load definition
            cons_L = {}
            for (arc_id, t) in self.arc_time_pairs:
                expr = gp.LinExpr()
                for (d_id, ti), g in gamma_cache.items():
                    val = g.get((arc_id, t), 0.0)
                    if val > 1e-12:
                        expr.add(z[(d_id, ti)], val)
                cons_L[(arc_id, t)] = m.addConstr(
                    lam[(arc_id, t)] == expr,
                    name=f"L_{arc_id}_{t}")

            ### Constraint (B): budget
            cons_B = {}
            for t in self.inst.T_star:
                kappa = self.inst.budget.get(t)
                if kappa is None:
                    continue
                expr = gp.LinExpr()
                for (d_id, ti), b in beta_cache.items():
                    if b > 0:
                        expr.add(z[(d_id, ti)], float(b))
                if expr.size() > 0:
                    cons_B[t] = m.addConstr(expr <= kappa, name=f"B_{t}")

            z1_star = z_stars[0] if z_stars else None

            ### Inherited constraints (Tj, Nj) for j < k
            #
            #   (Tj): lambda(a,t) <= z_j* + z_1* . y^(j)_{a,t}
            #   (Nj): Sum_{a,t} y^(j)_{a,t} <= j - 1
            #
            #   Interpretation: at most j-1 arcs may exceed the rank-j threshold,
            #   which enforces lambda'_j <= z_j*.
            #
            cons_Tj = {}
            for j in range(1, rank_k):
                zj = z_stars[j - 1]
                for (arc_id, t) in self.arc_time_pairs:
                    cons_Tj[(j, arc_id, t)] = m.addConstr(
                        lam[(arc_id, t)] <= zj + z1_star * y[(j, arc_id, t)],
                        name=f"T{j}_{arc_id}_{t}")
                m.addConstr(
                    gp.quicksum(y[(j, arc_id, t)]
                                for (arc_id, t) in self.arc_time_pairs) <= j - 1,
                    name=f"N{j}")

            ### Current rank k constraints
            #
            #   (Tk): lambda(a,t) <= z_thresh + z_1* . y^(k)_{a,t}
            #   (Nk): Sum y^(k) <= k - 1
            #
            #   For rank 1 (k=1): no y variables, only lambda(a,t) <= z_thresh
            #   (this is the definition of the MLU)
            #
            cons_Tk = {}
            if rank_k == 1:
                # Rank 1: directly minimize the MLU
                for (arc_id, t) in self.arc_time_pairs:
                    cons_Tk[(arc_id, t)] = m.addConstr(
                        lam[(arc_id, t)] <= z_thresh,
                        name=f"Tk_{arc_id}_{t}")
            else:
                # Rank k >= 2: constraint with surplus variable
                for (arc_id, t) in self.arc_time_pairs:
                    cons_Tk[(arc_id, t)] = m.addConstr(
                        lam[(arc_id, t)] <=
                        z_thresh + z1_star * y[(rank_k, arc_id, t)],
                        name=f"Tk_{arc_id}_{t}")
                m.addConstr(
                    gp.quicksum(y[(rank_k, arc_id, t)]
                                for (arc_id, t) in self.arc_time_pairs)
                    <= rank_k - 1,
                    name="Nk")

            ### Objective: minimize z_thresh (the current rank's threshold)
            m.setObjective(z_thresh, GRB.MINIMIZE)
            m.optimize()

            ### Status checks
            if not binary:
                # LP duals are only valid at proven optimality. A TIME_LIMIT
                # interior point yields duals that do not certify anything, and
                # pricing on them silently produces wrong columns.
                if m.Status != GRB.OPTIMAL or m.SolCount == 0:
                    return None
            else:
                # MIP: accept any feasible incumbent (e.g. under the time limit).
                if m.SolCount == 0:
                    return None

            ### Extract solution and duals
            result = {
                "z_k":    z_thresh.X,
                "status": m.Status,
            }
            if not binary:
                # Duals for the pricing
                result["pi"] = {
                    d_id: c.Pi for d_id, c in cons_S.items()}
                result["mu_k"] = {
                    (arc_id, t): c.Pi
                    for (arc_id, t), c in cons_Tk.items()}
                result["eta_j"] = {
                    (j, arc_id, t): c.Pi
                    for (j, arc_id, t), c in cons_Tj.items()}
                result["nu"] = {
                    t: c.Pi for t, c in cons_B.items()}

                # Stratified duals:
                mu_bar = {}
                for (arc_id, t) in self.arc_time_pairs:
                    val = result["mu_k"].get((arc_id, t), 0.0)
                    for j in range(1, rank_k):
                        val += result["eta_j"].get((j, arc_id, t), 0.0)
                    mu_bar[(arc_id, t)] = val
                result["mu_bar"] = mu_bar
            else:
                # Routing solution
                sol = {}
                for d_id in self.demands_info:
                    for ti, traj in enumerate(pool.get(d_id)):
                        if z[(d_id, ti)].X > 0.5:
                            sol[d_id] = traj
                            break
                result["solution"] = sol

            return result
        finally:
            # Free the model whatever the exit path (return / exception).
            m.dispose()


####################
###  MAIN
####################

def _run(args, name, net_f, tm_f, scen_f):
    # Results live under results/<instance>/<ALGO_NAME>/ so that different
    # TCG variants never overwrite each other's output on the same instance.
    res_dir = RESULTS_DIR / name / ALGO_NAME
    res_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("  T-ASR -- Pair-Path Column Generation  (Rank-by-Rank Lex)")
    print(f"  Algorithm    : TCG-TopK  (folder: {ALGO_NAME})")
    print(f"  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 68)
    print(f"  Instance     : {name}")
    print(f"  Dataset      : {net_f.parent}")
    print(f"  K ranks      : {args.K}")
    print(f"  time_limit   : {args.time_limit}s")
    print(f"  lp_time/rank : {args.lp_time}s")
    print(f"  mip_time/rank: {args.mip_time}s")
    print(f"  topk_cols    : {args.topk_cols}")
    print(f"  Results dir  : {res_dir}")
    print("=" * 68)

    from core.instance import load_instance, summary
    from core.ecmp import compute_all_splits, compute_loads, sorted_load_vector
    from stela.solver import write_solution
    from .cg_master0 import TrajectoryPool
    from .pricing_dp_heuristic import price_demand_topk

    t_global = time.perf_counter()

    # Load the instance
    inst = load_instance(net_f, tm_f, scen_f)
    print(f"\n  {summary(inst)}")

    if len(inst.T) != 2:
        raise ValueError(
            f"this driver is specialised to h = 2, but instance '{name}' has "
            f"h = {len(inst.T)}. Use an exact solver (ALEXA / STELA / CARLA) "
            f"on this instance.")

    r = compute_all_splits(inst)
    demands_info = {
        d_id: (src, tgt, vols)
        for (d_id, src, tgt, vols) in inst.demands
    }

    # Baseline solution (IGP)
    trivial = {(d_id, t): []
               for (d_id, _, _, _) in inst.demands for t in inst.T}
    base_vec = sorted_load_vector(compute_loads(trivial, inst, r), inst)
    print(f"  MLU baseline  : {base_vec[0]:.6f}")
    print(f"  Top-5 baseline: {[round(x,4) for x in base_vec[:5]]}")

    # Initial pool: trivial trajectories (IGP, IGP)
    # We start with the direct path for each demand in both periods
    pool = TrajectoryPool()
    for d_id in demands_info:
        pool.add(d_id, ([], []))
    print(f"\n  Initial pool : {pool.total_size()} trajectories")

    # Initialize the RMP solver for the lexicographic algorithm
    lex_rmp = LexRMP(inst, r)

    # Accumulated results
    z_stars = []          # Certified z_k* values at each rank
    best_solution = {}    # Best routing solution found
    best_vec = list(base_vec)

    # Log table
    print(f"\n  {'Rank':>5} {'CGit':>5} {'Pool':>7} {'LP_bound':>10}"
          f" {'z_k*':>10} {'Top3_sol':>22} {'Time':>7}")
    print(f"  {'-'*5} {'-'*5} {'-'*7} {'-'*10}"
          f" {'-'*10} {'-'*22} {'-'*7}")

    #  LOOP OVER LEXICOGRAPHIC RANKS
    for rank_k in range(1, args.K + 1):
        elapsed = time.perf_counter() - t_global
        if elapsed > args.time_limit:
            print(f"\n  Time limit reached before rank {rank_k}.")
            break
        t_rank_start = time.perf_counter()
        lp_bound = float('inf')
        cg_iters = 0

        # Internal CG loop for this rank k: add columns to the shared pool
        # until convergence
        for cg_iter in range(args.max_cg_iter):
            elapsed = time.perf_counter() - t_global
            if elapsed > args.time_limit:
                break
            remaining_lp = min(
                args.lp_time,
                args.time_limit - elapsed - 10)
            if remaining_lp < 5:
                break

            ## Solve LP-RMP(k) to get the current bound and duals
            lp_result = lex_rmp._build_model(
                pool, rank_k, z_stars,
                binary=False,
                time_limit=remaining_lp)
            if lp_result is None:
                break

            lp_bound = lp_result["z_k"]
            cg_iters += 1

            ## Extract stratified duals
            #
            #   mu_bar_k(a,t) = mu_k(a,t) + Sum_{j<k} eta_j(a,t)
            #
            #   These dual values reflect the cumulative pressure on each arc
            #   from all previously certified ranks.
            #
            mu_bar = lp_result["mu_bar"]
            pi_d = lp_result["pi"]
            nu1 = lp_result["nu"].get(1, 0.0)

            ## Pricing: find useful pairs (p0, p1)
            #
            #   Reduced cost of (p0, p1) for demand d:
            #   rc = -pi_d
            #       + Sum_{(i,j) in segs(p0)} theta_d(i,j,0)
            #       + Sum_{(i,j) in segs(p1)} theta_d(i,j,1)
            #       - nu(1) x dist(p0,p1)
            #
            #   theta_d(i,j,t) = Sum_a (-mu_bar_k(a,t)) x nu(d,t)/c(a) x r(i,j,a,t)
            #
            added_count = 0
            for d_id, (src, tgt, vols) in demands_info.items():
                if time.perf_counter() - t_global > args.time_limit:
                    break
                pi = pi_d.get(d_id, 0.0)
                results = price_demand_topk(
                    d_info=(d_id, src, tgt, vols),
                    inst=inst, r=r,
                    mu=mu_bar,        # stratified duals
                    pi_d=pi,
                    nu_1=nu1,
                    top_k=args.topk_cols,
                    rc_tol=args.rc_tol,
                )
                for res in results:
                    if pool.add(d_id, res.trajectory):
                        added_count += 1

            ## CG stopping criterion
            if added_count == 0:
                break   # optimal pool for this rank -> go to the MIP

        # Rank-k MIP: get z_k*
        elapsed = time.perf_counter() - t_global
        remaining_mip = min(
            args.mip_time,
            args.time_limit - elapsed - 5)
        if remaining_mip < 5:
            print(f"\n  Not enough time for MIP at rank {rank_k}.")
            break
        mip_result = lex_rmp._build_model(
            pool, rank_k, z_stars,
            binary=True,
            time_limit=remaining_mip)
        if mip_result is None:
            print(f"\n  MIP found no feasible incumbent at rank {rank_k}.")
            break
        z_k_star = mip_result["z_k"]
        sol_k = mip_result["solution"]
        z_stars.append(z_k_star)

        # Update the best solution
        if sol_k:
            best_solution = sol_k
            # Compute the lex vector of the current solution
            wp_cur = {}
            for d_id, traj in sol_k.items():
                wp_cur[(d_id, 0)] = list(traj[0])
                wp_cur[(d_id, 1)] = list(traj[1])
            cur_vec = sorted_load_vector(
                compute_loads(wp_cur, inst, r), inst)
            best_vec = cur_vec
            top3_str = str([round(x, 4) for x in cur_vec[:3]])
        else:
            top3_str = "N/A"
        t_rank = time.perf_counter() - t_rank_start
        print(f"  {rank_k:>5} {cg_iters:>5} {pool.total_size():>7}"
              f" {lp_bound:>10.6f} {z_k_star:>10.6f}"
              f" {top3_str:>22} {t_rank:>6.1f}s")

    ########################################################################
    ### Post-processing: build the final solution, verify, and save results
    ########################################################################

    # Build the final solution from the best MIP solution found across ranks
    waypoints = {}
    for d_id, traj in best_solution.items():
        waypoints[(d_id, 0)] = list(traj[0])
        waypoints[(d_id, 1)] = list(traj[1])
    if waypoints:
        final_vec = sorted_load_vector(
            compute_loads(waypoints, inst, r), inst)
    else:
        final_vec = base_vec
    mlu_sol = final_vec[0]
    pct = (base_vec[0] - mlu_sol) / max(base_vec[0], 1e-12) * 100
    print(f"\n  {'='*64}")
    print(f"  Algorithm       : TCG-TopK")
    print(f"  MLU baseline    : {base_vec[0]:.6f}")
    print(f"  MLU solution    : {mlu_sol:.6f}  ({pct:+.2f}%)")
    print(f"  Certified ranks : {len(z_stars)} / {args.K} requested")
    print(f"  Top-5 baseline  : {[round(x,4) for x in base_vec[:5]]}")
    print(f"  Top-5 solution  : {[round(x,4) for x in final_vec[:5]]}")
    print(f"  Pool size       : {pool.total_size()}")

    # Save the solution in the format expected by the checker
    sol_path = res_dir / f"{name}-srpaths.json"
    if waypoints:
        write_solution(waypoints, inst, sol_path)
        print(f"  Saved -> {sol_path}")

    # Verify with the checker
    checker_valid = None
    checker_mlu = None
    try:
        from core.checker import find_checker, run_checker
        checker_dir = Path(args.checker_dir) if args.checker_dir else CHECKER_DIR
        checker_bin = find_checker(checker_dir)
        if checker_bin and waypoints:
            print(f"\n  Running checker...")
            result = run_checker(
                net_file=net_f, tm_file=tm_f,
                scenario_file=scen_f, srpaths_file=sol_path,
                checker_bin=checker_bin,
            )
            if result:
                checker_valid = result.valid
                checker_mlu = result.max_load
                print(f"  Feasible    : {result.valid}")
                print(f"  MLU checker : {result.max_load:.6f}")
    except Exception as e:
        print(f"  Checker: {e}")

    # Save a results summary in JSON -- self-identifying via "algorithm"
    t_total = time.perf_counter() - t_global
    summary_path = res_dir / f"{name}-summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "algorithm":     ALGO_NAME,
            "instance":      name,
            "K":             args.K,
            "ranks_solved":  len(z_stars),
            "mlu_baseline":  base_vec[0],
            "mlu_solution":  mlu_sol,
            "reduction_pct": pct,
            "z_stars":       z_stars,
            "top5_baseline": [round(x, 6) for x in base_vec[:5]],
            "top5_solution": [round(x, 6) for x in final_vec[:5]],
            "pool_size":     pool.total_size(),
            "time_limit":    args.time_limit,
            "time_s":        round(t_total, 2),
            "checker_valid": checker_valid,
            "checker_mlu":   checker_mlu,
        }, f, indent=2)
    print(f"  Summary saved -> {summary_path}")

    print(f"\n  Total time      : {t_total:.1f}s")
    print("=" * 68)


def main():
    args = parse_args()
    name = instance_name(args)

    # Resolve BEFORE opening the log: resolve_paths exits on a missing instance,
    # and an empty log for a run that never started is just noise.
    net_f, tm_f, scen_f = resolve_paths(args)

    log_path, log_fh = _setup_logging(name)
    print(f"  Log          : {log_path}")

    # try/finally: the WLS token is released and the log closed even if the run
    # raises (failed rank, infeasible model, KeyboardInterrupt...).
    try:
        _run(args, name, net_f, tm_f, scen_f)
    finally:
        _release_gurobi()
        _teardown_logging(log_fh)


if __name__ == "__main__":
    main()