"""
Function: run_tcg_diag.py  -- unified rank-by-rank pair-path CG driver (h = 2, two time steps)
Role: Wires the lexicographic loop to the CORRECTED modules:
        * pricing_dp.py  -> exact diagonal Bellman-Ford + heuristic off-diagonal,
                            signature price_demand_topk(..., psi_1, ctx=...).
        * cg_master.py   -> trajectory_budget_cost (module-level) is reused; the load
                            contribution Gamma is computed here with an indexed,
                            cached routine (the old trajectory_load_contribution was
                            removed during the master fixes).

Algorithm :
  Shared pool grows across all lex ranks. For k = 1..K:
    1. Inner CG: solve LP-RMP(k) with inherited rows (Tj, Nj) for j<k, read
       stratified duals mu_bar_k = mu_k + sum_{j<k} eta_j, price all demands,
       add improving trajectories, repeat until no negative reduced cost.
    2. Solve MIP(k) on the pool -> certified z_k*.
    3. Inherit z_k* as a pinned row for rank k+1.

Dual signs: sigma_d (free), mu_k <= 0, eta_j <= 0, psi(t) <= 0. The pricing uses
(-mu_bar) as the per-(a,t) pressure and rho = -psi(1) as the per-shared-segment
budget penalty. Passing mu = mu_bar is correct AT LP OPTIMALITY (rho_L = -mu_bar),
which is why the LP must be solved to proven optimality before reading duals.

Usage (from Algorithm_codes/):
    python -m TCG_diag.run_tcg_diag --instance setA-01 --K 5
    python -m TCG_diag.run_tcg_diag --instance setA-10 --K 3 --time_limit 900
"""

import argparse
import json
import os
import sys
import time
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from core.paths import CODES_DIR, RESULTS_DIR, CHECKER_DIR

# Official algorithm identifier -- used for both the results and log
# subfolders, and stamped into every summary JSON, so that outputs from
# different TCG variants never collide and are self-identifying.
ALGO_NAME = "TCG_diag"

# Log directory (Algorithm_codes/logs/TCG_diag/)
LOGS_DIR = RESULTS_DIR.parent / "logs"


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
# Tee : duplicates sys.stdout to a log file
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


def _setup_logging(instance_name: str) -> tuple:
    """
    Creates logs/TCG_diag/, opens the log file and redirects
    sys.stdout to a _Tee.  Returns (log_path, log_fh) so the caller
    can close cleanly at the end.
    """
    log_dir = LOGS_DIR / ALGO_NAME
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = log_dir / f"{instance_name}_{timestamp}.log"
    log_fh    = open(log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_fh)
    return log_path, log_fh


def _teardown_logging(log_fh):
    """Restores sys.stdout and closes the log file."""
    sys.stdout = sys.__stdout__
    log_fh.close()


# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="T-ASR -- Rank-by-Rank Pair-Path CG (TCG-Diag)"
    )
    p.add_argument("--instance",    required=True)
    p.add_argument("--data_dir",    default=None)
    p.add_argument("--checker_dir", default="../checker")
    p.add_argument("--K",           type=int,   default=5,
                   help="Number of lex ranks to certify (default 5)")
    p.add_argument("--time_limit",  type=float, default=900.0)
    p.add_argument("--lp_time",     type=float, default=90.0,
                   help="Max time per LP-RMP")
    p.add_argument("--mip_time",    type=float, default=180.0,
                   help="Max time per rank MIP")
    p.add_argument("--max_cg_iter", type=int,   default=20,
                   help="Max CG iterations per rank")
    p.add_argument("--topk_cols",   type=int,   default=3,
                   help="Columns added per demand per iteration")
    p.add_argument("--rc_tol",      type=float, default=-1e-6)
    return p.parse_args()

def detect_base(name, override=None):
    if override:
        return Path(override).resolve()

    if name.startswith("setA"):
        return CODES_DIR / "setA" / name

    if name == "toy":
        return CODES_DIR / "toy_instance"

    return CODES_DIR / f"{name}_instance"


###############################################
###  LEX RMP  (master problem for a given rank)
##############################################
class LexRMP:
    """
    Function: LexRMP
    Role: Restricted Master Problem for rank k of the lex algorithm.

    The objective changes each rank; the inherited rows (Tj, Nj) for j < k are
    added cumulatively. Load is modelled with explicit lambda(a,t) variables and
    a (L) equality lambda = sum Gamma z, so the pricing pressure is rho_L = -mu_bar.
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

        # Precompute ONCE: capacities, and split coefficients indexed by (u,v)
        # per period. Same structure as the corrected RMPSolver, so Gamma is
        # cheap and the all-pairs requirement is checked (KeyError on a missing
        # waypoint hop instead of a silent zero).
        self.capa: Dict[int, float] = {
            l['id']: l['capacity'] for l in inst.links.values()
        }
        self.r_index: Dict[int, Dict[Tuple[int, int], List[Tuple[int, float]]]] = {}
        for t, r_t in self.r.items():
            idx: Dict[Tuple[int, int], List[Tuple[int, float]]] = {}
            for (u, v, arc_id), coeff in r_t.items():
                if coeff > 1e-12:
                    idx.setdefault((u, v), []).append((arc_id, coeff))
            self.r_index[t] = idx

        # Persistent caches keyed by (d_id, ti); Gamma/beta are intrinsic to the
        # trajectory (independent of the dual values and of the rank k).
        self._gamma_cache: Dict[Tuple[int, int], Dict[Tuple[int, int], float]] = {}
        self._beta_cache: Dict[Tuple[int, int], int] = {}

    # ----------------------------------------------------------------------

    def _gamma(self, d_id, ti, traj):
        """Gamma_{d,a,t}(traj) for all (a,t), cached by (d_id, ti)."""
        key = (d_id, ti)
        cached = self._gamma_cache.get(key)
        if cached is not None:
            return cached
        src, tgt, vols = self.demands_info[d_id]
        contrib: Dict[Tuple[int, int], float] = {}
        for t, path in enumerate(traj):                 # h = 2 -> t = 0, 1
            nu_dt = vols[t]
            if nu_dt == 0.0:
                continue
            nodes = [src] + list(path) + [tgt]
            idx_t = self.r_index.get(t, {})
            for (i, j) in zip(nodes[:-1], nodes[1:]):
                arcs = idx_t.get((i, j))
                if arcs is None:
                    raise KeyError(
                        f"no split coefficients r({i},{j},.,{t}); "
                        f"is r computed for ALL node pairs in G_t?")
                for (arc_id, coeff) in arcs:
                    k = (arc_id, t)
                    contrib[k] = contrib.get(k, 0.0) + (nu_dt / self.capa[arc_id]) * coeff
        self._gamma_cache[key] = contrib
        return contrib

    def _beta(self, d_id, ti, traj):
        """beta = dist(p0, p1) for the single h = 2 transition, cached."""
        key = (d_id, ti)
        cached = self._beta_cache.get(key)
        if cached is not None:
            return cached
        from .cg_master import trajectory_budget_cost
        src, tgt, _ = self.demands_info[d_id]
        b = trajectory_budget_cost(traj, src, tgt)
        self._beta_cache[key] = b
        return b

    def _check_pool(self, pool):
        for d_id in self.demands_info:
            if not pool.get(d_id):
                raise ValueError(f"demand {d_id} has no column; seed IGP first")

    # ----------------------------------------------------------------------

    def _build_model(self, pool, rank_k, z_stars,
                     binary=False, time_limit=120.0, verbose=False):
        """
        Build and solve the RMP at rank k.
          binary=False -> LP relaxation (duals for pricing); requires OPTIMAL.
          binary=True  -> MIP (routing solution); requires a feasible incumbent.
        Returns None on failure.
        """
        import gurobipy as gp
        from gurobipy import GRB

        self._check_pool(pool)
        m = gp.Model(f"RMP_rank{rank_k}_{'mip' if binary else 'lp'}")
        m.setParam("OutputFlag", 1 if verbose else 0)
        m.setParam("TimeLimit", time_limit)
        vtype = GRB.BINARY if binary else GRB.CONTINUOUS

        ## Routing variables z_{d,ti}
        z = {}
        for d_id in self.demands_info:
            for ti in range(len(pool.get(d_id))):
                z[(d_id, ti)] = m.addVar(lb=0.0, ub=1.0, vtype=vtype,
                                         name=f"z_{d_id}_{ti}")

        ## Current-rank threshold variable
        z_thresh = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="z_thresh")

        ## Load variables lambda(a,t)
        lam = {at: m.addVar(lb=0.0, vtype=GRB.CONTINUOUS,
                            name=f"lam_{at[0]}_{at[1]}")
               for at in self.arc_time_pairs}

        ## Exceedance variables y^(j)_{a,t}, j = 1..k
        y = {}
        for j in range(1, rank_k + 1):
            for at in self.arc_time_pairs:
                y[(j,) + at] = m.addVar(
                    lb=0.0, ub=1.0,
                    vtype=GRB.BINARY if binary else GRB.CONTINUOUS,
                    name=f"y_{j}_{at[0]}_{at[1]}")

        ## (S) selection
        cons_S = {}
        for d_id in self.demands_info:
            n = len(pool.get(d_id))
            cons_S[d_id] = m.addConstr(
                gp.quicksum(z[(d_id, ti)] for ti in range(n)) == 1,
                name=f"S_{d_id}")

        ## (L) load definition: lambda = sum Gamma z
        gamma = {(d_id, ti): self._gamma(d_id, ti, traj)
                 for d_id in self.demands_info
                 for ti, traj in enumerate(pool.get(d_id))}
        cons_L = {}
        for at in self.arc_time_pairs:
            expr = gp.LinExpr()
            for (d_id, ti), g in gamma.items():
                val = g.get(at, 0.0)
                if val > 1e-12:
                    expr.add(z[(d_id, ti)], val)
            cons_L[at] = m.addConstr(lam[at] == expr, name=f"L_{at[0]}_{at[1]}")

        #### (B) budget
        cons_B = {}
        for t in self.inst.T_star:
            kappa = self.inst.budget.get(t)
            if kappa is None:
                continue
            expr = gp.LinExpr()
            for d_id in self.demands_info:
                for ti, traj in enumerate(pool.get(d_id)):
                    b = self._beta(d_id, ti, traj)
                    if b > 0:
                        expr.add(z[(d_id, ti)], float(b))
            if expr.size() > 0:
                cons_B[t] = m.addConstr(expr <= kappa, name=f"B_{t}")
        z1_star = z_stars[0] if z_stars else None

        ## Inherited rows (Tj, Nj) for j < k
        #   (Tj) lambda(a,t) <= z_j* + z_1* * y^(j)_{a,t}
        #   (Nj) sum_{a,t} y^(j)_{a,t} <= j - 1
        # j = 1 forces all loads <= z_1* (since N1 sums to 0), which also makes
        # the big-M = z_1* valid in every threshold row.
        #
        cons_Tj = {}
        for j in range(1, rank_k):
            zj = z_stars[j - 1]
            for at in self.arc_time_pairs:
                cons_Tj[(j,) + at] = m.addConstr(
                    lam[at] <= zj + z1_star * y[(j,) + at],
                    name=f"T{j}_{at[0]}_{at[1]}")
            m.addConstr(
                gp.quicksum(y[(j,) + at] for at in self.arc_time_pairs) <= j - 1,
                name=f"N{j}")

        ## Current-rank rows (Tk, Nk)
        cons_Tk = {}
        if rank_k == 1:
            # Rank 1: minimise the MLU directly (no exceedances).
            for at in self.arc_time_pairs:
                cons_Tk[at] = m.addConstr(lam[at] <= z_thresh,
                                          name=f"Tk_{at[0]}_{at[1]}")
        else:
            for at in self.arc_time_pairs:
                cons_Tk[at] = m.addConstr(
                    lam[at] <= z_thresh + z1_star * y[(rank_k,) + at],
                    name=f"Tk_{at[0]}_{at[1]}")
            m.addConstr(
                gp.quicksum(y[(rank_k,) + at] for at in self.arc_time_pairs)
                <= rank_k - 1, name="Nk")

        ## Objective: minimise z_thresh (the current rank's MLU threshold)
        m.setObjective(z_thresh, GRB.MINIMIZE)
        m.optimize()

        ## Status checks
        if not binary:
            # LP duals are only valid at proven optimality.
            if m.Status != GRB.OPTIMAL or m.SolCount == 0:
                return None
        else:
            # MIP: accept any feasible incumbent (e.g. under the time limit).
            if m.SolCount == 0:
                return None
        result = {"z_k": z_thresh.X, "status": m.Status}
        if not binary:
            result["pi"] = {d_id: c.Pi for d_id, c in cons_S.items()}
            result["mu_k"] = {at: c.Pi for at, c in cons_Tk.items()}
            result["eta_j"] = {key: c.Pi for key, c in cons_Tj.items()}
            result["psi"] = {t: c.Pi for t, c in cons_B.items()}

            # Stratified dual: mu_bar_k(a,t) = mu_k(a,t) + sum_{j<k} eta_j(a,t).
            mu_bar = {}
            for at in self.arc_time_pairs:
                val = result["mu_k"].get(at, 0.0)
                for j in range(1, rank_k):
                    val += result["eta_j"].get((j,) + at, 0.0)
                mu_bar[at] = val
            result["mu_bar"] = mu_bar
        else:
            sol = {}
            for d_id in self.demands_info:
                for ti, traj in enumerate(pool.get(d_id)):
                    if z[(d_id, ti)].X > 0.5:
                        sol[d_id] = traj
                        break
            result["solution"] = sol
        return result


###################
###  MAIN
###################

def main():
    args = parse_args()
    name = args.instance

    # -- Logging setup ------------------------------------------------------
    log_path, _log_fh = _setup_logging(name)
    # -------------------------------------------------------------------------

    base = detect_base(name, args.data_dir)
    net_f  = base / f"{name}-net.json"
    tm_f   = base / f"{name}-tm.json"
    scen_f = base / f"{name}-scenario.json"

    # Results live under results/<instance>/<ALGO_NAME>/ so that different
    # TCG variants never overwrite each other's output on the same instance.
    res_dir = RESULTS_DIR / name / ALGO_NAME
    res_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("  T-ASR -- Pair-Path Column Generation  (Rank-by-Rank Lex)")
    print(f"  Algorithm    : TCG-Diag  (folder: {ALGO_NAME})")
    print(f"  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 68)
    print(f"  Instance     : {name}")
    print(f"  K ranks      : {args.K}")
    print(f"  time_limit   : {args.time_limit}s")
    print(f"  lp_time/rank : {args.lp_time}s")
    print(f"  mip_time/rank: {args.mip_time}s")
    print(f"  topk_cols    : {args.topk_cols}")
    print(f"  Log          : {log_path}")
    print(f"  Results dir  : {res_dir}")
    print("=" * 68)

    from core.instance   import load_instance, summary
    from core.ecmp     import compute_all_splits, compute_loads, sorted_load_vector
    from stela.solver     import write_solution
    from .cg_master  import TrajectoryPool
    from .pricing_dp import price_demand_topk, build_pricing_context

    t_global = time.perf_counter()
    ## Load instance
    inst = load_instance(net_f, tm_f, scen_f)
    print(f"\n  {summary(inst)}")
    assert len(inst.T) == 2, "this driver is specialised to h = 2"
    r = compute_all_splits(inst)
    demands_info = {
        d_id: (src, tgt, vols)
        for (d_id, src, tgt, vols) in inst.demands
    }

    ## Pricing context: reachability per period + capacities, built ONCE
    ctx = build_pricing_context(inst)

    ## Baseline (IGP, no waypoints)
    trivial  = {(d_id, t): []
                for (d_id, _, _, _) in inst.demands for t in inst.T}
    base_vec = sorted_load_vector(compute_loads(trivial, inst, r), inst)
    print(f"  MLU baseline  : {base_vec[0]:.6f}")
    print(f"  Top-5 baseline: {[round(x,4) for x in base_vec[:5]]}")

    ## Initial pool: trivial trajectories (IGP, IGP) per demand
    pool = TrajectoryPool()
    for d_id in demands_info:
        pool.add(d_id, ([], []))
    print(f"\n  Initial pool : {pool.total_size()} trajectories")
    lex_rmp = LexRMP(inst, r)

    z_stars       = []
    best_solution = {}
    best_vec      = list(base_vec)

    print(f"\n  {'Rank':>5} {'CGit':>5} {'Pool':>7} {'LP_bound':>10}"
          f" {'z_k*':>10} {'Top3_sol':>22} {'Time':>7}")
    print(f"  {'-'*5} {'-'*5} {'-'*7} {'-'*10} {'-'*10} {'-'*22} {'-'*7}")

    ## LEX RANK LOOP
    for rank_k in range(1, args.K + 1):
        if time.perf_counter() - t_global > args.time_limit:
            print(f"\n  Time limit reached before rank {rank_k}.")
            break
        t_rank_start = time.perf_counter()
        lp_bound     = float('inf')
        cg_iters     = 0

        ## Inner CG for this rank
        for cg_iter in range(args.max_cg_iter):
            elapsed = time.perf_counter() - t_global
            if elapsed > args.time_limit:
                break
            remaining_lp = min(args.lp_time, args.time_limit - elapsed - 10)
            if remaining_lp < 5:
                break

            # Solve LP-RMP(k): must be optimal for valid duals
            lp_result = lex_rmp._build_model(
                pool, rank_k, z_stars, binary=False, time_limit=remaining_lp)
            if lp_result is None:
                break
            lp_bound = lp_result["z_k"]
            cg_iters += 1

            # Stratified duals
            mu_bar = lp_result["mu_bar"]
            pi_d   = lp_result["pi"]
            psi1   = lp_result["psi"].get(1, 0.0)        # psi(1) <= 0

            # Pricing per demand (exact diagonal + heuristic off-diagonal)
            added_count = 0
            for d_id, (src, tgt, vols) in demands_info.items():
                if time.perf_counter() - t_global > args.time_limit:
                    break
                results = price_demand_topk(
                    (d_id, src, tgt, vols), inst, r,
                    mu=mu_bar,
                    pi_d=pi_d.get(d_id, 0.0),
                    psi_1=psi1,
                    ctx=ctx,
                    top_k=args.topk_cols,
                    rc_tol=args.rc_tol,
                )
                for res in results:
                    if pool.add(d_id, res.trajectory):
                        added_count += 1

            # CG stopping: no improving column -> pool optimal for this rank
            if added_count == 0:
                break

        ## Rank MIP: certified z_k*
        elapsed = time.perf_counter() - t_global
        remaining_mip = min(args.mip_time, args.time_limit - elapsed - 5)
        if remaining_mip < 5:
            print(f"\n  Not enough time for MIP at rank {rank_k}.")
            break
        mip_result = lex_rmp._build_model(
            pool, rank_k, z_stars, binary=True, time_limit=remaining_mip)
        if mip_result is None:
            print(f"\n  MIP failed at rank {rank_k}.")
            break
        z_k_star = mip_result["z_k"]
        sol_k    = mip_result["solution"]
        z_stars.append(z_k_star)

        if sol_k:
            best_solution = sol_k
            wp_cur = {}
            for d_id, traj in sol_k.items():
                wp_cur[(d_id, 0)] = list(traj[0])
                wp_cur[(d_id, 1)] = list(traj[1])
            cur_vec  = sorted_load_vector(compute_loads(wp_cur, inst, r), inst)
            best_vec = cur_vec
            top3_str = str([round(x, 4) for x in cur_vec[:3]])
        else:
            top3_str = "N/A"
        t_rank = time.perf_counter() - t_rank_start
        print(f"  {rank_k:>5} {cg_iters:>5} {pool.total_size():>7}"
              f" {lp_bound:>10.6f} {z_k_star:>10.6f}"
              f" {top3_str:>22} {t_rank:>6.1f}s")


    ########################################
    ## FINAL RESULTS
    ########################################
    waypoints = {}
    for d_id, traj in best_solution.items():
        waypoints[(d_id, 0)] = list(traj[0])
        waypoints[(d_id, 1)] = list(traj[1])
    final_vec = (sorted_load_vector(compute_loads(waypoints, inst, r), inst)
                 if waypoints else base_vec)
    mlu_sol = final_vec[0]
    pct = (base_vec[0] - mlu_sol) / max(base_vec[0], 1e-12) * 100
    print(f"\n  {'='*64}")
    print(f"  Algorithm       : TCG-Diag")
    print(f"  MLU baseline    : {base_vec[0]:.6f}")
    print(f"  MLU solution    : {mlu_sol:.6f}  ({pct:+.2f}%)")
    print(f"  Certified ranks : {[round(z,6) for z in z_stars]}")
    print(f"  Top-5 baseline  : {[round(x,4) for x in base_vec[:5]]}")
    print(f"  Top-5 solution  : {[round(x,4) for x in final_vec[:5]]}")
    print(f"  Pool size       : {pool.total_size()}")

    sol_path = res_dir / f"{name}-srpaths.json"
    if waypoints:
        write_solution(waypoints, inst, sol_path)
        print(f"  Saved -> {sol_path}")

    ## Orange checker
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

    ## JSON summary -- self-identifying via the "algorithm" field
    t_total = time.perf_counter() - t_global
    with open(res_dir / f"{name}-summary.json", "w") as f:
        json.dump({
            "algorithm":     ALGO_NAME,
            "instance":      name,
            "K":             args.K,
            "mlu_baseline":  base_vec[0],
            "mlu_solution":  mlu_sol,
            "reduction_pct": pct,
            "z_stars":       z_stars,
            "top5_baseline": [round(x, 6) for x in base_vec[:5]],
            "top5_solution": [round(x, 6) for x in final_vec[:5]],
            "pool_size":     pool.total_size(),
            "time_s":        round(t_total, 2),
            "checker_valid": checker_valid,
            "checker_mlu":   checker_mlu,
        }, f, indent=2)
    print(f"  Summary saved -> {res_dir / f'{name}-summary.json'}")

    print(f"\n  Total time      : {t_total:.1f}s")
    print("=" * 68)

    # -- Close log ------------------------------------------------------------
    _teardown_logging(_log_fh)
    # ---------------------------------------------------------------------------


if __name__ == "__main__":
    main()