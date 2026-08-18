"""
run_diag.py  (TCG_carla)
========================

TCG-Diag-CARLA: rank-by-rank pair-path CG driver (h = 2) with the CARLA
cumulative Ogryczak-Tamir rank encoding hardcoded. 

Usage (from Algorithm_codes/):
    python -m TCG_carla.run_diag --instance setA-01 --K 5
    python -m TCG_carla.run_diag setAP-001 --data-dir=../setAP/ --K 5
    python -m TCG_carla.run_diag --instance setA-13 --time-limit 6000 \
        --topk_cols 1 --max_cg_iter 6 --lp_time 200 --pricing_time 200 --mip_time 400

Outputs (always):
    results/<instance>/TCG_DIAG_CARLA/<instance>-srpaths.json
    results/<instance>/TCG_DIAG_CARLA/<instance>-summary.json
Logs (always):
    logs/TCG_DIAG_CARLA/<instance>_<timestamp>.log
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

# Hardcoded CARLA namespace -- never collides with the STELA TCG_DIAG runs.
ALGO_NAME = "TCG_DIAG_CARLA"
RANK_ENCODING = "carla"

# Small one-sided slack on the cumulative freeze rows (see TCG_carla.master).
_CARLA_FREEZE_EPS = 1e-6


### Gurobi setup
def _setup_gurobi():
    for p in [Path.home() / "gurobi.lic",
              CODES_DIR / "gurobi.lic"]:
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


def _teardown_logging(log_fh):
    sys.stdout = sys.__stdout__
    try:
        log_fh.close()
    except Exception:
        pass


def _release_gurobi():
    try:
        import gurobipy as gp
        gp.disposeDefaultEnv()
    except Exception:
        pass


#################################
### CLI
#################################

def parse_args():
    p = argparse.ArgumentParser(
        description="T-ASR -- Rank-by-Rank Pair-Path CG (TCG-Diag-CARLA)"
    )
    add_dataset_args(p)

    p.add_argument(
        "--checker-dir", "--checker_dir",
        dest="checker_dir", default=None,
        help="Checker binary directory",
    )
    p.add_argument("--K", type=int, default=12,
                   help="Number of lex ranks to certify (default 12)")
    p.add_argument(
        "--time-limit", "--time_limit",
        dest="time_limit", type=float, default=3600,
        help="Soft global cap; the last rank may overshoot it by "
             "(build time + mip_time)",
    )
    p.add_argument(
        "--lp-time", "--lp_time", dest="lp_time", type=float, default=240,
        help="Max time per LP-RMP optimize() call",
    )
    p.add_argument(
        "--mip-time", "--mip_time", dest="mip_time", type=float, default=240,
        help="Time granted to each rank MIP; always reserved, never skipped",
    )
    p.add_argument(
        "--pricing-time", "--pricing_time",
        dest="pricing_time", type=float, default=300,
        help="Max wall-clock per CG pricing sweep (all demands)",
    )
    p.add_argument(
        "--max-cg-iter", "--max_cg_iter",
        dest="max_cg_iter", type=int, default=20,
        help="Max CG iterations per rank",
    )
    p.add_argument(
        "--topk-cols", "--topk_cols",
        dest="topk_cols", type=int, default=5,
        help="Columns added per demand per iteration",
    )
    p.add_argument("--rc-tol", "--rc_tol",
                   dest="rc_tol", type=float, default=-1e-8)
    return p.parse_args()


###############################################
###  LEX RMP  (CARLA master for a given rank)
##############################################

class LexRMP:
    """
    Restricted Master Problem for rank k with the CARLA cumulative encoding.

    Load is modelled with explicit lambda(a,t) variables and a (L) equality
    lambda = sum Gamma z, so the pricing pressure is rho_L = -mu_bar, where
    mu_bar(a,t) is the sum of the duals of the Ogryczak-Tamir rows that bound
    lambda(a,t).
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
        from TCG_Diag.cg_master import trajectory_budget_cost
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
        Build and solve the CARLA RMP at rank k.
          binary=False -> LP relaxation (duals for pricing); requires OPTIMAL.
          binary=True  -> MIP (routing solution); requires a feasible incumbent.
        Returns None on failure.

        z_stars = [z_1*, ..., z_{k-1}*] are the certified sorted loads; they are
        cumulated into Theta_j* = sum_{r<=j} z_r* and frozen one-sidedly.
        """
        import gurobipy as gp
        from gurobipy import GRB

        self._check_pool(pool)
        m = gp.Model(f"RMP_rank{rank_k}_{'mip' if binary else 'lp'}_carla")
        m.setParam("OutputFlag", 1 if verbose else 0)
        m.setParam("TimeLimit", time_limit)
        vtype = GRB.BINARY if binary else GRB.CONTINUOUS

        try:
            ## Routing variables z_{d,ti}
            z = {}
            for d_id in self.demands_info:
                for ti in range(len(pool.get(d_id))):
                    z[(d_id, ti)] = m.addVar(lb=0.0, ub=1.0, vtype=vtype,
                                             name=f"z_{d_id}_{ti}")

            ## Load variables lambda(a,t)
            lam = {at: m.addVar(lb=0.0, vtype=GRB.CONTINUOUS,
                                name=f"lam_{at[0]}_{at[1]}")
                   for at in self.arc_time_pairs}

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

            ## CARLA cumulative rank rows -----------------------------------
            # Cumulative certified levels Theta_j* = sum_{r<=j} z_r*.
            theta_star: List[float] = []
            acc = 0.0
            for zr in z_stars:
                acc += float(zr)
                theta_star.append(acc)          # theta_star[j-1] = Theta_j*

            # Free levels q_j and nonnegative excesses d_{j,a,t}, j = 1..k.
            # Always continuous: no binary in the rank encoding -- only z (the
            # selection vars) toggles integrality.
            q = {}
            dvar = {}
            for j in range(1, rank_k + 1):
                q[j] = m.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS,
                                name=f"q_{j}")
                for at in self.arc_time_pairs:
                    dvar[(j,) + at] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS,
                                               name=f"d_{j}_{at[0]}_{at[1]}")

            # Ogryczak-Tamir rows: lambda(a,t) <= q_j + d_{j,a,t}, all j=1..k.
            # load_bounding_by_at collects them so mu_bar(a,t) = sum_j Pi_{j,a,t}.
            load_bounding_by_at: Dict[Tuple[int, int], list] = {
                at: [] for at in self.arc_time_pairs}
            cons_ot = {}
            for j in range(1, rank_k + 1):
                for at in self.arc_time_pairs:
                    c = m.addConstr(lam[at] <= q[j] + dvar[(j,) + at],
                                    name=f"OT{j}_{at[0]}_{at[1]}")
                    cons_ot[(j,) + at] = c
                    load_bounding_by_at[at].append(c)

            # One-sided cumulative freeze of inherited ranks j<k.
            for j in range(1, rank_k):
                theta_j = (j * q[j]
                           + gp.quicksum(dvar[(j,) + at] for at in self.arc_time_pairs))
                m.addConstr(theta_j <= theta_star[j - 1] + _CARLA_FREEZE_EPS,
                            name=f"freeze_theta_{j}")

            # Objective: minimise Theta_k = k*q_k + sum d_{k,a,t}.
            theta_k = (rank_k * q[rank_k]
                       + gp.quicksum(dvar[(rank_k,) + at] for at in self.arc_time_pairs))
            m.setObjective(theta_k, GRB.MINIMIZE)

            m.optimize()

            ## Status checks
            if not binary:
                if m.Status != GRB.OPTIMAL or m.SolCount == 0:
                    return None
            else:
                if m.SolCount == 0:
                    return None

            # z_k here is the cumulative objective Theta_k (for the log only);
            # the driver recovers z_k* from the incumbent's sorted load vector.
            result = {"z_k": float(m.ObjVal), "status": m.Status}
            if not binary:
                result["pi"] = {d_id: c.Pi for d_id, c in cons_S.items()}
                result["psi"] = {t: c.Pi for t, c in cons_B.items()}
                mu_bar = {}
                for at in self.arc_time_pairs:
                    mu_bar[at] = sum(c.Pi for c in load_bounding_by_at[at])
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
        finally:
            m.dispose()


###################
###  MAIN
###################

def _run(args, name, net_f, tm_f, scen_f):
    res_dir = RESULTS_DIR / name / ALGO_NAME
    res_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("  T-ASR -- Pair-Path Column Generation  (Rank-by-Rank Lex)")
    print(f"  Algorithm    : TCG-Diag-CARLA  (folder: {ALGO_NAME})")
    print(f"  Rank encoding: {RANK_ENCODING}")
    print(f"  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 68)
    print(f"  Instance     : {name}")
    print(f"  Dataset      : {net_f.parent}")
    print(f"  K ranks      : {args.K}")
    print(f"  time_limit   : {args.time_limit}s   (soft cap)")
    print(f"  lp_time/iter : {args.lp_time}s   (bounds ONE LP optimize())")
    print(f"  pricing_time : {args.pricing_time}s   (bounds ONE pricing sweep)")
    print(f"  mip_time/rank: {args.mip_time}s   (always reserved, never skipped)")
    print(f"  max_cg_iter  : {args.max_cg_iter}")
    print(f"  topk_cols    : {args.topk_cols}")
    print(f"  Results dir  : {res_dir}")
    print("=" * 68)

    from core.instance import load_instance, summary
    from core.ecmp import compute_all_splits, compute_loads, sorted_load_vector
    from stela.solver import write_solution
    from TCG_Diag.cg_master import TrajectoryPool
    from TCG_Diag.pricing_dp import price_demand_topk, build_pricing_context

    t_global = time.perf_counter()
    ## Load instance
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

    ## Pricing context: reachability per period + capacities, built ONCE
    ctx = build_pricing_context(inst)

    ## Baseline (IGP, no waypoints)
    trivial = {(d_id, t): []
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

    z_stars = []
    best_solution = {}
    best_vec = list(base_vec)

    print(f"\n  {'Rank':>5} {'CGit':>5} {'Pool':>7} {'Theta_k':>10}"
          f" {'z_k*':>10} {'Top3_sol':>22} {'Time':>7}")
    print(f"  {'-'*5} {'-'*5} {'-'*7} {'-'*10} {'-'*10} {'-'*22} {'-'*7}")

    ## LEX RANK LOOP
    for rank_k in range(1, args.K + 1):
        if time.perf_counter() - t_global > args.time_limit:
            print(f"\n  Time limit reached before rank {rank_k}.")
            break
        t_rank_start = time.perf_counter()
        lp_bound = float('inf')
        cg_iters = 0

        ## Inner CG for this rank.
        cg_deadline = t_global + args.time_limit - args.mip_time - 10
        for cg_iter in range(args.max_cg_iter):
            now = time.perf_counter()
            if now > cg_deadline:
                print(f"    r{rank_k} it{cg_iter+1:>2}  stop CG "
                      f"-> {args.mip_time:.0f}s reserved for the rank MIP")
                break
            remaining_lp = min(args.lp_time, cg_deadline - now)
            if remaining_lp < 5:
                print(f"    r{rank_k} it{cg_iter+1:>2}  stop CG (no room for an LP)")
                break

            t_lp0 = time.perf_counter()
            lp_result = lex_rmp._build_model(
                pool, rank_k, z_stars, binary=False, time_limit=remaining_lp)
            dt_lp = time.perf_counter() - t_lp0
            if lp_result is None:
                print(f"    r{rank_k} it{cg_iter+1:>2}  LP not optimal "
                      f"({dt_lp:.1f}s) -> stop CG")
                break
            lp_bound = lp_result["z_k"]          # Theta_k (cumulative)
            cg_iters += 1

            mu_bar = lp_result["mu_bar"]
            pi_d = lp_result["pi"]
            psi1 = lp_result["psi"].get(1, 0.0)        # psi(1) <= 0

            t_price = time.perf_counter()
            price_deadline = min(t_price + args.pricing_time, cg_deadline)
            added_count = 0
            n_priced = 0
            for d_id, (src, tgt, vols) in demands_info.items():
                if time.perf_counter() > price_deadline:
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
                n_priced += 1
            dt_price = time.perf_counter() - t_price

            print(f"    r{rank_k} it{cg_iter+1:>2}  pool {pool.total_size():>6}  "
                  f"Theta {lp_bound:>9.6f}  +{added_count:>4} cols  "
                  f"priced {n_priced}/{len(demands_info)}  "
                  f"lp {dt_lp:>5.1f}s  price {dt_price:>6.1f}s")

            if added_count == 0:
                break

        ## Rank MIP -- ALWAYS executed, with the full args.mip_time.
        t_mip0 = time.perf_counter()
        mip_result = lex_rmp._build_model(
            pool, rank_k, z_stars, binary=True, time_limit=args.mip_time)
        dt_mip = time.perf_counter() - t_mip0
        if mip_result is None:
            print(f"\n  MIP found no feasible incumbent at rank {rank_k} "
                  f"({dt_mip:.1f}s).")
            break
        sol_k = mip_result["solution"]

        # Sorted load vector of the incumbent -> reporting AND z_k* recovery.
        cur_vec = list(best_vec)
        if sol_k:
            best_solution = sol_k
            wp_cur = {}
            for d_id, traj in sol_k.items():
                wp_cur[(d_id, 0)] = list(traj[0])
                wp_cur[(d_id, 1)] = list(traj[1])
            cur_vec = sorted_load_vector(compute_loads(wp_cur, inst, r), inst)
            best_vec = cur_vec
            top3_str = str([round(x, 4) for x in cur_vec[:3]])
        else:
            top3_str = "N/A"

        # z_k* recovered from the incumbent's sorted load vector (the cumulative
        # master has no z_thresh); z_stars is cumulated into Theta_j* by the
        # master at the next rank.
        z_k_star = cur_vec[rank_k - 1] if rank_k - 1 < len(cur_vec) else 0.0
        z_stars.append(z_k_star)

        t_rank = time.perf_counter() - t_rank_start
        print(f"  {rank_k:>5} {cg_iters:>5} {pool.total_size():>7}"
              f" {lp_bound:>10.6f} {z_k_star:>10.6f}"
              f" {top3_str:>22} {t_rank:>6.1f}s  (mip {dt_mip:.1f}s)")

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
    print(f"  Algorithm       : TCG-Diag-CARLA")
    print(f"  MLU baseline    : {base_vec[0]:.6f}")
    print(f"  MLU solution    : {mlu_sol:.6f}  ({pct:+.2f}%)")
    print(f"  Certified ranks : {len(z_stars)} / {args.K} requested")
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

    ## Cumulative optima Theta_k* = sum of the k largest loads (CARLA reporting).
    theta_stars = []
    acc = 0.0
    for zk in z_stars:
        acc += float(zk)
        theta_stars.append(acc)

    ## JSON summary -- self-identifying via the "algorithm" field
    t_total = time.perf_counter() - t_global
    summary_path = res_dir / f"{name}-summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "algorithm":     ALGO_NAME,
            "rank_encoding": RANK_ENCODING,
            "instance":      name,
            "K":             args.K,
            "ranks_solved":  len(z_stars),
            "mlu_baseline":  base_vec[0],
            "mlu_solution":  mlu_sol,
            "reduction_pct": pct,
            "z_stars":       z_stars,
            "theta_stars":   theta_stars,
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

    net_f, tm_f, scen_f = resolve_paths(args)

    log_path, _log_fh = _setup_logging(name)
    print(f"  Log          : {log_path}")

    try:
        _run(args, name, net_f, tm_f, scen_f)
    finally:
        _release_gurobi()
        _teardown_logging(_log_fh)


if __name__ == "__main__":
    main()
