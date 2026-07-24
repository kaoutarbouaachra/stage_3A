"""
Function: solver.py
Role:  CARLA solver -- Cumulative Aggregation of Ranked Loads Algorithm.

"""

from __future__ import annotations
import os
from pathlib import Path as _Path
from core.paths import CODES_DIR

for _p in [
    _Path.home() / "gurobi.lic",
    CODES_DIR / "gurobi.lic",
]:
    if _p.exists():
        os.environ["GRB_LICENSE_FILE"] = str(_p)
        break

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import gurobipy as gp
import networkx as nx
from gurobipy import GRB

from core.ecmp import SplitCoeffs, compute_all_splits
from core.instance import Instance
from core.constraints import (
    add_flow_conservation,
    add_segment_count,
    add_budget,
    build_load_exprs,
)


#####################################################
###  Segment generation
#####################################################

def _useful_segments(inst: Instance) -> Dict[int, Set[Tuple[int, int]]]:
    """
    Role: For each time step t, return ONLY the (i,j) pairs that can appear
    as a segment in a valid SR-path with at most inst.max_seg segments.

    A path s->w1->...->wl->t has 3 types of segments:
      Type 1: (s, t)          -- direct, 0 waypoints
      Type 2: (s, w), (w, t)  -- first/last segments, >=1 waypoint
      Type 3: (w1, w2)        -- intermediate hop, >=2 waypoints (maxSeg>=3)
    """
    max_seg = inst.max_seg
    max_wps = max_seg - 1
    valid: Dict[int, Set[Tuple[int, int]]] = {}

    for t in inst.T:
        unavail = inst.interv.get(t, set())
        G = nx.DiGraph()
        for nid in inst.nodes:
            G.add_node(nid)
        for l in inst.links.values():
            if l['id'] not in unavail:
                G.add_edge(l['from'], l['to'], weight=l['metric'])

        # Pre-compute reachability ONCE per time step (shared across demands)
        Gr = G.reverse(copy=False)
        reach_from: Dict[int, Set[int]] = {}
        reach_to:   Dict[int, Set[int]] = {}
        for nid in inst.nodes:
            reach_from[nid] = set(
                nx.single_source_shortest_path_length(G,  nid).keys())
            reach_to[nid]   = set(
                nx.single_source_shortest_path_length(Gr, nid).keys())

        segs: Set[Tuple[int, int]] = set()
        for (d_id, src, tgt, _) in inst.demands:
            valid_wps = (reach_from[src] & reach_to[tgt]) - {src, tgt}
            # Type 1: direct segment (0 waypoints)
            if tgt in reach_from[src]:
                segs.add((src, tgt))
            # Type 2: (src->w) and (w->tgt) -- 1 waypoint
            if max_wps >= 1:
                for w in valid_wps:
                    segs.add((src, w))
                    segs.add((w,   tgt))
            # Type 3: (w1->w2) -- intermediate hop, >=2 waypoints
            if max_wps >= 2:
                for w1 in valid_wps:
                    for w2 in valid_wps:
                        if w2 != w1 and w2 in reach_from[w1]:
                            segs.add((w1, w2))
        valid[t] = segs

    total  = sum(len(v) for v in valid.values())
    naive  = len(inst.nodes) ** 2 * len(inst.T)
    factor = naive // max(total, 1)
    print(f"  -> {total:,} useful segment slots  "
          f"(naive: {naive:,} -> reduction x{factor})")
    return valid


#####################################################
###  Path reconstruction
#####################################################

def _reconstruct_path(
    src: int, tgt: int,
    active: Dict[Tuple[int, int], bool],
) -> List[int]:
    """Follow active segments from src to tgt via DFS."""
    nxt = defaultdict(list)
    for (i, j) in active:
        nxt[i].append(j)

    def dfs(cur, path, visited_edges):
        if cur == tgt:
            return path[:]
        for nb in nxt[cur]:
            edge = (cur, nb)
            if edge not in visited_edges:
                visited_edges.add(edge)
                path.append(nb)
                result = dfs(nb, path, visited_edges)
                if result is not None:
                    return result
                path.pop()
                visited_edges.remove(edge)
        return None

    result = dfs(src, [src], set())
    return result if result else []


#####################################################
###  Small helper
#####################################################

def _value_of(expr) -> float:
    """Evaluate a Gurobi LinExpr or Var at the current solution."""
    try:
        return expr.getValue()
    except AttributeError:
        return expr.X


#####################################################
###  Main solver
#####################################################

class CumulativeLexSolver:
    """
    Role: Implements CARLA -- sequential cumulative lex-minimax MILP for T-ASR.

    Parameters:
        inst           : parsed Instance (inst.max_seg read automatically)
        r              : ECMP split coefficients {t: SplitCoeffs}
        time_limit     : Gurobi time limit per sub-problem (seconds)
        mip_gap        : relative MIP gap. Should be 0.0 here: the objective
                         Theta_k grows like k * lambda_1, so a constant
                         relative gap yields an absolute error on lambda_k
                         that grows with the rank.
        mip_gap_abs    : absolute MIP gap. This is the meaningful tolerance:
                         since Theta_{k-1} is a fixed constant, an absolute
                         error delta on Theta_k is an absolute error delta on
                         lambda_k.
        epsilon        : one-sided slack used when freezing a level.
        max_iter       : stop after this many ranks (default = N = |A|*|T|)
        free_threshold : if False (default), a frozen level j is enforced by
                         fixing z_j = lambda_j and bounding sum_i u^(j)_i.
                         If True, z_j is left free and the aggregated form
                         j*z_j + sum_i u^(j)_i <= Theta_j is used instead.
                         Both are exact; they differ in size and conditioning.
        verbose        : print Gurobi log
    """

    def __init__(
        self,
        inst: Instance,
        r: Dict[int, SplitCoeffs],
        time_limit: float = 300.0,
        mip_gap: float = 0.0,
        mip_gap_abs: float = 1e-8,
        epsilon: float = 1e-9,
        max_iter: Optional[int] = None,
        free_threshold: bool = False,
        verbose: bool = False,
    ) -> None:
        self.inst           = inst
        self.r              = r
        self.time_limit     = time_limit
        self.mip_gap        = mip_gap
        self.mip_gap_abs    = mip_gap_abs
        self.epsilon        = epsilon
        self.free_threshold = free_threshold
        self.verbose        = verbose

        self.components: List[Tuple[int, int]] = [
            (a, t) for t in inst.T for a in inst.arc_ids
        ]
        self.N        = len(self.components)
        self.max_iter = max_iter if max_iter is not None else self.N

        print(f"  Pre-computing useful segments "
              f"(maxSeg={inst.max_seg}, |V|={len(inst.nodes)}, "
              f"|D|={len(inst.demands)}, |T|={len(inst.T)}) ...")
        self.valid_segs = _useful_segments(inst)

    # -------------------------------------------------

    def solve(self) -> Tuple[Dict, List[float], List[float]]:
        """
        Role: Run the sequential cumulative lex-minimax algorithm.

        Returns:
            waypoints  : {(demand_id, t): [w1, w2, ...]}
            lam_bars   : [lambda_1, lambda_2, ...]  (individual sorted loads)
            theta_bars : [Theta_1,  Theta_2,  ...]  (cumulative optima)
        """
        inst = self.inst
        N    = self.N
        t0   = time.perf_counter()

        ### Build Gurobi model
        m = gp.Model("T-ASR-CARLA")
        m.Params.OutputFlag = int(self.verbose)
        m.Params.MIPGap     = self.mip_gap
        m.Params.MIPGapAbs  = self.mip_gap_abs

        ### x[d_id, t, i, j] -- binary segment indicator
        x_keys = [
            (d_id, t, i, j)
            for (d_id, _, _, _) in inst.demands
            for t in inst.T
            for (i, j) in self.valid_segs[t]
        ]
        x = m.addVars(x_keys, vtype=GRB.BINARY, name="x")
        print(f"\n  x variables : {len(x_keys):,}")

        ### delta[d_id, t, i, j] -- linearises |x^dt - x^{d,t-1}|
        delta_keys = [
            (d_id, t, i, j)
            for (d_id, _, _, _) in inst.demands
            for t in inst.T_star
            for (i, j) in (self.valid_segs[t] | self.valid_segs[t - 1])
        ]
        delta = m.addVars(delta_keys, vtype=GRB.BINARY, name="delta")

        ### Load expressions lambda(arc, t)
        lam = build_load_exprs(m, x, inst, self.r,
                               self.components, self.valid_segs)

        ### Base constraints (C1), (C2), (C4)
        add_flow_conservation(m, x, inst, self.valid_segs)
        add_segment_count(m, x, inst, self.valid_segs)
        add_budget(m, x, delta, inst, self.valid_segs)
        m.update()
        print(f"  Base model : {m.NumVars:,} vars, "
              f"{m.NumConstrs:,} constraints, "
              f"{m.NumNZs:,} nonzeros\n")

        ### Sequential iterations
        lam_bars:       List[float]    = []   # lambda_1, lambda_2, ...
        theta_bars:     List[float]    = []   # Theta_1,  Theta_2,  ...
        best_waypoints: Optional[Dict] = None
        iter_times:     List[float]    = []

        print(f"\n  {'k':>5} {'Theta_k':>12} {'lambda_k':>12} "
              f"{'iter_time':>10} {'elapsed':>10} {'status':>9}")
        print(f"  {'-'*5} {'-'*12} {'-'*12} "
              f"{'-'*10} {'-'*10} {'-'*9}")

        for k in range(1, self.max_iter + 1):
            iter_t0 = time.perf_counter()

            # ---- current-level block -------------------------------------
            ub_z = lam_bars[-1] if lam_bars else GRB.INFINITY
            z_k  = m.addVar(lb=0.0, ub=ub_z, name=f"z{k}")

            if k == 1:
                # Rank 1 degenerates: Theta_1 = lambda_1 = MLU, and the
                # frozen form of level 1 is simply the global bound
                # lam_i <= lambda_1. No u^(1) variables are needed.
                u_k = None
                for idx in range(N):
                    m.addConstr(lam[idx] <= z_k, name=f"glob_{idx}")
                m.setObjective(z_k, GRB.MINIMIZE)
            else:
                # lam_i <= lambda_1 already enforced by level 1, so
                # u^(k)_i = (lam_i - z_k)_+ <= lambda_1 is a valid bound.
                u_k = m.addVars(N, lb=0.0, ub=lam_bars[0], name=f"u{k}")
                for idx in range(N):
                    m.addConstr(u_k[idx] >= lam[idx] - z_k,
                                name=f"link_{k}_{idx}")
                m.setObjective(
                    k * z_k + gp.quicksum(u_k[idx] for idx in range(N)),
                    GRB.MINIMIZE,
                )

            m.Params.TimeLimit = self.time_limit
            m.update()
            m.optimize()
            status = m.Status

            if status == GRB.INFEASIBLE:
                print(f"  [k={k}] INFEASIBLE -- stopping.")
                m.computeIIS()
                m.write("infeasible_carla.ilp")
                break

            if status not in (GRB.OPTIMAL, GRB.TIME_LIMIT,
                              GRB.NODE_LIMIT, GRB.SOLUTION_LIMIT):
                print(f"  [k={k}] Unexpected status {status} -- stopping.")
                break

            if m.SolCount == 0:
                print(f"  [k={k}] No integer solution found -- "
                      "try increasing time_limit.")
                break

            # ---- read the incumbent --------------------------------------
            # lambda_k is recovered by EVALUATING the returned scheme, not by
            # differencing Theta_k - Theta_{k-1}: the theorem guarantees
            # P*_k in F_k, hence lambda_k^down(P*_k) = lambda_k exactly.
            # Note also that the optimal z_k is only known to lie in
            # [lambda_{k+1}, lambda_k], so it must NOT be used as lambda_k.
            loads = sorted((_value_of(lam[i]) for i in range(N)), reverse=True)
            lam_k   = loads[k - 1]
            theta_k = sum(loads[:k])
            c_k     = theta_k - k * lam_k          # = sum_{r<k} (lam_r - lam_k)

            if abs(theta_k - m.ObjVal) > 1e-6 * max(1.0, abs(theta_k)):
                print(f"  [k={k}] WARNING: Theta_k from incumbent "
                      f"({theta_k:.9f}) differs from ObjVal "
                      f"({m.ObjVal:.9f}).")

            lam_bars.append(lam_k)
            theta_bars.append(theta_k)
            best_waypoints = self._extract_waypoints(x)

            # ---- freeze level k ------------------------------------------
            # One-sided only. A two-sided window would exclude schemes lying
            # exactly at the boundary as soon as the recorded constants carry
            # any arithmetic error, and the exclusion propagates downwards.
            eps = self.epsilon
            if k == 1:
                # lam_i <= z_1 <= lambda_1 + eps
                z_k.ub = lam_k + eps
                # lb stays at 0: z_1 no longer appears in any objective.
            elif self.free_threshold:
                m.addConstr(
                    k * z_k + gp.quicksum(u_k[i] for i in range(N))
                    <= theta_k + eps,
                    name=f"frz_{k}",
                )
            else:
                z_k.lb = lam_k
                z_k.ub = lam_k
                m.addConstr(
                    gp.quicksum(u_k[i] for i in range(N)) <= c_k + eps,
                    name=f"frz_{k}",
                )
                box = max(0.0, lam_bars[0] - lam_k) + eps
                for i in range(N):
                    u_k[i].ub = box

            m.update()

            # ---- logging --------------------------------------------------
            iter_time     = time.perf_counter() - iter_t0
            total_elapsed = time.perf_counter() - t0
            iter_times.append(iter_time)

            status_label = {
                2: "OPTIMAL",
                9: "TIME_LIM",
                3: "INFEASIB",
            }.get(status, f"st={status}")

            filled = int(20 * k / self.max_iter)
            bar    = '#' * filled + '.' * (20 - filled)
            pct    = 100 * k / self.max_iter

            print(
                f"  {k:>5} {theta_k:>12.6f} {lam_k:>12.6f} "
                f"{iter_time:>9.1f}s {total_elapsed:>9.1f}s "
                f"{status_label:>9}  [{bar}] {pct:.0f}%"
            )

            if lam_k < 1e-9:
                print("  All remaining loads are 0 -- converged.")
                break

        total_elapsed = time.perf_counter() - t0
        print(f"\n  Done.  {len(lam_bars)} ranks certified "
              f"in {total_elapsed:.1f}s")
        if lam_bars:
            print(f"  MLU (lambda_1) = {lam_bars[0]:.6f}")

        return best_waypoints, lam_bars, theta_bars

    #####################################################
    ###  Extract solution
    #####################################################

    def _extract_waypoints(self, x: gp.tupledict) -> Dict:
        """Convert x^dt_ij binary values to {(demand_id, t): [w1, w2, ...]}."""
        inst = self.inst
        wp: Dict[Tuple[int, int], List[int]] = {}

        for (d_id, src, tgt, _) in inst.demands:
            for t in inst.T:
                active = {
                    (i, j): True
                    for (i, j) in self.valid_segs[t]
                    if (d_id, t, i, j) in x
                    and x[d_id, t, i, j].X > 0.5
                }
                if not active:
                    wp[(d_id, t)] = []
                    continue

                path = _reconstruct_path(src, tgt, active)
                wp[(d_id, t)] = path[1:-1]

        return wp


#####################################################
### Output writer
#####################################################

def write_solution(
    waypoints: Dict[Tuple[int, int], List[int]],
    inst: Instance,
    out_path: Path,
) -> None:
    """Write solution to srpaths.json format."""
    srpaths = []
    for (d_id, t), wps in sorted(waypoints.items()):
        if wps:
            srpaths.append({"d": d_id, "t": t, "w": wps})
    with open(out_path, "w") as f:
        json.dump({"srpaths": srpaths}, f, indent=2)
    print(f"  Solution written -> {out_path}")
    