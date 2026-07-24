"""
Function: solver.py
Role: Sequential lexicographic loop over the ALEXA assignment-sort model.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import gurobipy as gp
from gurobipy import GRB

from core.ecmp import SplitCoeffs
from core.instance import Instance

from .model import AlexaModel, build_alexa_model

Waypoints = Dict[Tuple[int, int], List[int]]


def _reconstruct_path(src: int, tgt: int, active: Dict[Tuple[int, int], bool]) -> List[int]:
    """Follow active segments from src to tgt via DFS with backtracking."""
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


class AssignmentLexSolver:
    """
    Parameters:
        inst              : parsed Instance (inst.max_seg read automatically)
        r                 : ECMP split coefficients {t: SplitCoeffs}
        time_limit         : Gurobi time limit PER RANK/ITERATION (seconds)
        mip_gap            : relative MIP optimality gap, applied per rank
        eps                : one-sided freeze tolerance (L[k] <= l_k_opt + eps)
        max_rank           : stop after this many ranks (default: all N)
        verbose             : print the Gurobi log
    """

    def __init__(
        self,
        inst: Instance,
        r: Dict[int, SplitCoeffs],
        time_limit: float = 300.0,
        mip_gap: float = 1e-8,
        eps: float = 1e-11,
        max_rank: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        self.inst = inst
        self.r = r
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.eps = eps
        self.verbose = verbose

        print(f"  Building ALEXA model "
              f"(maxSeg={inst.max_seg}, |V|={len(inst.nodes)}, "
              f"|D|={len(inst.demands)}, |T|={len(inst.T)}) ...")
        self.built: AlexaModel = build_alexa_model(inst, r, verbose=verbose)
        self.max_rank = max_rank if max_rank is not None else self.built.N

    def solve(self) -> Tuple[Optional[Waypoints], List[float]]:
        inst = self.inst
        built = self.built
        m, L = built.m, built.L
        N = built.N
        t0 = time.perf_counter()

        z_stars: List[float] = []
        best_waypoints: Optional[Waypoints] = None

        print(f"\n  {'k':>5} {'z*_k':>10} {'iter_time':>10} "
              f"{'elapsed':>10} {'status':>8}")
        print(f"  {'-'*5} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

        status_labels = {2: "OPTIMAL", 9: "TIME_LIM", 3: "INFEASIB"}

        for k in range(min(self.max_rank, N)):
            iter_t0 = time.perf_counter()
            m.setObjective(L[k], GRB.MINIMIZE)
            m.Params.TimeLimit = self.time_limit
            m.Params.MIPGap = self.mip_gap
            m.update()
            m.optimize()
            status = m.Status

            if status == GRB.INFEASIBLE:
                print(f"  [k={k+1}] INFEASIBLE -- stopping.")
                m.computeIIS()
                m.write("alexa_infeasible.ilp")
                break
            if status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.NODE_LIMIT, GRB.SOLUTION_LIMIT):
                print(f"  [k={k+1}] Unexpected status {status} -- stopping.")
                break
            if m.SolCount == 0:
                print(f"  [k={k+1}] No integer solution found -- "
                      "try increasing time_limit.")
                break

            l_k_val = L[k].X
            z_stars.append(l_k_val)

            iter_time = time.perf_counter() - iter_t0
            elapsed = time.perf_counter() - t0
            status_label = status_labels.get(status, f"st={status}")
            print(f"  {k+1:>5} {l_k_val:>10.6f} {iter_time:>9.1f}s "
                  f"{elapsed:>9.1f}s {status_label:>8}")

            best_waypoints = self._extract_waypoints()

            # One-sided tolerant freeze (matches the reference fix; NOT the
            # symmetric +/-1e-6 band currently in stela/solver.py).
            m.addConstr(L[k] <= l_k_val + self.eps, name=f"lex_fix_{k}")
            m.update()

            if l_k_val < 1e-9:
                print("  All loads are 0 -- converged.")
                break

        elapsed = time.perf_counter() - t0
        print(f"\n  Done.  {len(z_stars)} ranks in {elapsed:.1f}s")
        if z_stars:
            print(f"  MLU (z*_1) = {z_stars[0]:.6f}")

        return best_waypoints, z_stars

    def _extract_waypoints(self) -> Waypoints:
        """Convert x^dt_ij binary values to {(demand_id, t): [w1, w2, ...]}."""
        inst = self.inst
        x = self.built.x
        valid_segs = self.built.valid_segs
        wp: Waypoints = {}

        for (d_id, src, tgt, _vols) in inst.demands:
            for t in inst.T:
                active = {
                    (i, j): True
                    for (i, j) in valid_segs[t]
                    if (d_id, t, i, j) in x and x[d_id, t, i, j].X > 0.5
                }
                if not active:
                    wp[(d_id, t)] = []
                    continue
                path = _reconstruct_path(src, tgt, active)
                wp[(d_id, t)] = path[1:-1]

        return wp
