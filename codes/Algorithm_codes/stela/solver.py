"""
Function: solver.py
Role:  Stela solver.
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
    Role:For each time step t, return ONLY the (i,j) pairs that can appear
    as a segment in a valid SR-path with at most inst.max_seg segments.

    A path s→w1→...→wl→t has 3 types of segments:
      Type 1: (s, t)          — direct, 0 waypoints
      Type 2: (s, w), (w, t)  — first/last segments, ≥1 waypoint
      Type 3: (w1, w2)        — intermediate hop, ≥2 waypoints (maxSeg≥3)

    Reduction vs _reachable_segments:
      Old: up to |V|² = 10000 pairs per timestep → OOM
      New: ~2×|V*(d,t)| pairs per demand → 50× fewer variables
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
            # Valid waypoints for this demand at this time step
            valid_wps = (reach_from[src] & reach_to[tgt]) - {src, tgt}
            # Type 1: direct segment (0 waypoints)
            if tgt in reach_from[src]:
                segs.add((src, tgt))
            # Type 2: (src→w) and (w→tgt) — 1 waypoint
            if max_wps >= 1:
                for w in valid_wps:
                    segs.add((src, w))
                    segs.add((w,   tgt))
            # Type 3: (w1→w2) — intermediate hop, ≥2 waypoints
            if max_wps >= 2:
                for w1 in valid_wps:
                    for w2 in valid_wps:
                        if w2 != w1 and w2 in reach_from[w1]:
                            segs.add((w1, w2))
        valid[t] = segs
    total  = sum(len(v) for v in valid.values())
    naive  = len(inst.nodes) ** 2 * len(inst.T)
    factor = naive // max(total, 1)
    print(f"  → {total:,} useful segment slots  "
          f"(naive: {naive:,} → reduction ×{factor})")
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
###  Main solver
#####################################################

class SequentialLexSolver:
    """
    Role: Implements Algorithm A: Lex-Minimax Sequential MILP for T-ASR.
    
    Parameters:
            inst       : parsed Instance  (inst.max_seg read automatically)
            r          : ECMP split coefficients  {t: SplitCoeffs}
            time_limit : Gurobi time limit per sub-problem (seconds)
            mip_gap    : relative MIP optimality gap
            max_iter   : stop after this many iterations (default = N = |A|·|T|)
            verbose    : print Gurobi log
    """

    def __init__(
        self,
        inst: Instance,
        r: Dict[int, SplitCoeffs],
        time_limit: float = 300.0,
        mip_gap: float = 1e-8,
        epsilon: float = 1e-8,
        max_iter: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        self.inst       = inst
        self.r          = r
        self.time_limit = time_limit
        self.mip_gap    = mip_gap
        self.epsilon    = epsilon
        self.verbose    = verbose

        self.components: List[Tuple[int, int]] = [
            (a, t) for t in inst.T for a in inst.arc_ids
        ]
        self.N        = len(self.components)
        self.max_iter = max_iter if max_iter is not None else self.N

        print(f"  Pre-computing useful segments "
              f"(maxSeg={inst.max_seg}, |V|={len(inst.nodes)}, "
              f"|D|={len(inst.demands)}, |T|={len(inst.T)}) …")
        self.valid_segs = _useful_segments(inst)


    def solve(self) -> Tuple[Dict, List[float]]:
        """
        Role: Run the sequential lex-minimax algorithm.

        Returns:
            waypoints : {(demand_id, t): [w1, w2, …]}
            z_stars   : [z*_1, z*_2, …]
        """
        inst = self.inst
        N    = self.N
        t0   = time.perf_counter()
        
        ### Build Gurobi model 
        m = gp.Model("T-ASR")
        m.Params.OutputFlag = int(self.verbose)
        m.Params.MIPGap     = self.mip_gap

        ### x[d_id, t, i, j] — binary segment indicator
        x_keys = [
            (d_id, t, i, j)
            for (d_id, _, _, _) in inst.demands
            for t in inst.T
            for (i, j) in self.valid_segs[t]
        ]
        x = m.addVars(x_keys, vtype=GRB.BINARY, name="x")
        print(f"\n  x variables : {len(x_keys):,}")

        ### delta[d_id, t, i, j] — linearises |x^dt - x^{d,t-1}|
        delta_keys = [
            (d_id, t, i, j)
            for (d_id, _, _, _) in inst.demands
            for t in inst.T_star
            for (i, j) in (self.valid_segs[t] | self.valid_segs[t - 1])
        ]
        delta = m.addVars(delta_keys, vtype=GRB.BINARY, name="delta")


        ### Load expressions λ(arc, t)
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
        z_stars:        List[float]     = []
        z1_val:         Optional[float] = None
        best_waypoints: Optional[Dict]  = None
        z_vars:         List[gp.Var]    = []
        iter_times:     List[float]     = []   # time per iteration

        print(f"\n  {'k':>5} {'z*_k':>10} {'iter_time':>10} "
              f"{'elapsed':>10} {'Δz*':>10} {'status':>8}")
        print(f"  {'-'*5} {'-'*10} {'-'*10} "
              f"{'-'*10} {'-'*10} {'-'*8}")
        for k in range(1, self.max_iter + 1):
            iter_t0 = time.perf_counter()
            ub_z = z_stars[-1] if z_stars else GRB.INFINITY
            z_k  = m.addVar(lb=0.0, ub=ub_z, name=f"z{k}")
            z_vars.append(z_k)
            if k == 1:
                ### (P_1): all loads ≤ z_1
                for idx in range(N):
                    m.addConstr(lam[idx] <= z_k, name=f"C1_{idx}")

                # NB: NO z1bound constraints at k=2.
                # After k=1, z_k is fixed via z_k.lb = z_k.ub = z*_1.
                # The constraints C1_{idx} already enforce λ_i ≤ z*_1.

            else:
                # (P_k): at most k-1 loads may exceed z_k
                # Linearised: λ_i ≤ z_k + z*_1 · y^(k)_i
                y_k = m.addVars(N, vtype=GRB.BINARY, name=f"y{k}")

                for idx in range(N):
                    m.addConstr(
                        lam[idx] <= z_k + z1_val * y_k[idx],
                        name=f"Ck_{k}_{idx}",
                    )
                m.addConstr(
                    gp.quicksum(y_k[idx] for idx in range(N)) <= k - 1,
                    name=f"Nk_{k}",
                ) 
            m.setObjective(z_k, GRB.MINIMIZE)
            m.Params.TimeLimit = self.time_limit
            m.update()
            m.optimize()
            status = m.Status
            if status == GRB.INFEASIBLE:
                print(f"  [k={k}] INFEASIBLE — stopping.")
                m.computeIIS()
                m.write("infeasible.ilp")
                break
            
            if status not in (GRB.OPTIMAL, GRB.TIME_LIMIT,
                               GRB.NODE_LIMIT, GRB.SOLUTION_LIMIT):
                print(f"  [k={k}] Unexpected status {status} — stopping.")
                break

            if m.SolCount == 0:
                print(f"  [k={k}] No integer solution found — "
                      "try increasing time_limit.")
                break

            z_k_val = z_k.X
            z_stars.append(z_k_val)
            
            if k == 1:
                z1_val = z_k_val

            iter_time    = time.perf_counter() - iter_t0
            total_elapsed = time.perf_counter() - t0
            iter_times.append(iter_time)

            # Marginal gain vs previous iteration
            delta_z = ""
            if len(z_stars) >= 1:
                dz = z_stars[-2] - z_k_val if len(z_stars) >= 2 else 0.0
                delta_z = f"{dz:+.6f}"

            # Progress bar
            filled = int(20 * k / self.max_iter)
            bar    = '█' * filled + '░' * (20 - filled)
            pct    = 100 * k / self.max_iter

            # Status label
            status_label = {
                2: "OPTIMAL",
                9: "TIME_LIM",
                3: "INFEASIB",
            }.get(status, f"st={status}")

            print(
                f"  {k:>5} {z_k_val:>10.6f} {iter_time:>9.1f}s "
                f"{total_elapsed:>9.1f}s {delta_z:>10} {status_label:>8}"
                f"  [{bar}] {pct:.0f}%"
            )
            best_waypoints = self._extract_waypoints(x)

            z_k.lb = z_k_val - self.epsilon
            z_k.ub = z_k_val + self.epsilon
            m.update()

            if z_k_val < 1e-9:
                print("  All loads are 0 — converged.")
                break

        total_elapsed = time.perf_counter() - t0
        print(f"\n  Done.  {len(z_stars)} iterations in {total_elapsed:.1f}s")
        if z_stars:
            print(f"  MLU (z*_1) = {z_stars[0]:.6f}")

        return best_waypoints, z_stars

    #####################################################
    ###  Extract solution
    #####################################################
    def _extract_waypoints(self, x: gp.tupledict) -> Dict:
        """Convert x^dt_ij binary values to {(demand_id, t): [w1, w2, …]}."""
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
    print(f"  Solution written → {out_path}")