
"""
Function: constraints.py
Role: Add Gurobi constraints for the T-ASR problem.

Each function takes the Gurobi model `m`, the decision variables, and
the instance data, and mutates `m` in-place (adds constraints).

Variables:
  -x[d_id, t, i, j]  : binary, 1 if demand d at time t uses segment (i→j)
  -delta[d_id, t, i, j]: continuous [0,1], linearises |x^dt_ij - x^{d,t-1}_ij|
"""

from typing import Dict, List, Set, Tuple
import gurobipy as gp
from gurobipy import GRB
from .instance import Instance
from .ecmp_from_scratch import SplitCoeffs


###############################
### Flow-conservation  (C1)
###############################
def add_flow_conservation(
    m: gp.Model,
    x: gp.tupledict,
    inst: Instance,
    valid_segments: Dict[int, Set[Tuple[int, int]]],
) -> None:
    """
    Σ_{j≠i} x^dt_ij  -  Σ_{j≠i} x^dt_ji  =  { +1 if i=s,  -1 if i=t,  0 otherwise }

    Only the arcs that belong to valid_segments[t] are considered, so the
    constraint is only built over actually-reachable segments.
    """
    for (d_id, src, tgt, _) in inst.demands:
        for t in inst.T:
            segs_t = valid_segments[t]
            for i in inst.nodes:
                out_flow = gp.quicksum(
                    x[d_id, t, i, j]
                    for j in inst.nodes
                    if j != i and (i, j) in segs_t
                )
                in_flow = gp.quicksum(
                    x[d_id, t, j, i]
                    for j in inst.nodes
                    if j != i and (j, i) in segs_t
                )
                rhs = 1 if i == src else (-1 if i == tgt else 0)
                m.addConstr(out_flow - in_flow == rhs,
                            name=f"flow_d{d_id}_t{t}_i{i}")


###############################
### Maximum number of segments  (C2)
###############################
def add_segment_count(
    m: gp.Model,
    x: gp.tupledict,
    inst: Instance,
    valid_segments: Dict[int, Set[Tuple[int, int]]],
) -> None:
    """
    Σ_{i,j} x^dt_ij  ≤  maxSeg    ∀ d ∈ D, ∀ t ∈ T
    """
    for (d_id, _, _, _) in inst.demands:
        for t in inst.T:
            segs_t = valid_segments[t]
            m.addConstr(
                gp.quicksum(
                    x[d_id, t, i, j]
                    for (i, j) in segs_t
                ) <= inst.max_seg,
                name=f"maxseg_d{d_id}_t{t}",
            )


###############################
### Budget  (C4)
###############################
def add_budget(
    m: gp.Model,
    x: gp.tupledict,
    delta: gp.tupledict,
    inst: Instance,
    valid_segments: Dict[int, Set[Tuple[int, int]]],
) -> None:
    """
    Objective:Linearises the budget constraint

    Auxiliary variable delta[d, t, i, j] ≥ |x^dt_ij − x^{d,t-1}_ij|.
    Since both x are binary the LP relaxation of delta is tight.

    """
    for t in inst.T_star:
        kappa = inst.budget.get(t, None)
        if kappa is None:           # infinite budget at this step → skip
            continue
        
        # Union of segments present at t-1 OR t (others trivially have Δ=0)
        segs_prev = valid_segments[t - 1]
        segs_curr = valid_segments[t]
        affected  = segs_prev | segs_curr

        total = gp.LinExpr()
        for (d_id, _, _, _) in inst.demands:
            for (i, j) in affected:
                x_curr = x.get((d_id, t,     i, j), None)
                x_prev = x.get((d_id, t - 1, i, j), None)
                d_var  = delta.get((d_id, t, i, j), None)
                if d_var is None:
                    continue
                if x_curr is not None and x_prev is not None:
                    m.addConstr(d_var >= x_curr - x_prev,
                                name=f"dpos_d{d_id}_t{t}_i{i}_j{j}")
                    m.addConstr(d_var >= x_prev - x_curr,
                                name=f"dneg_d{d_id}_t{t}_i{i}_j{j}")
                elif x_curr is not None and x_prev is None:
                    m.addConstr(d_var >= x_curr,
                                name=f"dpos_d{d_id}_t{t}_i{i}_j{j}")
                    m.addConstr(d_var >= -x_curr,
                                name=f"dneg_d{d_id}_t{t}_i{i}_j{j}")

                elif x_prev is not None and x_curr is None:
                    m.addConstr(d_var >= x_prev,
                                name=f"dpos_d{d_id}_t{t}_i{i}_j{j}")
                    m.addConstr(d_var >= -x_prev,
                                name=f"dneg_d{d_id}_t{t}_i{i}_j{j}")
                total += d_var

        m.addConstr(total <= kappa, name=f"budget_t{t}")


###############################
### Load expressions  (C3)
###############################

def build_load_exprs(
    m: gp.Model,
    x: gp.tupledict,
    inst: Instance,
    r: Dict[int, SplitCoeffs],
    components: List[Tuple[int, int]],
    valid_segments: Dict[int, Set[Tuple[int, int]]],
) -> List[gp.LinExpr]:
    """
    λ(a, t) = Σ_d Σ_{(u,v)} r(u,v,a,t) · ν(d,t) · x^dt_{uv} / c(a)

    Returns a list of N = |A|×|T| linear expressions, one per component
    (arc_id, t), in the same order as `components`.
    """
    lam: List[gp.LinExpr] = []

    for (arc_id, t) in components:
        cap  = inst.links[arc_id]['capacity']
        expr = gp.LinExpr()

        for (d_id, _src, _tgt, vols) in inst.demands:
            vol = vols[t]
            if vol == 0.0:
                continue
            segs_t = valid_segments[t]
            for (u, v) in segs_t:
                coeff = r[t].get((u, v, arc_id), 0.0)
                if coeff < 1e-12:
                    continue
                x_var = x.get((d_id, t, u, v))
                if x_var is None:
                    continue
                expr.addTerms(coeff * vol / cap, x_var)

        lam.append(expr)

    return lam