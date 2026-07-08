"""
Function: model.py
Role: Builds the ALEXA Gurobi model -- the compact formulation with explicit
permutation-matrix sorting (the assignment-based mechanism that defines
ALEXA, as opposed to STELA's threshold-exceedance rows).

"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

import gurobipy as gp
from gurobipy import GRB

from core.constraints import (
    add_budget,
    add_flow_conservation,
    add_segment_count,
    build_load_exprs,
)
from core.ecmp import SplitCoeffs
from core.instance import Instance

from .bigm import compute_utilization_bigm
from .segments import useful_segments

ArcTime = Tuple[int, int]


class AlexaModel:
    """
    Container for the built Gurobi model and everything the sequential
    solve loop needs afterwards: the model itself, the sorted-load
    variables L, the assignment variables p, the components order (so L[k]
    and p[.., k] line up with a specific (arc, period) meaning only through
    the assignment, never a fixed one), the x variables (for solution
    extraction), and valid_segs (needed to rebuild waypoints).
    """

    def __init__(self, m: gp.Model, x: gp.tupledict, valid_segs: Dict[int, Set[Tuple[int, int]]],
                components: List[ArcTime], L: List[gp.Var], p: Dict[Tuple[ArcTime, int], gp.Var],
                bigm: Dict[ArcTime, float]):
        self.m = m
        self.x = x
        self.valid_segs = valid_segs
        self.components = components
        self.L = L
        self.p = p
        self.bigm = bigm
        self.N = len(components)


def build_alexa_model(inst: Instance, r: Dict[int, SplitCoeffs], verbose: bool = False) -> AlexaModel:
    valid_segs = useful_segments(inst)

    m = gp.Model("ALEXA")
    m.Params.OutputFlag = int(verbose)

    # --- x[d,t,i,j]: routing, only over pruned candidate segments ---------
    x_keys = [
        (d_id, t, i, j)
        for (d_id, _s, _tt, _v) in inst.demands
        for t in inst.T
        for (i, j) in valid_segs[t]
    ]
    x = m.addVars(x_keys, vtype=GRB.BINARY, name="x")
    print(f"\n  x variables : {len(x_keys):,}")

    # --- delta[d,t,i,j]: budget linearization, matches core.constraints ---
    delta_keys = [
        (d_id, t, i, j)
        for (d_id, _s, _tt, _v) in inst.demands
        for t in inst.T_star
        for (i, j) in (valid_segs[t] | valid_segs[t - 1])
    ]
    delta = m.addVars(delta_keys, vtype=GRB.BINARY, name="delta")

    # --- Base constraints, shared with STELA -------------------------------
    add_flow_conservation(m, x, inst, valid_segs)
    add_segment_count(m, x, inst, valid_segs)
    add_budget(m, x, delta, inst, valid_segs)

    # --- Load / utilization expressions, shared with STELA -----------------
    components: List[ArcTime] = [(a, t) for t in inst.T for a in inst.arc_ids]
    lam = build_load_exprs(m, x, inst, r, components, valid_segs)

    # --- Fixed, data-driven big-M (see bigm.py) -----------------------------
    bigm = compute_utilization_bigm(inst, r)
    global_m = max(bigm.values()) if bigm else 1.0

    # --- Sorted-load variables L[0] >= L[1] >= ... >= L[N-1] ----------------
    N = len(components)
    L = [m.addVar(lb=0.0, ub=global_m, name=f"L{k}") for k in range(N)]
    for k in range(N - 1):
        m.addConstr(L[k] >= L[k + 1], name=f"sorted_order_{k}")

    # --- Assignment (permutation) variables p[(a,t), k] ---------------------
    p: Dict[Tuple[ArcTime, int], gp.Var] = {}
    for at in components:
        for k in range(N):
            p[(at, k)] = m.addVar(vtype=GRB.BINARY, name=f"p_{at[0]}_{at[1]}_{k}")

    for at in components:
        m.addConstr(gp.quicksum(p[(at, k)] for k in range(N)) == 1,
                    name=f"rank_assignment_{at[0]}_{at[1]}")
    for k in range(N):
        m.addConstr(gp.quicksum(p[(at, k)] for at in components) == 1,
                    name=f"load_assignment_{k}")

    # --- Big-M linking: L[k] >= lam(a,t) - M[a,t] * (1 - p[(a,t),k]) --------
    for idx, at in enumerate(components):
        m_at = bigm.get(at, global_m)
        for k in range(N):
            m.addConstr(L[k] >= lam[idx] - m_at * (1 - p[(at, k)]),
                        name=f"big_m_linking_{at[0]}_{at[1]}_{k}")

    m.update()
    print(f"  Base model : {m.NumVars:,} vars, {m.NumConstrs:,} constraints, "
          f"{m.NumNZs:,} nonzeros\n")

    return AlexaModel(m, x, valid_segs, components, L, p, bigm)
