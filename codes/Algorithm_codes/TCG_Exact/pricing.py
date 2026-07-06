#!/usr/bin/env python3
"""
cg_exact.pricing
================

Exact corrected pair-path pricing MIP (layered formulation).

The corrected formulation implements the trail constraint

    sum_{ell=1}^K y[t, ell, e] <= 1        for every real segment e, period t,

which makes the layer variables, the binary usage indicators x, and the
pricing objective consistent: each real segment appears in at most one layer
of a period path. Without it, the load term double-counts segments that appear
in several layers, and the equivalence with (PP) fails.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

from .data import (
    GRB,
    Arc,
    ArcTime,
    DemandId,
    Path,
    Period,
    PricingInfeasibleError,
    PricingNotOptimalError,
    Segment,
    TASRData,
    TASRError,
    TrajectoryKey,
    _require_gurobi,
    gp,
    padded_path,
    safe_name,
)


@dataclass
class PricingResult:
    demand_id: DemandId
    value: float
    reduced_cost: float
    p0: Optional[Path]
    p1: Optional[Path]
    status: int
    runtime: float
    explored_nodes: Optional[float] = None

    @property
    def has_negative_reduced_cost(self) -> bool:
        return self.reduced_cost < 0.0 and self.p0 is not None and self.p1 is not None


class ExactPairPathPricer:
    """Exact corrected pair-path pricing MIP."""

    def __init__(
        self,
        data: TASRData,
        output: bool = False,
        mip_gap: float = 0.0,
        feasibility_tol: float = 1e-9,
        int_feas_tol: float = 1e-9,
        time_limit: Optional[float] = None,
        gurobi_params: Optional[Mapping[str, Any]] = None,
    ) -> None:
        _require_gurobi()
        self.data = data
        self.output = output
        self.mip_gap = mip_gap
        self.feasibility_tol = feasibility_tol
        self.int_feas_tol = int_feas_tol
        self.time_limit = time_limit
        self.gurobi_params = dict(gurobi_params or {})

    def solve(
        self,
        demand_id: DemandId,
        theta: Mapping[Period, Mapping[Segment, float]],
        rho: float,
        sigma: float = 0.0,
        forbidden_trajectories: Optional[Iterable[TrajectoryKey]] = None,
        forced_segment_values: Optional[Mapping[Tuple[Period, Segment], int]] = None,
        use_cutoff: bool = False,
        pricing_tol: float = 1e-7,
    ) -> PricingResult:
        if rho < 0.0 and rho > -1e-10:
            rho = 0.0
        if rho < 0.0:
            raise ValueError(f"rho must be nonnegative, got {rho}.")

        data = self.data
        K = data.max_segments
        d = data.demand_by_id[demand_id]
        s = d.source
        r = d.target
        dummy = (r, r)

        real_E: Dict[Period, Set[Segment]] = {}
        layer_E: Dict[Period, Set[Segment]] = {}
        for t in data.periods:
            E_t = set(data.feasible_real_segments(demand_id, t))
            E_t.discard(dummy)
            real_E[t] = {e for e in E_t if e[0] != r}
            layer_E[t] = set(real_E[t]) | {dummy}

        E_union = set(real_E[0]) | set(real_E[1])
        if not E_union:
            raise PricingInfeasibleError(f"No real feasible segments for demand {demand_id!r}.")

        m = gp.Model(f"pricing_{safe_name(demand_id)}")
        m.Params.OutputFlag = 1 if self.output else 0
        m.Params.MIPGap = self.mip_gap
        m.Params.FeasibilityTol = self.feasibility_tol
        m.Params.IntFeasTol = self.int_feas_tol
        if self.time_limit is not None:
            m.Params.TimeLimit = self.time_limit
        for key, val in self.gurobi_params.items():
            setattr(m.Params, key, val)
        if use_cutoff:
            m.Params.Cutoff = float(sigma) - float(pricing_tol)

        y: Dict[Tuple[Period, int, Segment], gp.Var] = {}
        for t in data.periods:
            for ell in range(K):
                for e in layer_E[t]:
                    y[(t, ell, e)] = m.addVar(vtype=GRB.BINARY, name=f"y[{t},{ell},{safe_name(e)}]")

        x: Dict[Tuple[Period, Segment], gp.Var] = {}
        for t in data.periods:
            for e in E_union:
                x[(t, e)] = m.addVar(vtype=GRB.BINARY, name=f"x[{t},{safe_name(e)}]")

        h: Dict[Segment, gp.Var] = {
            e: m.addVar(vtype=GRB.BINARY, name=f"h[{safe_name(e)}]") for e in E_union
        }
        m.update()

        # (3) exactly one segment per layer
        for t in data.periods:
            for ell in range(K):
                m.addConstr(gp.quicksum(y[(t, ell, e)] for e in layer_E[t]) == 1,
                            name=f"one_segment[t={t},ell={ell}]")

        # (4) first segment leaves source
        for t in data.periods:
            m.addConstr(gp.quicksum(y[(t, 0, e)] for e in layer_E[t] if e[0] == s) == 1,
                        name=f"start[t={t}]")

        # (5) continuity
        node_set = set(data.nodes)
        for t in data.periods:
            for e in layer_E[t]:
                node_set.add(e[0]); node_set.add(e[1])
        for t in data.periods:
            for ell in range(K - 1):
                for v in node_set:
                    incoming = gp.quicksum(y[(t, ell, e)] for e in layer_E[t] if e[1] == v)
                    outgoing = gp.quicksum(y[(t, ell + 1, e)] for e in layer_E[t] if e[0] == v)
                    m.addConstr(incoming == outgoing, name=f"flow[t={t},ell={ell},v={safe_name(v)}]")

        # (6) last layer reaches target
        for t in data.periods:
            m.addConstr(gp.quicksum(y[(t, K - 1, e)] for e in layer_E[t] if e[1] == r) == 1,
                        name=f"end[t={t}]")

        # (7) corrected trail constraint: no repeated real segment
        for t in data.periods:
            for e in real_E[t]:
                m.addConstr(gp.quicksum(y[(t, ell, e)] for ell in range(K)) <= 1,
                            name=f"trail[t={t},e={safe_name(e)}]")

        # (8)-(10) x is the binary usage indicator
        for t in data.periods:
            for e in E_union:
                if e not in real_E[t]:
                    m.addConstr(x[(t, e)] == 0, name=f"x_infeasible[t={t},e={safe_name(e)}]")
                    continue
                for ell in range(K):
                    m.addConstr(x[(t, e)] >= y[(t, ell, e)],
                                name=f"x_ge_y[t={t},ell={ell},e={safe_name(e)}]")
                m.addConstr(x[(t, e)] <= gp.quicksum(y[(t, ell, e)] for ell in range(K)),
                            name=f"x_le_sum_y[t={t},e={safe_name(e)}]")

        # Optional branch restrictions (for branch-and-price).
        if forced_segment_values:
            for (t, e), val in forced_segment_values.items():
                if int(val) not in (0, 1):
                    raise ValueError("forced_segment_values must contain 0/1 values.")
                if e not in E_union:
                    if int(val) == 1:
                        raise PricingInfeasibleError(f"Cannot force unavailable segment {e!r} at period {t}.")
                    continue
                m.addConstr(x[(t, e)] == int(val), name=f"forced_x[t={t},e={safe_name(e)}]")

        # (11)-(12) h_e = |x_e^0 - x_e^1|
        for e in E_union:
            m.addConstr(h[e] >= x[(0, e)] - x[(1, e)], name=f"h_pos[e={safe_name(e)}]")
            m.addConstr(h[e] >= x[(1, e)] - x[(0, e)], name=f"h_neg[e={safe_name(e)}]")
            m.addConstr(h[e] <= x[(0, e)] + x[(1, e)], name=f"h_ub1[e={safe_name(e)}]")
            m.addConstr(h[e] <= 2 - x[(0, e)] - x[(1, e)], name=f"h_ub2[e={safe_name(e)}]")

        # Forbid exact layer sequences, useful if a duplicate column is returned.
        if forbidden_trajectories:
            for idx, traj in enumerate(forbidden_trajectories):
                p0, p1 = traj
                padded = {0: padded_path(tuple(p0), r, K), 1: padded_path(tuple(p1), r, K)}
                terms = []
                skip = False
                for t in data.periods:
                    for ell, e in enumerate(padded[t]):
                        if e not in layer_E[t]:
                            skip = True
                            break
                        terms.append(1 - y[(t, ell, e)])
                    if skip:
                        break
                if not skip:
                    m.addConstr(gp.quicksum(terms) >= 1, name=f"nogood_traj[{idx}]")

        # Corrected objective: load term is counted by y, not by segment multiplicity in a walk.
        obj = gp.LinExpr()
        for t in data.periods:
            theta_t = theta.get(t, {})
            for ell in range(K):
                for e in real_E[t]:
                    obj += float(theta_t.get(e, 0.0)) * y[(t, ell, e)]
        obj += float(rho) * gp.quicksum(h[e] for e in E_union)
        m.setObjective(obj, GRB.MINIMIZE)
        m.optimize()

        if m.Status == GRB.CUTOFF:
            return PricingResult(demand_id, float("inf"), float("inf"), None, None,
                                 m.Status, float(m.Runtime), getattr(m, "NodeCount", None))
        if m.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
            raise PricingInfeasibleError(f"Pricing infeasible for demand {demand_id!r}.")
        if m.Status != GRB.OPTIMAL:
            raise PricingNotOptimalError(
                f"Pricing for demand {demand_id!r} not solved to optimality. Status={m.Status}."
            )

        value = float(m.ObjVal)
        reduced_cost = value - float(sigma)
        p0 = self._extract_path(data, demand_id, 0, y, layer_E[0], dummy)
        p1 = self._extract_path(data, demand_id, 1, y, layer_E[1], dummy)
        return PricingResult(demand_id, value, reduced_cost, p0, p1, m.Status,
                             float(m.Runtime), getattr(m, "NodeCount", None))

    def _extract_path(self, data: TASRData, demand_id: DemandId, t: Period,
                      y: Mapping[Tuple[Period, int, Segment], Any],
                      layer_edges: Set[Segment], dummy: Segment) -> Path:
        d = data.demand_by_id[demand_id]
        path: List[Segment] = []
        reached = False
        for ell in range(data.max_segments):
            chosen = [e for e in layer_edges if y[(t, ell, e)].X > 0.5]
            if len(chosen) != 1:
                raise TASRError(f"Extraction error for demand {demand_id!r}, t={t}, layer={ell}.")
            e = chosen[0]
            if e == dummy:
                reached = True
                break
            path.append(e)
            if e[1] == d.target:
                reached = True
                break
        if not reached:
            raise TASRError(f"Extracted path for demand {demand_id!r}, period {t}, does not reach target.")
        return tuple(path)