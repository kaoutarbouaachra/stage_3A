#!/usr/bin/env python3
"""
cg_exact.master
===============

Pair-path restricted master LP, root-node column generation with exact
pricing, and a pool-restricted integer solve.

    solve_lp(rank_values=None)            solves the MLU LP.
    solve_lp(rank_values=[z1,...,z_{k-1}]) solves the rank-k STELA LP.
    solve_integer_master(...)             solves the integer master over the
                                          currently generated pool.

The inherited-rank big-M is the tightened STELA value (z1* - zj*), and the
current-rank big-M is z1*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .data import (
    GRB,
    Arc,
    ArcTime,
    Column,
    DemandId,
    MasterInfeasibleError,
    Period,
    PricingInfeasibleError,
    TASRData,
    TASRError,
    TrajectoryKey,
    _require_gurobi,
    gp,
    safe_name,
)
from .pricing import ExactPairPathPricer


@dataclass
class MasterResult:
    model: Any
    objective_value: float
    z_value: float
    sigma: Dict[DemandId, float]
    aggregate_mu: Dict[ArcTime, float]
    psi: float
    xi_values: Dict[Tuple[DemandId, TrajectoryKey], float]
    y_values: Dict[Any, float] = field(default_factory=dict)


@dataclass
class ColumnGenerationLogEntry:
    iteration: int
    objective_value: float
    added_columns: int
    min_reduced_cost: float
    pricing_values: Dict[DemandId, float]


@dataclass
class ColumnGenerationResult:
    objective_value: float
    z_value: float
    iterations: int
    columns: Dict[DemandId, List[Column]]
    xi_values: Dict[Tuple[DemandId, TrajectoryKey], float]
    log: List[ColumnGenerationLogEntry]
    master_result: MasterResult


class PairPathColumnGeneration:
    """Root-node column generation with corrected exact pricing."""

    def __init__(self, data: TASRData,
                 initial_columns: Optional[Mapping[DemandId, Iterable[Column]]] = None,
                 output: bool = False,
                 pricing_output: bool = False,
                 pricing_tol: float = 1e-7,
                 max_iterations: int = 10000,
                 master_params: Optional[Mapping[str, Any]] = None,
                 pricing_params: Optional[Mapping[str, Any]] = None) -> None:
        _require_gurobi()
        self.data = data
        self.output = output
        self.pricing_output = pricing_output
        self.pricing_tol = pricing_tol
        self.max_iterations = max_iterations
        self.master_params = dict(master_params or {})
        self.pricing_params = dict(pricing_params or {})
        self.columns: Dict[DemandId, List[Column]] = {d.id: [] for d in data.demands}
        self.column_keys: Dict[DemandId, Set[TrajectoryKey]] = {d.id: set() for d in data.demands}

        if initial_columns is None:
            self._build_basic_initial_pool()
        else:
            for d_id, cols in initial_columns.items():
                for col in cols:
                    self.add_column(col)

    def add_column(self, col: Column) -> bool:
        if col.demand_id not in self.columns:
            raise ValueError(f"Unknown demand id {col.demand_id!r}.")
        key = col.key()
        if key in self.column_keys[col.demand_id]:
            return False
        self.columns[col.demand_id].append(col)
        self.column_keys[col.demand_id].add(key)
        return True

    def solve_lp(self, rank_values: Optional[Sequence[float]] = None) -> ColumnGenerationResult:
        rank_values = list(rank_values or [])
        log: List[ColumnGenerationLogEntry] = []
        pricer = ExactPairPathPricer(self.data, output=self.pricing_output,
                                     gurobi_params=self.pricing_params)
        last_master: Optional[MasterResult] = None

        for iteration in range(1, self.max_iterations + 1):
            master = self._solve_master(relax_integrality=True, rank_values=rank_values)
            last_master = master
            rho = max(0.0, -float(master.psi))
            added = 0
            min_rc = math.inf
            pricing_values: Dict[DemandId, float] = {}

            for d in self.data.demands:
                theta = self.data.segment_costs(d.id, master.aggregate_mu)
                forbidden: Set[TrajectoryKey] = set()
                while True:
                    pricing = pricer.solve(demand_id=d.id, theta=theta, rho=rho,
                                           sigma=master.sigma[d.id],
                                           forbidden_trajectories=forbidden,
                                           use_cutoff=False,
                                           pricing_tol=self.pricing_tol)
                    pricing_values[d.id] = pricing.value
                    min_rc = min(min_rc, pricing.reduced_cost)
                    if pricing.reduced_cost >= -self.pricing_tol:
                        break
                    if pricing.p0 is None or pricing.p1 is None:
                        break
                    col = self.data.make_column(d.id, pricing.p0, pricing.p1)
                    if self.add_column(col):
                        added += 1
                        break
                    forbidden.add(col.key())
                    if len(forbidden) > len(self.columns[d.id]) + 5:
                        break

            log.append(ColumnGenerationLogEntry(iteration, master.objective_value,
                                                added, min_rc, pricing_values))
            if self.output:
                print(f"[CG] iter={iteration:04d} obj={master.objective_value:.10g} "
                      f"added={added} min_rc={min_rc:.3e}")
            if added == 0:
                return ColumnGenerationResult(master.objective_value, master.z_value,
                                              iteration, self.columns, master.xi_values,
                                              log, master)

        raise TASRError(f"Column generation did not converge in {self.max_iterations} iterations.")

    def solve_integer_master(self, rank_values: Optional[Sequence[float]] = None,
                             time_limit: Optional[float] = None,
                             mip_gap: Optional[float] = None) -> MasterResult:
        extra: Dict[str, Any] = {}
        if time_limit is not None:
            extra["TimeLimit"] = time_limit
        if mip_gap is not None:
            extra["MIPGap"] = mip_gap
        old = self.master_params.copy()
        self.master_params.update(extra)
        try:
            return self._solve_master(relax_integrality=False, rank_values=list(rank_values or []))
        finally:
            self.master_params = old

    def _solve_master(self, relax_integrality: bool, rank_values: Sequence[float]) -> MasterResult:
        data = self.data
        m = gp.Model("pair_path_restricted_master")
        m.Params.OutputFlag = 1 if self.output else 0
        for key, val in self.master_params.items():
            setattr(m.Params, key, val)

        z = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, obj=1.0, name="z")
        xi: Dict[Tuple[DemandId, int], gp.Var] = {}
        for d in data.demands:
            if not self.columns[d.id]:
                raise MasterInfeasibleError(f"No column in pool for demand {d.id!r}.")
            for idx, _ in enumerate(self.columns[d.id]):
                vtype = GRB.CONTINUOUS if relax_integrality else GRB.BINARY
                xi[(d.id, idx)] = m.addVar(lb=0.0, ub=1.0, obj=0.0, vtype=vtype,
                                           name=f"xi[{safe_name(d.id)},{idx}]")

        rank_values = list(rank_values)
        k = len(rank_values) + 1
        z1 = rank_values[0] if rank_values else None
        y_inh: Dict[Tuple[int, ArcTime], gp.Var] = {}
        y_cur: Dict[ArcTime, gp.Var] = {}
        if k >= 2:
            assert z1 is not None
            y_type = GRB.CONTINUOUS if relax_integrality else GRB.BINARY
            for j in range(2, k):
                for at in data.arc_times:
                    y_inh[(j, at)] = m.addVar(lb=0.0, ub=1.0, vtype=y_type,
                                              name=f"y_inh[{j},{safe_name(at)}]")
            for at in data.arc_times:
                y_cur[at] = m.addVar(lb=0.0, ub=1.0, vtype=y_type,
                                     name=f"y_cur[{safe_name(at)}]")
        m.update()

        selection_constr: Dict[DemandId, gp.Constr] = {}
        for d in data.demands:
            selection_constr[d.id] = m.addConstr(
                gp.quicksum(xi[(d.id, idx)] for idx in range(len(self.columns[d.id]))) == 1,
                name=f"select[{safe_name(d.id)}]"
            )

        budget_expr = gp.LinExpr()
        for d in data.demands:
            for idx, col in enumerate(self.columns[d.id]):
                budget_expr += float(col.beta) * xi[(d.id, idx)]
        budget_constr = m.addConstr(budget_expr <= float(data.budget), name="budget")

        load_expr: Dict[ArcTime, gp.LinExpr] = {}
        for at in data.arc_times:
            expr = gp.LinExpr()
            for d in data.demands:
                for idx, col in enumerate(self.columns[d.id]):
                    coeff = float(col.gamma.get(at, 0.0))
                    if coeff:
                        expr += coeff * xi[(d.id, idx)]
            load_expr[at] = expr

        load_constrs_by_at: Dict[ArcTime, List[gp.Constr]] = {at: [] for at in data.arc_times}
        if k == 1:
            for at in data.arc_times:
                con = m.addConstr(load_expr[at] <= z, name=f"load_rank1[{safe_name(at)}]")
                load_constrs_by_at[at].append(con)
        else:
            assert z1 is not None
            for at in data.arc_times:
                con = m.addConstr(load_expr[at] <= float(z1), name=f"global_bound[{safe_name(at)}]")
                load_constrs_by_at[at].append(con)
            for j in range(2, k):
                z_j = float(rank_values[j - 1])
                M_j = max(0.0, float(z1) - z_j)
                for at in data.arc_times:
                    con = m.addConstr(load_expr[at] <= z_j + M_j * y_inh[(j, at)],
                                      name=f"inh_rank[{j},{safe_name(at)}]")
                    load_constrs_by_at[at].append(con)
                m.addConstr(gp.quicksum(y_inh[(j, at)] for at in data.arc_times) <= j - 1,
                            name=f"inh_card[{j}]")
            for at in data.arc_times:
                con = m.addConstr(load_expr[at] <= z + float(z1) * y_cur[at],
                                  name=f"cur_rank[{k},{safe_name(at)}]")
                load_constrs_by_at[at].append(con)
            m.addConstr(gp.quicksum(y_cur[at] for at in data.arc_times) <= k - 1,
                        name=f"cur_card[{k}]")
            m.addConstr(z <= float(rank_values[-1]), name="monotonicity")

        m.ModelSense = GRB.MINIMIZE
        m.optimize()

        if m.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
            try:
                m.computeIIS()
                iis_rows = [c.ConstrName for c in m.getConstrs() if c.IISConstr]
            except Exception:
                iis_rows = []
            raise MasterInfeasibleError(
                "Restricted master is infeasible. Provide a feasible initial pool or check the budget. "
                "IIS rows: " + ", ".join(iis_rows[:20])
            )
        if m.Status != GRB.OPTIMAL:
            raise TASRError(f"Master did not solve to optimality. Gurobi status={m.Status}.")

        if relax_integrality:
            sigma = {d.id: float(selection_constr[d.id].Pi) for d in data.demands}
            psi = float(budget_constr.Pi)
            aggregate_mu = {at: sum(float(con.Pi) for con in load_constrs_by_at[at])
                            for at in data.arc_times}
        else:
            sigma = {d.id: float("nan") for d in data.demands}
            psi = float("nan")
            aggregate_mu = {at: float("nan") for at in data.arc_times}

        xi_values: Dict[Tuple[DemandId, TrajectoryKey], float] = {}
        for d in data.demands:
            for idx, col in enumerate(self.columns[d.id]):
                val = float(xi[(d.id, idx)].X)
                if abs(val) > 1e-10:
                    xi_values[(d.id, col.key())] = val

        y_values: Dict[Any, float] = {}
        if k >= 2:
            for key, var in y_inh.items():
                if abs(var.X) > 1e-10:
                    y_values[("inh", key)] = float(var.X)
            for key, var in y_cur.items():
                if abs(var.X) > 1e-10:
                    y_values[("cur", key)] = float(var.X)

        return MasterResult(m, float(m.ObjVal), float(z.X), sigma, aggregate_mu,
                            psi, xi_values, y_values)

    def _build_basic_initial_pool(self) -> None:
        pricer = ExactPairPathPricer(self.data, output=self.pricing_output,
                                     gurobi_params=self.pricing_params)
        for d in self.data.demands:
            theta = {
                0: {e: 0.0 for e in self.data.feasible_real_segments(d.id, 0)},
                1: {e: 0.0 for e in self.data.feasible_real_segments(d.id, 1)},
            }
            pricing = pricer.solve(d.id, theta=theta, rho=1.0, sigma=0.0)
            if pricing.p0 is None or pricing.p1 is None:
                raise PricingInfeasibleError(f"Could not create initial pair-path for {d.id!r}.")
            self.add_column(self.data.make_column(d.id, pricing.p0, pricing.p1))